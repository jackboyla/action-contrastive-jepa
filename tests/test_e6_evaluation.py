import ast
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import eval as evaluation
from experiments.e6 import runner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset_evaluation(tmp_path, monkeypatch):
    with h5py.File(tmp_path / "tiny.h5", "w") as dataset:
        dataset["ep_len"] = np.array([80, 80])
        dataset["ep_offset"] = np.array([0, 80])
        dataset["ep_idx"] = np.repeat(np.arange(2), 80)
        dataset["step_idx"] = np.tile(np.arange(80), 2)
        dataset["row_id"] = np.arange(160)
        dataset["id"] = np.arange(160)+1000
        dataset["action"] = np.arange(320, dtype=np.float32).reshape(160, 2)

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg = compose(config_name="scene", overrides=[
            f"cache_dir={tmp_path}", "eval.dataset_name=tiny", "solver.device=cpu",
            "eval.num_eval=4", "eval.env_batch_size=2", "+eval.progress_bar=false",
            "+eval.random_action_seed=42", f"output.filename={tmp_path / 'outcomes.txt'}",
            f"+output.metadata_path={tmp_path / 'tasks.json'}",
        ])
    actions, goal_rows = [], []

    def create_world(cfg, num_envs):
        # Stub the simulator while retaining real World policy attachment and Gym sampling.
        world = evaluation.swm.World.__new__(evaluation.swm.World)
        world.envs = SimpleNamespace(
            num_envs=num_envs,
            action_space=gym.spaces.Box(-1., 1., shape=(num_envs, 2), dtype=np.float32),
        )

        def evaluate_from_dataset(dataset, *, episodes_idx, start_steps, goal_offset_steps, **kwargs):
            starts = np.asarray(start_steps)
            chunks = dataset.load_chunk(np.asarray(episodes_idx), starts, starts + goal_offset_steps)
            goal_rows.extend(int(chunk["row_id"][-1]) for chunk in chunks)
            actions.append(np.stack([world.policy.get_action(None) for _ in range(3)]))
            return {"episode_successes": np.ones(num_envs, dtype=bool), "seeds": None}

        world.evaluate_from_dataset = evaluate_from_dataset
        return world

    monkeypatch.setattr(evaluation, "create_world", create_world)
    return cfg, actions, goal_rows


def test_dataset_random_actions_replay_with_seed_and_vary_by_batch(dataset_evaluation):
    cfg, actions, _ = dataset_evaluation
    evaluation.run.__wrapped__(cfg)
    first = np.stack(actions)
    actions.clear()
    evaluation.run.__wrapped__(cfg)
    np.testing.assert_array_equal(first, np.stack(actions))
    assert not np.array_equal(first[0], first[1])
    actions.clear()
    cfg.eval.random_action_seed = 43
    evaluation.run.__wrapped__(cfg)
    assert not np.array_equal(first, np.stack(actions))


def test_dataset_metadata_names_the_goal_rows_loaded_from_hdf5(dataset_evaluation):
    cfg, _, goal_rows = dataset_evaluation
    evaluation.run.__wrapped__(cfg)
    metadata = json.loads(Path(cfg.output.metadata_path).read_text())
    assert metadata["goal_rows"] == goal_rows
    assert [c["id"]["goal"] for c in metadata["consumed_chunks"]] == [r+1000 for r in goal_rows]
    assert metadata["config"]["eval"]["goal_offset_steps"] == 25
    assert len(goal_rows) == cfg.eval.num_eval
    assert metadata["metrics"]["episode_successes"] == [True] * cfg.eval.num_eval


@pytest.mark.parametrize("interruption", [None, "completed_cell", "ambiguous_dispatch", "pilot_budget"])
def test_supervisor_projection_and_restart_receipts(tmp_path, monkeypatch, interruption):
    # Execute the shipped body without importing Modal decorators or contacting its service.
    module = ast.parse((ROOT / "modal_app.py").read_text())
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_e6_supervisor")
    function.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                   str(ROOT / "modal_app.py"), "exec")
    now, calls, interrupted = [0.], [], [False]

    class Preempted(BaseException):
        pass

    class Call:
        def __init__(self, job):
            self.job = job
            self.object_id = job["name"]

        def get(self, timeout):
            if interruption == "completed_cell" and self is calls[1] and not interrupted[0]:
                interrupted[0] = True
                raise Preempted()
            if self is calls[0]:
                now[0] += 1500  # Provider delay, before either GPU starts working.
            elapsed = 3600 if interruption == "pilot_budget" else 10
            now[0] += elapsed
            return {"job": self.job, "elapsed_seconds": elapsed}

        def cancel(self, **kwargs):
            pytest.fail("Successful jobs should not be cancelled")

    def spawn(job, attempt, deadline_unix):
        saved = json.loads((tmp_path / "evaluations/test-attempt/status.json").read_text())
        assert deadline_unix == saved['deadline_unix']
        call = Call(job)
        calls.append(call)
        if interruption == "ambiguous_dispatch" and not interrupted[0]:
            interrupted[0] = True
            raise Preempted()
        return call

    def commit():
        status = json.loads((tmp_path / "evaluations/test-attempt/status.json").read_text())
        if interruption == "pilot_budget" and len(status['completed']) == 2 and not interrupted[0]:
            interrupted[0] = True
            raise Preempted()

    monkeypatch.setattr(runner, "verify_assets", lambda cache: {})
    namespace = {
        "STABLEWM_HOME": tmp_path, "REPO_DIR": ROOT,
        "time": SimpleNamespace(time=lambda: now[0]),
        "modal": SimpleNamespace(current_function_call_id=lambda: "supervisor-1",
                                 FunctionCall=SimpleNamespace(from_id=lambda ident: next(
                                     call for call in calls if call.object_id == ident))),
        "volume": SimpleNamespace(reload=lambda: None, commit=commit),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: None),
        "run_e6_job": SimpleNamespace(spawn=spawn),
    }
    exec(code, namespace)
    run = namespace["run_e6_supervisor"]
    if interruption:
        with pytest.raises(Preempted):
            run("test-attempt", prior_gpu_seconds=600)
        before = json.loads((tmp_path / "evaluations/test-attempt/status.json").read_text())
    status = run("test-attempt", prior_gpu_seconds=600)
    if interruption:
        assert status['deadline_unix'] == before['deadline_unix']
    if interruption == "ambiguous_dispatch":
        assert status['state'] == 'failed'
        assert len(calls) == 1
        assert 'Ambiguous dispatch' in status['failures'][0]['error']
        return
    if interruption == "pilot_budget":
        assert status['state'] == 'budget_blocked'
        assert len(calls) == 2
        assert len(status['completed']) == 2
        assert status['projected_gpu_seconds'] > 12 * 3600
        return
    assert status["state"] == "complete"
    assert [call.job for call in calls] == runner.jobs()
    assert len(status["completed"]) == 28
    assert status["spent_gpu_seconds"] == 880
    assert status["projected_gpu_seconds"] == 880

    assert run("test-attempt", prior_gpu_seconds=600) == status
    assert len(calls) == 28
    namespace['modal'].current_function_call_id = lambda: 'different-supervisor'
    with pytest.raises(ValueError, match='another call'):
        run("test-attempt", prior_gpu_seconds=600)


def test_worker_rejects_expired_deadline_before_spawning(tmp_path, monkeypatch):
    def unexpected_spawn(*args, **kwargs):
        pytest.fail('An expired worker must not launch an evaluator')

    monkeypatch.setattr(runner.subprocess, 'Popen', unexpected_spawn)
    with pytest.raises(TimeoutError):
        runner.run_job(runner.jobs()[-1], tmp_path, tmp_path / 'expired',
                       deadline_unix=time.time() - 1)


def test_worker_deadline_stops_real_evaluator_subprocess(tmp_path, monkeypatch):
    # Use a CPU sleep instead of a simulator; exercise actual process launch and deadline cleanup.
    (tmp_path / 'eval.py').write_text("import time\nprint('evaluator started', flush=True)\ntime.sleep(60)\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=''))
    original_popen = runner.subprocess.Popen
    children = []

    def record_spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, 'Popen', record_spawn)
    started = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError)):
        runner.run_job(runner.jobs()[-1], tmp_path, tmp_path / 'bounded',
                       deadline_unix=time.time() + 1)
    assert time.monotonic() - started < 10
    assert len(children) == 1 and children[0].poll() is not None
    assert 'evaluator started' in (tmp_path / 'bounded/eval.log').read_text()
