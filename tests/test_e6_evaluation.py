import ast
import json
from pathlib import Path
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
    assert metadata["config"]["eval"]["goal_offset_steps"] == 25
    assert len(goal_rows) == cfg.eval.num_eval
    assert metadata["metrics"]["episode_successes"] == [True] * cfg.eval.num_eval


def test_supervisor_projection_excludes_provider_queue_delay(tmp_path, monkeypatch):
    # Execute the shipped body without importing Modal decorators or contacting its service.
    module = ast.parse((ROOT / "modal_app.py").read_text())
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_e6_supervisor")
    function.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                   str(ROOT / "modal_app.py"), "exec")
    now, calls = [0.], []

    class Call:
        def __init__(self, job):
            self.job = job
            self.object_id = job["name"]

        def get(self, timeout):
            if self is calls[0]:
                now[0] += 1500  # Provider delay, before either GPU starts working.
            now[0] += 10
            return {"job": self.job, "elapsed_seconds": 10}

        def cancel(self, **kwargs):
            pytest.fail("Successful jobs should not be cancelled")

    def spawn(job):
        call = Call(job)
        calls.append(call)
        return call

    monkeypatch.setattr(runner, "verify_assets", lambda cache: {})
    namespace = {
        "STABLEWM_HOME": tmp_path, "REPO_DIR": ROOT,
        "time": SimpleNamespace(monotonic=lambda: now[0]),
        "volume": SimpleNamespace(reload=lambda: None, commit=lambda: None),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: None),
        "run_e6_job": SimpleNamespace(spawn=spawn),
    }
    exec(code, namespace)
    status = namespace["run_e6_supervisor"]()
    assert status["state"] == "complete"
    assert [call.job for call in calls] == runner.jobs()
    assert len(status["completed"]) == 28
    assert status["spent_gpu_seconds"] == 280
    assert status["projected_gpu_seconds"] == 280
