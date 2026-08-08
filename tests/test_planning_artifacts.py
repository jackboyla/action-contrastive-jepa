from types import SimpleNamespace

import torch
from torch import nn

from eval import RecordingWorldModelPolicy, aggregate_eval_metrics
from render_planning_videos import build_executed_latent_sequence


class FakeActionSpace:
    shape = (1, 2)


class FakeEnv:
    num_envs = 1
    action_space = FakeActionSpace()


class FakeSolver:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"
        self._actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    def configure(self, *, action_space, n_envs, config):
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config

    @property
    def action_dim(self):
        return 2

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def horizon(self):
        return self._config.horizon

    def solve(self, info_dict, init_action=None):
        return self(info_dict, init_action=init_action)

    def __call__(self, info_dict, init_action=None):
        return {"actions": self._actions.clone()}


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def get_cost(self, info_dict, action_candidates):
        batch, samples, horizon = action_candidates.shape[:3]
        context = info_dict["pixels"].shape[2]
        values = torch.arange(
            batch * samples * (context + horizon) * 4, dtype=torch.float32
        )
        info_dict["predicted_emb"] = values.view(batch, samples, context + horizon, 4)
        return torch.zeros(batch, samples)


def test_recording_policy_exports_selected_planning_latents(tmp_path):
    model = FakeModel()
    solver = FakeSolver(model)
    cfg = SimpleNamespace(
        horizon=3,
        receding_horizon=2,
        action_block=1,
        warm_start=True,
    )
    policy = RecordingWorldModelPolicy(solver=solver, config=cfg, model=model)
    policy.set_env(FakeEnv())

    action = policy.get_action(
        {
            "pixels": torch.zeros(1, 1, 3, 2, 2),
            "goal": torch.zeros(1, 1, 3, 2, 2),
        }
    )

    assert action.shape == (1, 2)

    task_dir = tmp_path / "task_000"
    policy.export_task_artifacts([task_dir])
    payload = torch.load(
        task_dir / "planning_artifacts.pt", map_location="cpu", weights_only=False
    )

    assert payload["action_block"] == 1
    assert len(payload["replans"]) == 1
    record = payload["replans"][0]
    assert record["state_plan_latents"].shape == (3, 4)
    assert record["executed_action_plan"].shape == (2, 2)


def test_build_executed_latent_sequence_uses_only_executed_replan_prefixes():
    replans = [
        {
            "state_plan_latents": torch.arange(12, dtype=torch.float32).view(3, 4),
            "executed_action_plan": torch.zeros(2, 2),
        },
        {
            "state_plan_latents": torch.arange(20, dtype=torch.float32).view(5, 4),
            "executed_action_plan": torch.zeros(1, 2),
        },
    ]

    sequence = build_executed_latent_sequence(replans)

    assert sequence.shape == (3, 4)
    assert torch.equal(sequence[:2], replans[0]["state_plan_latents"][:2])
    assert torch.equal(sequence[2:], replans[1]["state_plan_latents"][:1])


def test_aggregate_eval_metrics_concatenates_successes():
    metrics = aggregate_eval_metrics(
        [
            {
                "success_rate": 50.0,
                "episode_successes": torch.tensor([True, False]).numpy(),
                "seeds": None,
            },
            {
                "success_rate": 100.0,
                "episode_successes": torch.tensor([True]).numpy(),
                "seeds": None,
            },
        ]
    )

    assert abs(metrics["success_rate"] - (100.0 * 2 / 3)) < 1e-12
    assert metrics["episode_successes"].tolist() == [True, False, True]
    assert metrics["seeds"] is None
