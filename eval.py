import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import json
import shutil
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hydra
import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from progress import ProgressPrinter
from project_paths import configure_stablewm_home

configure_stablewm_home()

from ogb_prep import _antsoccer_topdown_pixels_from_qpos

import stable_worldmodel as swm


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def _clone_for_analysis(info_dict: dict) -> dict:
    cloned = {}
    for key, value in info_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = deepcopy(value)
    return cloned


def _model_device(model: torch.nn.Module, fallback: str | torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


class RecordingWorldModelPolicy(swm.policy.WorldModelPolicy):
    """World-model policy that exports the selected planning latent rollouts."""

    def __init__(self, *args, model: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self._task_artifacts = None
        self._env_steps = None

    def set_env(self, env):
        _set_eval_policy_env(self, env)
        n_envs = getattr(env, "num_envs", 1)
        self._task_artifacts = [[] for _ in range(n_envs)]
        self._env_steps = np.zeros(n_envs, dtype=np.int64)

    def _analyze_action_plan(
        self,
        info_dict: dict,
        action_plan: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = _model_device(self.model, getattr(self.solver, "device", "cpu"))
        if not torch.is_tensor(action_plan):
            action_plan = torch.as_tensor(action_plan)
        if action_plan.ndim == 3:
            action_plan = action_plan.unsqueeze(1)
        action_plan = action_plan.to(device)

        expanded_info = {}
        for key, value in _clone_for_analysis(info_dict).items():
            if torch.is_tensor(value):
                expanded_info[key] = value.unsqueeze(1)
            elif isinstance(value, np.ndarray):
                expanded_info[key] = np.expand_dims(value, axis=1)
            else:
                expanded_info[key] = value

        with torch.inference_mode():
            costs = self.model.get_cost(expanded_info, action_plan)

        predicted_emb = expanded_info.get("predicted_emb")
        if predicted_emb is None:
            raise RuntimeError(
                "model.get_cost did not expose predicted_emb; cannot save planning latents"
            )
        context_len = int(expanded_info["pixels"].shape[2])
        state_plan_latents = predicted_emb[:, 0, context_len:].detach().cpu()
        return {
            "action_plan": action_plan[:, 0].detach().cpu(),
            "cost": costs[:, 0].detach().cpu(),
            "state_plan_latents": state_plan_latents,
        }

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)

        if len(self._action_buffer) == 0:
            outputs = _solve_policy_actions(self, info_dict)

            actions = outputs["actions"]
            plan_analysis = self._analyze_action_plan(info_dict, actions)
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            plan = plan.reshape(
                self.env.num_envs, self.flatten_receding_horizon, -1
            )
            self._action_buffer.extend(plan.transpose(0, 1))

            if self._task_artifacts is not None:
                for env_idx in range(self.env.num_envs):
                    self._task_artifacts[env_idx].append(
                        {
                            "replan_index": len(self._task_artifacts[env_idx]),
                            "env_step": int(self._env_steps[env_idx]),
                            "cost": plan_analysis["cost"][env_idx].clone(),
                            "action_plan": plan_analysis["action_plan"][
                                env_idx
                            ].clone(),
                            "executed_action_plan": plan_analysis["action_plan"][
                                env_idx, :keep_horizon
                            ].clone(),
                            "warm_start_action_plan": plan_analysis["action_plan"][
                                env_idx, keep_horizon:
                            ].clone(),
                            "state_plan_latents": plan_analysis[
                                "state_plan_latents"
                            ][env_idx].clone(),
                        }
                    )

        action = _format_policy_action(self, self._action_buffer.popleft())
        if self._env_steps is not None:
            self._env_steps += 1

        return action

    def export_task_artifacts(self, task_dirs) -> None:
        if self._task_artifacts is None:
            return
        for task_dir, replans in zip(task_dirs, self._task_artifacts, strict=True):
            task_dir.mkdir(parents=True, exist_ok=True)
            state_latent_payload = {
                "policy_type": self.type,
                "action_block": int(self.cfg.action_block),
                "receding_horizon": int(self.cfg.receding_horizon),
                "plan_horizon": int(self.cfg.horizon),
                "replans": [
                    {
                        "replan_index": record["replan_index"],
                        "env_step": record["env_step"],
                        "cost": record["cost"],
                        "state_plan_latents": record["state_plan_latents"],
                    }
                    for record in replans
                ],
            }
            torch.save(state_latent_payload, task_dir / "planning_latents.pt")
            torch.save(
                {
                    "policy_type": self.type,
                    "action_block": int(self.cfg.action_block),
                    "receding_horizon": int(self.cfg.receding_horizon),
                    "plan_horizon": int(self.cfg.horizon),
                    "replans": replans,
                },
                task_dir / "planning_artifacts.pt",
            )


class EvalWorldModelPolicy(swm.policy.WorldModelPolicy):
    """World-model policy with solver configuration fixes for Gym vector envs."""

    def set_env(self, env):
        _set_eval_policy_env(self, env)

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)

        if len(self._action_buffer) == 0:
            outputs = _solve_policy_actions(self, info_dict)

            actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            plan = plan.reshape(
                self.env.num_envs, self.flatten_receding_horizon, -1
            )
            self._action_buffer.extend(plan.transpose(0, 1))

        return _format_policy_action(self, self._action_buffer.popleft())


def _set_eval_policy_env(policy, env) -> None:
    """Configure solvers with the action-space shape they expect.

    The upstream CEM solver expects a vectorized Box space whose leading
    dimension is `num_envs`; the PGD solver expects the single-env Discrete
    space. Gymnasium's vector wrapper exposes discrete vector actions as
    MultiDiscrete, so we reconfigure discrete solvers with `single_action_space`
    after the base policy setup.
    """

    policy.env = env
    n_envs = getattr(env, "num_envs", 1)
    single_action_space = getattr(env, "single_action_space", None)
    solver_action_space = (
        single_action_space
        if isinstance(single_action_space, gym.spaces.Discrete)
        else env.action_space
    )
    policy.solver.configure(
        action_space=solver_action_space,
        n_envs=n_envs,
        config=policy.cfg,
    )
    policy._action_buffer = deque(maxlen=policy.flatten_receding_horizon)


def _uses_discrete_single_action_space(env) -> bool:
    single_action_space = getattr(env, "single_action_space", None)
    return isinstance(single_action_space, gym.spaces.Discrete)


def _solve_policy_actions(policy, info_dict: dict) -> dict:
    kwargs = {"init_action": policy._next_init}
    if _uses_discrete_single_action_space(policy.env):
        kwargs["from_scalar"] = True
    return policy.solver(info_dict, **kwargs)


def _format_policy_action(policy, action: torch.Tensor) -> np.ndarray:
    """Convert a buffered planner action into the shape expected by the env."""

    if _uses_discrete_single_action_space(policy.env):
        single_action_space = policy.env.single_action_space
        action_arr = action.reshape(policy.env.num_envs, -1).detach().cpu().numpy()
        action_ids = np.rint(action_arr[:, 0]).astype(np.int64)
        return np.clip(action_ids, 0, single_action_space.n - 1)

    action_arr = (
        action.reshape(*policy.env.action_space.shape).detach().cpu().numpy()
    )
    if "action" in policy.process:
        action_arr = policy.process["action"].inverse_transform(action_arr)
    return action_arr


def create_eval_artifact_dir(cfg: DictConfig, results_path: Path) -> Path:
    if cfg.output.get("planning_artifacts_dir") is not None:
        root = Path(str(cfg.output.planning_artifacts_dir)).expanduser().resolve()
    else:
        root = results_path / "planning_artifacts"
    policy_name = str(cfg.policy).replace("/", "__")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = root / f"{timestamp}_{policy_name}_seed{cfg.seed}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def build_task_dirs(
    artifact_dir: Path, episodes: np.ndarray, start_steps: np.ndarray
) -> list[Path]:
    task_dirs = []
    for idx, (episode, start_step) in enumerate(zip(episodes, start_steps)):
        task_dir = artifact_dir / f"task_{idx:03d}_ep{int(episode)}_start{int(start_step)}"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_dirs.append(task_dir)
    return task_dirs


def save_task_metadata(
    *,
    task_dirs: list[Path],
    episodes: np.ndarray,
    start_steps: np.ndarray,
    cfg: DictConfig,
    metrics: dict,
) -> None:
    episode_successes = metrics.get("episode_successes")
    for idx, task_dir in enumerate(task_dirs):
        success = None
        if episode_successes is not None:
            success = bool(episode_successes[idx])
        payload = {
            "task_index": idx,
            "episode_idx": int(episodes[idx]),
            "start_step": int(start_steps[idx]),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "eval_budget": int(cfg.eval.eval_budget),
            "success": success,
        }
        (task_dir / "task_info.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def relocate_rollout_videos(video_dir: Path, task_dirs: list[Path]) -> None:
    for idx, task_dir in enumerate(task_dirs):
        src = video_dir / f"rollout_{idx}.mp4"
        if src.exists():
            shutil.move(str(src), str(task_dir / "rollout.mp4"))


def save_policy_artifacts(policy, task_dirs: list[Path]) -> None:
    if hasattr(policy, "export_task_artifacts"):
        policy.export_task_artifacts(task_dirs)


def instrument_world_step(world, total_steps: int, label: str, *, min_interval_s: float = 1.0):
    """Wrap ``world.step`` so each env step prints a progress bar with ETA.

    The slow planning loop lives inside ``world.evaluate_from_dataset`` (one
    ``self.step()`` per env step, ``eval_budget`` total). Shadowing the bound
    method on the instance lets us report progress without editing the
    ``stable_worldmodel`` package.
    """
    original_step = world.step
    printer = ProgressPrinter(total_steps, label=label, min_interval_s=min_interval_s)

    def step_with_progress(*args, **kwargs):
        result = original_step(*args, **kwargs)
        printer.update(1)
        return result

    world.step = step_with_progress
    return printer


def create_world(cfg: DictConfig, num_envs: int):
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = int(num_envs)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    height = int(world_cfg.pop("height", cfg.eval.img_size))
    width = int(world_cfg.pop("width", cfg.eval.img_size))
    return swm.World(**world_cfg, image_shape=(height, width))


def build_policy(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None,
    process: dict,
    transform: dict,
):
    if str(cfg.policy) == "random":
        return swm.policy.RandomPolicy()

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy_cls = (
        RecordingWorldModelPolicy
        if cfg.output.get("save_planning_artifacts", False)
        else EvalWorldModelPolicy
    )
    kwargs = {"model": model} if policy_cls is RecordingWorldModelPolicy else {}
    return policy_cls(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
        **kwargs,
    )


def aggregate_eval_metrics(batch_metrics: list[dict]) -> dict:
    if not batch_metrics:
        return {}

    metrics = {}
    successes = [
        item.get("episode_successes")
        for item in batch_metrics
        if item.get("episode_successes") is not None
    ]
    if successes:
        episode_successes = np.concatenate(
            [np.asarray(success, dtype=bool) for success in successes]
        )
        metrics["success_rate"] = float(episode_successes.mean() * 100.0)
        metrics["episode_successes"] = episode_successes
    elif all("success_rate" in item for item in batch_metrics):
        metrics["success_rate"] = float(
            np.mean([float(item["success_rate"]) for item in batch_metrics])
        )

    seeds = [
        item.get("seeds")
        for item in batch_metrics
        if item.get("seeds") is not None
    ]
    metrics["seeds"] = None if not seeds else seeds
    return metrics


def _make_official_vector_env(env_name: str, num_envs: int):
    # Import registers the OGBench gymnasium env ids.
    import ogbench.manipspace  # noqa: F401

    return gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_name) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


def _make_ogb_vector_env(dataset_name: str, num_envs: int, env_kwargs: dict):
    import ogbench

    def make_env():
        return ogbench.make_env_and_datasets(
            dataset_name,
            env_only=True,
            **env_kwargs,
        )

    return gym.vector.SyncVectorEnv(
        [make_env for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


def _as_rgb_pixels(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim < 4:
        raise ValueError(f"Expected batched image observations, got {arr.shape}")
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[-1] == 6:
        return arr[..., :3]
    raise ValueError(f"Expected RGB or Powderworld RGB+overlay observations, got {arr.shape}")


def _action_model_dim(process: dict, envs) -> int:
    action_processor = process.get("action")
    if action_processor is not None and hasattr(action_processor, "mean_"):
        return int(np.asarray(action_processor.mean_).reshape(-1).shape[0])

    single_action_space = getattr(envs, "single_action_space", envs.action_space)
    if isinstance(single_action_space, gym.spaces.Discrete):
        return int(single_action_space.n)
    return int(np.prod(single_action_space.shape))


def _action_to_model_vector(actions, action_dim: int) -> np.ndarray:
    arr = np.asarray(actions)
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 0 else arr.reshape(1, 1)
    if flat.shape[1] == 1 and np.issubdtype(flat.dtype, np.integer) and action_dim > 1:
        indices = flat[:, 0].astype(np.int64)
        out = np.zeros((indices.shape[0], action_dim), dtype=np.float32)
        valid = (indices >= 0) & (indices < action_dim)
        out[np.arange(indices.shape[0])[valid], indices[valid]] = 1.0
        return out
    return flat.astype(np.float32, copy=False)


def _policy_step_info(obs: np.ndarray, goal: np.ndarray) -> dict:
    batch_size = int(obs.shape[0])
    return {
        "pixels": obs[:, None, ...],
        "goal": goal[:, None, ...],
        # JEPA.get_cost normalizes goal_* keys then drops goal["action"] before
        # encoding the goal. Dataset-goal eval naturally provides goal_action;
        # official OGBench reset goals are image-only, so provide a harmless
        # placeholder to preserve that cost-model contract.
        "goal_action": np.zeros((batch_size, 1, 5), dtype=np.float32),
    }


def _ogb_policy_step_info(
    obs: np.ndarray,
    goal: np.ndarray,
    last_action: np.ndarray,
    synthetic_pixels: str | None = None,
) -> dict:
    if synthetic_pixels == "antsoccer_topdown":
        obs_pixels = _antsoccer_topdown_pixels_from_qpos(np.asarray(obs)[:, :22])
        goal_pixels = _antsoccer_topdown_pixels_from_qpos(np.asarray(goal)[:, :22])
    elif synthetic_pixels:
        raise ValueError(f"Unknown eval.synthetic_pixels={synthetic_pixels}")
    else:
        obs_pixels = _as_rgb_pixels(obs)
        goal_pixels = _as_rgb_pixels(goal)
    return {
        "pixels": obs_pixels[:, None, ...],
        "goal": goal_pixels[:, None, ...],
        "action": last_action[:, None, :],
    }


def _current_ogb_ob(single_env) -> np.ndarray:
    env = single_env.unwrapped
    if hasattr(env, "get_ob"):
        return np.asarray(env.get_ob())
    if hasattr(env, "compute_observation"):
        return np.asarray(env.compute_observation())
    if hasattr(env, "_get_ob"):
        return np.asarray(env._get_ob())
    return np.asarray(env.render())


def _powderworld_world_from_rgb(single_env, rgb: np.ndarray):
    env = single_env.unwrapped
    rgb = np.asarray(rgb)[..., :3].astype(np.float32)
    colors = np.rint(env.pwr.elem_vecs_array * 255.0).astype(np.float32)
    flat = rgb.reshape(-1, 3)
    dists = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
    elem_ids = dists.argmin(axis=1).reshape(rgb.shape[:2]).astype(np.uint8)
    return env.pw.np_to_pw(elem_ids[None, ...]).copy()


def _set_powderworld_from_pixels(single_env, pixels: np.ndarray, goal_pixels: np.ndarray) -> None:
    env = single_env.unwrapped
    env._world = _powderworld_world_from_rgb(single_env, pixels)
    env.cur_goal_world = _powderworld_world_from_rgb(single_env, goal_pixels)[0, 0].copy()
    if getattr(env, "cur_task_info", None) is None:
        env.cur_task_info = {"tol": 64}
    env._action_step = 0
    env._action_elem_id = None
    env._action_x = None
    env._action_y = None


def _ogb_goal_xy(dataset_name: str, row: dict[str, np.ndarray]) -> np.ndarray | None:
    if "qpos" not in row:
        return None
    qpos = np.asarray(row["qpos"])
    name = dataset_name.lower()
    if "antsoccer" in name:
        return qpos[-7:-5].copy()
    if "antmaze" in name:
        return qpos[:2].copy()
    return None


def _set_ogb_env_from_rows(
    single_env,
    *,
    dataset_name: str,
    start_row: dict[str, np.ndarray],
    goal_row: dict[str, np.ndarray],
) -> np.ndarray:
    env = single_env.unwrapped
    name = dataset_name.lower()

    if "powderworld" in name:
        _set_powderworld_from_pixels(single_env, start_row["pixels"], goal_row["pixels"])
        return _current_ogb_ob(single_env)

    if "qpos" in start_row and "qvel" in start_row:
        qpos = np.asarray(start_row["qpos"])
        qvel = np.asarray(start_row["qvel"])
        if "button_states" in start_row:
            env.set_state(qpos, qvel, np.asarray(start_row["button_states"]))
        else:
            env.set_state(qpos, qvel)

    if "puzzle" in name and "button_states" in goal_row:
        env._target_button_states = np.asarray(goal_row["button_states"]).copy()
        env._target_task = "all"

    goal_xy = _ogb_goal_xy(dataset_name, goal_row)
    if goal_xy is not None and hasattr(env, "set_goal"):
        env.set_goal(goal_xy=goal_xy)

    return _current_ogb_ob(single_env)


def _make_row_dict(row_batch: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[idx] for key, value in row_batch.items()}


def _ogb_trajectory_success(
    *,
    dataset_name: str,
    obs: np.ndarray,
    infos: dict,
    goal_rows: dict[str, np.ndarray],
    cfg: DictConfig,
) -> np.ndarray:
    name = dataset_name.lower()
    batch_size = int(np.asarray(obs).shape[0])
    success = np.asarray(infos.get("success", np.zeros(batch_size)), dtype=bool)

    if "puzzle" in name and "button_states" in infos and "button_states" in goal_rows:
        current = np.asarray(infos["button_states"])
        target = np.asarray(goal_rows["button_states"])
        success |= np.all(current == target, axis=1)
        return success

    if ("antmaze" in name or "antsoccer" in name) and "qpos" in infos and "qpos" in goal_rows:
        qpos = np.asarray(infos["qpos"])
        goal_qpos = np.asarray(goal_rows["qpos"])
        current_xy = qpos[:, -7:-5] if "antsoccer" in name else qpos[:, :2]
        goal_xy = goal_qpos[:, -7:-5] if "antsoccer" in name else goal_qpos[:, :2]
        tolerance = float(cfg.eval.get("success_distance_tolerance", 0.5))
        success |= np.linalg.norm(current_xy - goal_xy, axis=1) <= tolerance
        return success

    if "pixels" in goal_rows:
        current_pixels = _as_rgb_pixels(np.asarray(obs))
        goal_pixels = _as_rgb_pixels(np.asarray(goal_rows["pixels"]))
        diff = np.abs(current_pixels.astype(np.int16) - goal_pixels.astype(np.int16))
        per_pixel_close = (diff <= int(cfg.eval.get("pixel_success_channel_tolerance", 8))).all(axis=-1)
        mismatch = (~per_pixel_close).sum(axis=tuple(range(1, per_pixel_close.ndim)))
        threshold = int(cfg.eval.get("pixel_success_mismatch_tolerance", 64))
        success |= mismatch <= threshold

    return success


def evaluate_official_scene(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None,
    process: dict,
    transform: dict,
) -> dict:
    """Evaluate on the official OGBench Visual Scene fixed-goal tasks."""

    task_ids = [int(task_id) for task_id in cfg.eval.task_ids]
    episodes_per_task = int(cfg.eval.num_eval)
    episode_start = int(cfg.eval.get("episode_start", 0))
    episode_end = episode_start + episodes_per_task
    env_batch_size = int(cfg.eval.env_batch_size or 1)
    env_batch_size = max(1, min(env_batch_size, episodes_per_task))
    max_episode_steps = int(cfg.eval.max_episode_steps)
    progress_enabled = bool(cfg.eval.get("progress_bar", True))
    env_template = str(cfg.eval.env_template)

    all_successes = []
    all_steps = []
    per_task = {}
    total_budget = len(task_ids) * episodes_per_task * max_episode_steps
    progress = (
        ProgressPrinter(total_budget, label="official-scene", min_interval_s=1.0)
        if progress_enabled
        else None
    )

    try:
        for task_id in task_ids:
            env_name = env_template.format(task_id=task_id)
            task_successes = []
            task_steps = []
            print(
                f"Official Scene task {task_id}: env={env_name}, "
                f"episodes={episode_start}-{episode_end - 1}, "
                f"max_steps={max_episode_steps}",
                flush=True,
            )

            for batch_start in range(episode_start, episode_end, env_batch_size):
                batch_end = min(batch_start + env_batch_size, episode_end)
                batch_size = batch_end - batch_start
                envs = _make_official_vector_env(env_name, batch_size)
                policy = build_policy(
                    cfg,
                    model=model,
                    process=process,
                    transform=transform,
                )
                policy.set_env(envs)

                seeds = [
                    int(cfg.seed) + task_id * 100000 + ep_idx
                    for ep_idx in range(batch_start, batch_end)
                ]
                obs, infos = envs.reset(seed=seeds)
                if "goal" not in infos:
                    envs.close()
                    raise RuntimeError(
                        f"{env_name} reset did not provide info['goal']"
                    )
                goal = np.asarray(infos["goal"])
                active = np.ones(batch_size, dtype=bool)
                successes = np.zeros(batch_size, dtype=bool)
                steps_taken = np.full(batch_size, max_episode_steps, dtype=np.int32)

                for step_idx in range(max_episode_steps):
                    if not active.any():
                        break

                    if str(cfg.policy) == "random":
                        actions = envs.action_space.sample()
                    else:
                        info_for_policy = _policy_step_info(np.asarray(obs), goal)
                        actions = policy.get_action(info_for_policy)

                    actions = np.asarray(actions, dtype=np.float32)
                    actions[~active] = 0.0
                    obs, rewards, terminated, truncated, infos = envs.step(actions)

                    terminated = np.asarray(terminated, dtype=bool)
                    truncated = np.asarray(truncated, dtype=bool)
                    success_info = np.asarray(
                        infos.get("success", terminated), dtype=bool
                    )
                    newly_done = active & (terminated | truncated)
                    newly_success = active & (terminated | success_info)
                    successes |= newly_success
                    steps_taken[newly_done] = step_idx + 1
                    active &= ~newly_done

                    if progress is not None:
                        progress.update(batch_size)

                envs.close()
                task_successes.extend(successes.tolist())
                task_steps.extend(steps_taken.tolist())

            task_success_arr = np.asarray(task_successes, dtype=bool)
            task_steps_arr = np.asarray(task_steps, dtype=np.int32)
            per_task[f"task{task_id}"] = {
                "success_rate": float(task_success_arr.mean() * 100.0),
                "episode_successes": task_success_arr,
                "episode_steps": task_steps_arr,
                "mean_steps": float(task_steps_arr.mean()),
            }
            all_successes.append(task_success_arr)
            all_steps.append(task_steps_arr)
            print(
                f"Official Scene task {task_id} success_rate="
                f"{per_task[f'task{task_id}']['success_rate']:.1f}% "
                f"mean_steps={per_task[f'task{task_id}']['mean_steps']:.1f}",
                flush=True,
            )
    finally:
        if progress is not None:
            progress.close()

    episode_successes = np.concatenate(all_successes) if all_successes else np.array([])
    episode_steps = np.concatenate(all_steps) if all_steps else np.array([])
    return {
        "protocol": "official_scene",
        "task_ids": task_ids,
        "episode_start": episode_start,
        "episode_end": episode_end,
        "success_rate": float(episode_successes.mean() * 100.0)
        if episode_successes.size
        else 0.0,
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
        "mean_steps": float(episode_steps.mean()) if episode_steps.size else 0.0,
        "per_task": per_task,
    }


def evaluate_ogb_online(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None,
    process: dict,
    transform: dict,
) -> dict:
    """Evaluate an offline-trained model on OGBench reset goals."""

    dataset_name = str(cfg.eval.ogb_dataset_name)
    episodes = int(cfg.eval.num_eval)
    env_batch_size = int(cfg.eval.env_batch_size or episodes)
    env_batch_size = max(1, min(env_batch_size, episodes))
    env_kwargs = OmegaConf.to_container(cfg.eval.get("env_kwargs", {}), resolve=True)
    env_kwargs = env_kwargs or {}
    synthetic_pixels = cfg.eval.get("synthetic_pixels", None)
    progress_enabled = bool(cfg.eval.get("progress_bar", True))

    all_successes = []
    all_steps = []
    all_seeds = []

    max_steps = cfg.eval.get("max_episode_steps")
    total_budget = episodes * int(max_steps or 1)
    progress = None
    if progress_enabled and max_steps is not None:
        progress = ProgressPrinter(
            total_budget,
            label="ogb-online",
            min_interval_s=float(cfg.eval.get("progress_refresh_s", 1.0)),
        )

    try:
        for batch_start in range(0, episodes, env_batch_size):
            batch_end = min(batch_start + env_batch_size, episodes)
            batch_size = batch_end - batch_start
            envs = _make_ogb_vector_env(dataset_name, batch_size, env_kwargs)
            episode_max_steps = int(max_steps or envs.envs[0].spec.max_episode_steps)
            if progress_enabled and progress is None:
                progress = ProgressPrinter(
                    episodes * episode_max_steps,
                    label="ogb-online",
                    min_interval_s=float(cfg.eval.get("progress_refresh_s", 1.0)),
                )

            policy = build_policy(
                cfg,
                model=model,
                process=process,
                transform=transform,
            )
            policy.set_env(envs)

            seeds = [int(cfg.seed) + ep_idx for ep_idx in range(batch_start, batch_end)]
            obs, infos = envs.reset(seed=seeds)
            if "goal" not in infos:
                envs.close()
                raise RuntimeError(f"{dataset_name} reset did not provide info['goal']")
            goal = np.asarray(infos["goal"])
            action_dim = _action_model_dim(process, envs)
            last_action = np.zeros((batch_size, action_dim), dtype=np.float32)
            active = np.ones(batch_size, dtype=bool)
            successes = np.zeros(batch_size, dtype=bool)
            steps_taken = np.full(batch_size, episode_max_steps, dtype=np.int32)

            for step_idx in range(episode_max_steps):
                if not active.any():
                    break

                if str(cfg.policy) == "random":
                    actions = envs.action_space.sample()
                else:
                    info_for_policy = _ogb_policy_step_info(
                        obs,
                        goal,
                        last_action,
                        synthetic_pixels=synthetic_pixels,
                    )
                    actions = policy.get_action(info_for_policy)

                actions_for_step = np.asarray(actions)
                if actions_for_step.ndim == 0:
                    actions_for_step = actions_for_step.reshape(1)
                if actions_for_step.shape[0] != batch_size:
                    actions_for_step = actions_for_step.reshape(batch_size, -1)
                if np.issubdtype(actions_for_step.dtype, np.floating):
                    actions_for_step = actions_for_step.astype(np.float32, copy=False)
                    if hasattr(envs.single_action_space, "low"):
                        low = np.asarray(envs.single_action_space.low)
                        high = np.asarray(envs.single_action_space.high)
                        actions_for_step = np.clip(actions_for_step, low, high)
                actions_for_step = actions_for_step.copy()
                if actions_for_step.ndim > 1:
                    actions_for_step[~active] = 0
                else:
                    actions_for_step[~active] = 0

                obs, rewards, terminated, truncated, infos = envs.step(actions_for_step)
                last_action = _action_to_model_vector(actions_for_step, action_dim)

                terminated = np.asarray(terminated, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool)
                success_info = np.asarray(infos.get("success", terminated), dtype=bool)
                newly_done = active & (terminated | truncated)
                newly_success = active & (terminated | success_info)
                successes |= newly_success
                steps_taken[newly_done] = step_idx + 1
                active &= ~newly_done

                if progress is not None:
                    progress.update(batch_size)

            envs.close()
            all_successes.append(successes)
            all_steps.append(steps_taken)
            all_seeds.extend(seeds)
            print(
                f"OGB online {dataset_name} batch {batch_start}-{batch_end - 1} "
                f"success_rate={successes.mean() * 100.0:.1f}% "
                f"mean_steps={steps_taken.mean():.1f}",
                flush=True,
            )
    finally:
        if progress is not None:
            progress.close()

    episode_successes = np.concatenate(all_successes) if all_successes else np.array([])
    episode_steps = np.concatenate(all_steps) if all_steps else np.array([])
    return {
        "protocol": "ogb_online",
        "ogb_dataset_name": dataset_name,
        "success_rate": float(episode_successes.mean() * 100.0)
        if episode_successes.size
        else 0.0,
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
        "mean_steps": float(episode_steps.mean()) if episode_steps.size else 0.0,
        "seeds": np.asarray(all_seeds, dtype=np.int64),
    }


def evaluate_ogb_trajectory(
    cfg: DictConfig,
    *,
    model: torch.nn.Module | None,
    process: dict,
    transform: dict,
    dataset,
) -> dict:
    """Evaluate on future goals sampled from the converted OGB HDF5 dataset."""

    dataset_name = str(cfg.eval.ogb_dataset_name)
    episodes = int(cfg.eval.num_eval)
    env_batch_size = int(cfg.eval.env_batch_size or episodes)
    env_batch_size = max(1, min(env_batch_size, episodes))
    env_kwargs = OmegaConf.to_container(cfg.eval.get("env_kwargs", {}), resolve=True)
    env_kwargs = env_kwargs or {}
    synthetic_pixels = cfg.eval.get("synthetic_pixels", None)
    progress_enabled = bool(cfg.eval.get("progress_bar", True))
    goal_offset_steps = int(cfg.eval.goal_offset_steps)
    eval_budget = int(cfg.eval.eval_budget)

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        int(ep_id): int(max_start_idx[idx]) for idx, ep_id in enumerate(ep_indices)
    }
    episode_ids_per_row = dataset.get_col_data(col_name)
    max_start_per_row = np.array(
        [max_start_idx_dict[int(ep_id)] for ep_id in episode_ids_per_row]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < episodes:
        raise ValueError(
            f"Only {len(valid_indices)} valid OGB trajectory starts for {episodes} evals"
        )

    rng = np.random.default_rng(int(cfg.seed))
    selected = np.sort(rng.choice(valid_indices, size=episodes, replace=False))
    goal_indices = selected + goal_offset_steps
    start_episode_ids = dataset.get_row_data(selected)[col_name]
    start_steps = dataset.get_row_data(selected)["step_idx"]

    all_successes = []
    all_steps = []
    progress = (
        ProgressPrinter(
            episodes * eval_budget,
            label="ogb-trajectory",
            min_interval_s=float(cfg.eval.get("progress_refresh_s", 1.0)),
        )
        if progress_enabled
        else None
    )

    try:
        for batch_start in range(0, episodes, env_batch_size):
            batch_end = min(batch_start + env_batch_size, episodes)
            batch_size = batch_end - batch_start
            batch_indices = selected[batch_start:batch_end]
            batch_goal_indices = goal_indices[batch_start:batch_end]

            start_rows = dataset.get_row_data(batch_indices)
            goal_rows = dataset.get_row_data(batch_goal_indices)
            envs = _make_ogb_vector_env(dataset_name, batch_size, env_kwargs)
            policy = build_policy(
                cfg,
                model=model,
                process=process,
                transform=transform,
            )
            policy.set_env(envs)

            seeds = [int(cfg.seed) + ep_idx for ep_idx in range(batch_start, batch_end)]
            envs.reset(seed=seeds)
            obs = []
            for env_idx, single_env in enumerate(envs.envs):
                obs.append(
                    _set_ogb_env_from_rows(
                        single_env,
                        dataset_name=dataset_name,
                        start_row=_make_row_dict(start_rows, env_idx),
                        goal_row=_make_row_dict(goal_rows, env_idx),
                    )
                )
            obs = np.stack(obs)

            if synthetic_pixels == "antsoccer_topdown" and "qpos" in goal_rows:
                policy_goal = np.asarray(goal_rows["qpos"])
            else:
                policy_goal = np.asarray(goal_rows["pixels"])

            action_dim = _action_model_dim(process, envs)
            last_action = np.zeros((batch_size, action_dim), dtype=np.float32)
            active = np.ones(batch_size, dtype=bool)
            successes = np.zeros(batch_size, dtype=bool)
            steps_taken = np.full(batch_size, eval_budget, dtype=np.int32)

            for step_idx in range(eval_budget):
                if not active.any():
                    break

                if str(cfg.policy) == "random":
                    actions = envs.action_space.sample()
                else:
                    info_for_policy = _ogb_policy_step_info(
                        obs,
                        policy_goal,
                        last_action,
                        synthetic_pixels=synthetic_pixels,
                    )
                    actions = policy.get_action(info_for_policy)

                actions_for_step = np.asarray(actions)
                if actions_for_step.ndim == 0:
                    actions_for_step = actions_for_step.reshape(1)
                if actions_for_step.shape[0] != batch_size:
                    actions_for_step = actions_for_step.reshape(batch_size, -1)
                if np.issubdtype(actions_for_step.dtype, np.floating):
                    actions_for_step = actions_for_step.astype(np.float32, copy=False)
                    if hasattr(envs.single_action_space, "low"):
                        low = np.asarray(envs.single_action_space.low)
                        high = np.asarray(envs.single_action_space.high)
                        actions_for_step = np.clip(actions_for_step, low, high)
                actions_for_step = actions_for_step.copy()
                if actions_for_step.ndim > 1:
                    actions_for_step[~active] = 0
                else:
                    actions_for_step[~active] = 0

                obs, rewards, terminated, truncated, infos = envs.step(actions_for_step)
                last_action = _action_to_model_vector(actions_for_step, action_dim)

                terminated = np.asarray(terminated, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool)
                custom_success = _ogb_trajectory_success(
                    dataset_name=dataset_name,
                    obs=np.asarray(obs),
                    infos=infos,
                    goal_rows=goal_rows,
                    cfg=cfg,
                )
                newly_success = active & custom_success
                newly_done = active & (terminated | truncated | newly_success)
                successes |= newly_success
                steps_taken[newly_done] = step_idx + 1
                active &= ~newly_done

                if progress is not None:
                    progress.update(batch_size)

            envs.close()
            all_successes.append(successes)
            all_steps.append(steps_taken)
            print(
                f"OGB trajectory {dataset_name} batch {batch_start}-{batch_end - 1} "
                f"success_rate={successes.mean() * 100.0:.1f}% "
                f"mean_steps={steps_taken.mean():.1f}",
                flush=True,
            )
    finally:
        if progress is not None:
            progress.close()

    episode_successes = np.concatenate(all_successes) if all_successes else np.array([])
    episode_steps = np.concatenate(all_steps) if all_steps else np.array([])
    return {
        "protocol": "ogb_trajectory",
        "ogb_dataset_name": dataset_name,
        "dataset_name": str(cfg.eval.dataset_name),
        "goal_offset_steps": goal_offset_steps,
        "eval_budget": eval_budget,
        "success_rate": float(episode_successes.mean() * 100.0)
        if episode_successes.size
        else 0.0,
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
        "mean_steps": float(episode_steps.mean()) if episode_steps.size else 0.0,
        "start_episode_ids": np.asarray(start_episode_ids, dtype=np.int64),
        "start_steps": np.asarray(start_steps, dtype=np.int64),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    protocol = str(cfg.eval.get("protocol", "dataset"))
    policy = cfg.get("policy", "random")
    process = {}
    dataset = None
    ep_indices = None
    random_without_stats = protocol in {"official_scene", "ogb_online"} and str(policy) == "random"
    if not random_without_stats:
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

        for col in cfg.dataset.keys_to_cache:
            if col in ["pixels"]:
                continue
            processor = preprocessing.StandardScaler()
            col_data = stats_dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
            process[col] = processor

            if col != "action":
                process[f"goal_{col}"] = process[col]

    # -- run evaluation
    device = resolve_device(str(cfg.solver.get("device", "auto")))
    cfg.solver.device = device

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
    else:
        model = None

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    if protocol == "official_scene":
        start_time = time.time()
        metrics = evaluate_official_scene(
            cfg,
            model=model,
            process=process,
            transform=transform,
        )
        end_time = time.time()
        print(metrics)

        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write("\n")
            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")
            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {_to_jsonable(metrics)}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        return

    if protocol == "ogb_trajectory":
        start_time = time.time()
        metrics = evaluate_ogb_trajectory(
            cfg,
            model=model,
            process=process,
            transform=transform,
            dataset=dataset,
        )
        end_time = time.time()
        print(metrics)

        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write("\n")
            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")
            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {_to_jsonable(metrics)}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        return

    if protocol == "ogb_online":
        start_time = time.time()
        metrics = evaluate_ogb_online(
            cfg,
            model=model,
            process=process,
            transform=transform,
        )
        end_time = time.time()
        print(metrics)

        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write("\n")
            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")
            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {_to_jsonable(metrics)}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        return

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    artifact_dir = None
    task_dirs = []
    eval_video_path = results_path
    if cfg.output.get("save_planning_artifacts", False):
        artifact_dir = create_eval_artifact_dir(cfg, results_path)
        task_dirs = build_task_dirs(artifact_dir, eval_episodes, eval_start_idx)
        eval_video_path = artifact_dir / "_videos"

    env_batch_size = int(cfg.eval.env_batch_size or cfg.eval.num_eval)
    env_batch_size = max(1, min(env_batch_size, int(cfg.eval.num_eval)))
    n_batches = (int(cfg.eval.num_eval) + env_batch_size - 1) // env_batch_size
    progress_enabled = bool(cfg.eval.get("progress_bar", True))
    print(
        f"Evaluating policy={cfg.policy} | episodes={cfg.eval.num_eval} "
        f"| batches={n_batches} (size {env_batch_size}) "
        f"| eval_budget={cfg.eval.eval_budget} steps/episode",
        flush=True,
    )
    batch_metrics = []
    start_time = time.time()
    for batch_idx, batch_start in enumerate(
        range(0, int(cfg.eval.num_eval), env_batch_size)
    ):
        batch_end = min(batch_start + env_batch_size, int(cfg.eval.num_eval))
        batch_episodes = eval_episodes[batch_start:batch_end]
        batch_start_idx = eval_start_idx[batch_start:batch_end]

        if env_batch_size < int(cfg.eval.num_eval):
            print(
                f"Evaluating batch {batch_idx + 1}: "
                f"items {batch_start}-{batch_end - 1}",
                flush=True,
            )

        world = create_world(cfg, num_envs=len(batch_episodes))
        policy = build_policy(
            cfg,
            model=model,
            process=process,
            transform=transform,
        )
        world.set_policy(policy)

        progress = None
        if progress_enabled:
            label = f"eval b{batch_idx + 1}/{n_batches}" if n_batches > 1 else "eval"
            progress = instrument_world_step(
                world,
                int(cfg.eval.eval_budget),
                label,
                min_interval_s=float(cfg.eval.get("progress_refresh_s", 1.0)),
            )

        batch_video_path = eval_video_path
        if (
            cfg.output.get("save_video", True)
            and env_batch_size < int(cfg.eval.num_eval)
        ):
            batch_video_path = eval_video_path / f"batch_{batch_idx:03d}"

        metrics_i = world.evaluate_from_dataset(
            dataset,
            start_steps=batch_start_idx.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=batch_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            save_video=cfg.output.get("save_video", True),
            video_path=batch_video_path,
        )
        if progress is not None:
            progress.close(
                suffix=f"success_rate={metrics_i.get('success_rate', 0.0):.1f}%"
            )
        batch_metrics.append(metrics_i)

        if artifact_dir is not None:
            batch_task_dirs = task_dirs[batch_start:batch_end]
            if cfg.output.get("save_video", True):
                relocate_rollout_videos(batch_video_path, batch_task_dirs)
                if batch_video_path.exists() and not any(batch_video_path.iterdir()):
                    batch_video_path.rmdir()
            save_task_metadata(
                task_dirs=batch_task_dirs,
                episodes=batch_episodes,
                start_steps=batch_start_idx,
                cfg=cfg,
                metrics=metrics_i,
            )
            save_policy_artifacts(policy, batch_task_dirs)

    metrics = aggregate_eval_metrics(batch_metrics)
    end_time = time.time()

    if artifact_dir is not None:
        if cfg.output.get("save_video", True):
            if eval_video_path.exists() and not any(eval_video_path.iterdir()):
                eval_video_path.rmdir()
        (artifact_dir / "eval_metadata.json").write_text(
            json.dumps(
                {
                    "policy": str(cfg.policy),
                    "dataset_name": str(cfg.eval.dataset_name),
                    "metrics": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in metrics.items()
                    },
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        if artifact_dir is not None:
            f.write(f"artifacts_dir: {artifact_dir}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
