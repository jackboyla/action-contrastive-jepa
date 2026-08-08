#!/usr/bin/env python3
"""Human-play demo — drive the world-model tasks yourself.

This opens an interactive window that shows you *exactly what the model sees*
during evaluation: the current 224x224 RGB observation on the left and the
224x224 goal image on the right. You issue actions from the keyboard in the
same continuous action space the planner uses, the environment steps, and you
watch the transition. Reach the goal within the step budget and the env's own
success criterion fires — the same one used in eval.

It mirrors ``eval.py`` / ``World.evaluate_from_dataset`` semantics:
  * Observation  = the wrapped env's ``info['pixels']`` (224x224, what the model encodes).
  * Goal         = a real rendered frame of a reachable goal state (``info['goal']``).
  * Action       = the env's native ``action_space`` (Box[-1, 1]).
  * Success      = the env's native ``terminated`` flag / goal check.
  * Budget       = a fixed number of steps to reach the goal (like ``eval_budget``).

Goals are made reachable without needing the expert HDF5 datasets:
  * PushT / TwoRoom / Cube use the env's own goal sampled at reset.
  * Reacher rolls a throwaway copy forward to a real, reachable arm pose.
  * ``--goal rollout`` forces the roll-forward strategy for any task (a goal
    that is literally ``--steps-ahead`` env steps away from the start).

Usage:
    .venv/bin/python play.py --task pusht        # or tworoom / reacher / cube
    .venv/bin/python play.py --task puzzle4x4    # broad OGB tasks are playable too
    .venv/bin/python play.py --task reacher --goal rollout --steps-ahead 25
    .venv/bin/python play.py --list
    .venv/bin/python play.py --task pusht --selftest   # headless logic check, no window

Controls are drawn on screen; press H for the full legend. Esc quits, Q also
quits on tasks that do not bind Q as an action, R starts a new episode, N
switches task, Space takes a no-op step (watch the passive dynamics), T toggles
real-time vs. turn-based, [ / ] change the action magnitude, S saves a
screenshot.
"""

from __future__ import annotations

import argparse
import os
import sys

# MuJoCo offscreen backend must be chosen before dm_control / mujoco import.
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import warnings

import numpy as np
from PIL import Image

# The swm envs return torch tensors / float64 boxes that trip gymnasium's
# passive checker; the wrappers handle them fine, so quiet the noise.
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from project_paths import configure_stablewm_home

configure_stablewm_home()

import gymnasium as gym  # noqa: E402
import stable_worldmodel as swm  # noqa: E402  (registers swm/* envs)
from stable_worldmodel.wrapper import MegaWrapper  # noqa: E402

IMG = 224  # the resolution the model sees (matches eval img_size)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def last_frame(info: dict, key: str):
    """Return the most recent (H, W, 3) uint8 frame for a stacked info key."""
    v = info.get(key)
    if v is None:
        return None
    v = np.asarray(v)
    return v[-1] if v.ndim == 4 else v


def resize_rgb(img: np.ndarray, size: int = IMG) -> np.ndarray:
    """Convert an RGB-like image to contiguous uint8 HWC at model resolution."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        f = arr.astype(np.float32)
        if f.max(initial=0.0) <= 1.0:
            f = f * 255.0
        arr = np.clip(f, 0, 255).astype(np.uint8)
    if arr.shape[:2] != (size, size):
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(arr)


def make_env(env_id: str, make_kwargs: dict, budget: int) -> gym.Env:
    """Wrap a single env with the *exact* eval preprocessing pipeline."""
    env = gym.make(env_id, max_episode_steps=max(4 * budget, 200), **make_kwargs)
    return MegaWrapper(
        env, image_shape=(IMG, IMG), history_size=1, frame_skip=1, separate_goal=True
    )


def make_ogb_env(dataset_name: str, make_kwargs: dict) -> "OGBPlayWrapper":
    """Create an OGBench env and adapt observations to the play.py interface."""
    import ogbench

    env = ogbench.make_env_and_datasets(dataset_name, env_only=True, **make_kwargs)
    return OGBPlayWrapper(env)


class OGBPlayWrapper:
    """Normalize OGBench envs to the pixel-info interface used by play.py."""

    def __init__(self, env):
        self.env = env
        self.last_obs = None
        self.last_info = {}
        self.goal_img = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _obs_image(self, obs) -> np.ndarray:
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[-1] <= 3:
            return resize_rgb(arr)
        return resize_rgb(self.env.render())

    def _goal_image(self, info: dict) -> np.ndarray:
        goal_rendered = info.get("goal_rendered")
        if goal_rendered is not None:
            arr = np.asarray(goal_rendered)
            if arr.ndim == 3:
                return resize_rgb(arr)
        goal = info.get("goal")
        if goal is not None:
            arr = np.asarray(goal)
            if arr.ndim == 3:
                return resize_rgb(arr)
        return resize_rgb(self.env.render())

    def _normalize_info(self, obs, info: dict) -> dict:
        out = dict(info)
        out["pixels"] = self._obs_image(obs)
        if self.goal_img is None:
            self.goal_img = self._goal_image(out)
        out["goal"] = self.goal_img
        self.last_obs = obs
        self.last_info = out
        return out

    def reset(self, *args, **kwargs):
        self.goal_img = None
        obs, info = self.env.reset(*args, **kwargs)
        return obs, self._normalize_info(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._normalize_info(obs, info)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def should_handle_keydown(event, action_keys: set[int], noop_key: int) -> bool:
    """Allow key repeat only for controls that intentionally step the env."""
    return (
        not bool(getattr(event, "repeat", False))
        or event.key in action_keys
        or event.key == noop_key
    )


def is_quit_key(key: int, action_keys: set[int], escape_key: int, q_key: int) -> bool:
    """Return true for quit keys without stealing task-specific Q actions."""
    return key == escape_key or (key == q_key and key not in action_keys)


# --------------------------------------------------------------------------- #
# Task adapters                                                                #
# --------------------------------------------------------------------------- #
class Binding:
    """A keyboard key that pushes ``val`` onto action dimension ``dim``."""

    __slots__ = ("key", "dim", "val")

    def __init__(self, key: str, dim: int, val: float):
        self.key = key  # pygame attribute suffix, e.g. "LEFT" -> pygame.K_LEFT
        self.dim = dim
        self.val = val


class DiscreteBinding:
    """A keyboard key that selects one discrete action id."""

    __slots__ = ("key", "action")

    def __init__(self, key: str, action: int):
        self.key = key
        self.action = int(action)


class Task:
    """Base class describing one playable environment."""

    name: str = ""
    env_id: str = ""
    make_kwargs: dict = {}
    action_dim: int = 0
    default_budget: int = 50
    default_scale: float = 1.0
    default_goal: str = "native"  # "native" or "rollout"
    blurb: str = ""
    action_help: list[str] = []  # human-readable per-line key legend
    bindings: list[Binding] = []
    discrete_bindings: list[DiscreteBinding] = []

    # ---- construction ----------------------------------------------------- #
    def build(self, budget: int) -> gym.Env:
        return make_env(self.env_id, self.make_kwargs, budget)

    def action_key_names(self) -> list[str]:
        return [b.key for b in self.bindings] + [b.key for b in self.discrete_bindings]

    def noop_action(self):
        if self.discrete_bindings:
            return np.int64(0)
        return np.zeros(self.action_dim, dtype=np.float32)

    def coerce_action(self, action):
        if self.discrete_bindings:
            return np.int64(np.asarray(action).reshape(-1)[0])
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def display_action(self, action) -> np.ndarray:
        if self.discrete_bindings:
            return np.asarray([int(np.asarray(action).reshape(-1)[0])], dtype=np.float32)
        return np.asarray(action, dtype=np.float32)

    def action_from_pressed(self, pressed, pygame, scale: float):
        if self.discrete_bindings:
            for b in self.discrete_bindings:
                if pressed[getattr(pygame, "K_" + b.key)]:
                    return np.int64(b.action)
            return np.int64(0)
        a = np.zeros(self.action_dim, dtype=np.float32)
        for b in self.bindings:
            if pressed[getattr(pygame, "K_" + b.key)]:
                a[b.dim] += b.val
        a = np.clip(a, -1.0, 1.0) * scale
        return np.clip(a, -1.0, 1.0)

    # ---- goal handling (override per task) -------------------------------- #
    def read_goal_repr(self, env: gym.Env) -> dict:
        """Snapshot the goal-defining quantity from a rolled-forward env."""
        raise NotImplementedError

    def install_goal(self, env: gym.Env, repr_: dict) -> None:
        """Wire a goal snapshot into the primary env's success machinery."""
        raise NotImplementedError

    def native_goal_image(self, env: gym.Env, info: dict, seed: int):
        """Return a goal frame for the env's *own* reset-sampled goal."""
        raise NotImplementedError

    def status(self, env: gym.Env) -> tuple[bool, float, str]:
        """Return (success, distance_to_goal, distance_label) for the HUD."""
        raise NotImplementedError

    # ---- shared goal preparation ------------------------------------------ #
    def prepare_goal(self, env, info, seed: int, steps_ahead: int, kind: str):
        """Set up the goal and return its image. Leaves ``env`` at the start."""
        if kind == "rollout":
            sec = self.build(self.default_budget)
            sec.reset(seed=seed)
            rng = np.random.default_rng(seed + 7919)
            ginfo = None
            for _ in range(steps_ahead):
                if self.discrete_bindings and hasattr(sec.action_space, "n"):
                    a = np.int64(rng.integers(sec.action_space.n))
                else:
                    a = rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
                _, _, term, _, ginfo = sec.step(a)
                if term:
                    break
            goal_img = last_frame(ginfo, "pixels").copy()
            repr_ = self.read_goal_repr(sec)
            sec.close()
            self.install_goal(env, repr_)
            return goal_img
        # native
        return self.native_goal_image(env, info, seed)


class PushTTask(Task):
    name = "pusht"
    env_id = "swm/PushT-v1"
    make_kwargs = {}
    action_dim = 2
    default_budget = 60
    default_scale = 0.5
    default_goal = "native"
    blurb = "Push the gray block so its pose matches the faint goal outline."
    action_help = [
        "Arrows  move the pusher (the small dot)",
        "  the action is a relative target offset (Box[-1,1]^2)",
    ]
    bindings = [
        Binding("LEFT", 0, -1.0),
        Binding("RIGHT", 0, +1.0),
        Binding("UP", 1, -1.0),
        Binding("DOWN", 1, +1.0),
    ]

    def native_goal_image(self, env, info, seed):
        # reset() already sampled a goal pose, set goal_state, and rendered it.
        return last_frame(info, "goal").copy()

    def read_goal_repr(self, env):
        return {"state": np.asarray(env.unwrapped._get_obs(), dtype=float).copy()}

    def install_goal(self, env, repr_):
        env.unwrapped._set_goal_state(repr_["state"])

    def status(self, env):
        u = env.unwrapped
        succ, dist = u.eval_state(u.goal_state, u._get_obs())
        return bool(succ), float(dist), "state L2 dist"


class TwoRoomTask(Task):
    name = "tworoom"
    env_id = "swm/TwoRoom-v1"
    make_kwargs = {}
    action_dim = 2
    default_budget = 80
    default_scale = 1.0
    default_goal = "native"
    blurb = "Navigate the dot through the doorway to the goal location."
    action_help = [
        "Arrows  move the dot (2-D velocity, Box[-1,1]^2)",
    ]
    bindings = [
        Binding("LEFT", 0, -1.0),
        Binding("RIGHT", 0, +1.0),
        Binding("UP", 1, -1.0),
        Binding("DOWN", 1, +1.0),
    ]

    def native_goal_image(self, env, info, seed):
        u = env.unwrapped
        img = u._render_frame(agent_pos=u.target_position).cpu().numpy()
        return np.ascontiguousarray(img.transpose(1, 2, 0))  # CHW -> HWC

    def read_goal_repr(self, env):
        return {"agent": env.unwrapped.agent_position.detach().cpu().numpy().copy()}

    def install_goal(self, env, repr_):
        env.unwrapped._set_goal_state(repr_["agent"])

    def status(self, env):
        u = env.unwrapped
        d = float(np.linalg.norm(u.agent_position.numpy() - u.target_position.numpy()))
        return d < 16.0, d, "px to target"


class ReacherTask(Task):
    name = "reacher"
    env_id = "swm/ReacherDMControl-v0"
    make_kwargs = {"task": "qpos_match"}
    action_dim = 2
    default_budget = 50
    default_scale = 1.0
    default_goal = "rollout"  # no native goal; roll forward to a reachable pose
    blurb = "Torque the 2-joint arm so it matches the goal arm configuration."
    action_help = [
        "Left/Right  shoulder-joint torque -/+",
        "Up/Down     wrist-joint torque +/-   (Box[-1,1]^2)",
    ]
    bindings = [
        Binding("LEFT", 0, -1.0),
        Binding("RIGHT", 0, +1.0),
        Binding("UP", 1, +1.0),
        Binding("DOWN", 1, -1.0),
    ]

    def native_goal_image(self, env, info, seed):
        # Reacher has no env-native goal; always use the rollout path.
        return self.prepare_goal(env, info, seed, steps_ahead=25, kind="rollout")

    def read_goal_repr(self, env):
        return {"qpos": env.unwrapped.env.physics.data.qpos.copy()}

    def install_goal(self, env, repr_):
        env.unwrapped.set_target_qpos(repr_["qpos"])

    def status(self, env):
        u = env.unwrapped
        target = u.env.task.target_qpos
        if target is None:
            return False, float("nan"), "max|dqpos| rad"
        d = float(np.max(np.abs(u.env.physics.data.qpos - target)))
        return d < 0.05, d, "max|dqpos| rad"


class CubeTask(Task):
    name = "cube"
    env_id = "swm/OGBCube-v0"
    make_kwargs = {
        "env_type": "single",
        "ob_type": "states",
        "width": IMG,
        "height": IMG,
        "terminate_at_goal": True,
    }
    action_dim = 5
    default_budget = 120
    default_scale = 1.0
    default_goal = "native"
    blurb = "5-DOF arm: move the cube onto the translucent target marker. (Hard!)"
    action_help = [
        "Left/Right  end-effector x -/+      Up/Down  y +/-",
        "Q/E  z up/down      Z/C  yaw -/+      X  close gripper",
    ]
    bindings = [
        Binding("LEFT", 0, -1.0),
        Binding("RIGHT", 0, +1.0),
        Binding("UP", 1, +1.0),
        Binding("DOWN", 1, -1.0),
        Binding("q", 2, +1.0),
        Binding("e", 2, -1.0),
        Binding("z", 3, -1.0),
        Binding("c", 3, +1.0),
        Binding("x", 4, -1.0),  # close; released gripper drifts open
    ]

    def native_goal_image(self, env, info, seed):
        import mujoco

        sec = self.build(self.default_budget)
        sec.reset(seed=seed)
        u = sec.unwrapped
        tgt = u._data.mocap_pos[u._cube_target_mocap_ids[0]].copy()
        u._data.joint("object_joint_0").qpos[:3] = tgt
        mujoco.mj_forward(u._model, u._data)
        _, _, _, _, ginfo = sec.step(np.zeros(self.action_dim, dtype=np.float32))
        img = last_frame(ginfo, "pixels").copy()
        sec.close()
        return img

    def read_goal_repr(self, env):
        u = env.unwrapped
        qpos = u._data.joint("object_joint_0").qpos
        return {"pos": qpos[:3].copy(), "quat": qpos[3:7].copy()}

    def install_goal(self, env, repr_):
        env.unwrapped.set_target_pos(0, repr_["pos"], repr_["quat"])

    def status(self, env):
        u = env.unwrapped
        cube = u._data.joint("object_joint_0").qpos[:3]
        tgt = u._data.mocap_pos[u._cube_target_mocap_ids[0]]
        d = float(np.linalg.norm(cube - tgt))
        return d < 0.04, d, "m to target"


class OGBTask(Task):
    """Playable OGBench task using the env's native reset goal."""

    dataset_name: str = ""
    make_kwargs: dict = {}

    def build(self, budget: int) -> gym.Env:
        return make_ogb_env(self.dataset_name, self.make_kwargs)

    def native_goal_image(self, env, info, seed):
        return last_frame(info, "goal").copy()

    def read_goal_repr(self, env):
        info = getattr(env, "last_info", {})
        repr_ = {"goal_img": last_frame(info, "pixels").copy()}
        if "button_states" in info:
            repr_["button_states"] = np.asarray(info["button_states"]).copy()
        if "qpos" in info:
            repr_["qpos"] = np.asarray(info["qpos"]).copy()
        return repr_

    def install_goal(self, env, repr_):
        env.goal_img = repr_["goal_img"]
        u = env.unwrapped
        if "button_states" in repr_ and hasattr(u, "_target_button_states"):
            u._target_button_states = np.asarray(repr_["button_states"]).copy()
            u._target_task = "all"
        elif "qpos" in repr_ and hasattr(u, "set_goal"):
            qpos = np.asarray(repr_["qpos"])
            xy = qpos[-7:-5] if "soccer" in self.name else qpos[:2]
            u.set_goal(goal_xy=xy)

    def status(self, env):
        info = getattr(env, "last_info", {})
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        return success, float("nan"), "native success"


class OGBPuzzleTask(OGBTask):
    action_dim = 5
    default_budget = 150
    default_scale = 1.0
    default_goal = "native"
    action_help = [
        "Arrow keys  end-effector x/y       Q/E  z up/down",
        "Z/C  yaw -/+      X  close gripper / press buttons",
    ]
    bindings = CubeTask.bindings

    def status(self, env):
        u = env.unwrapped
        current = np.asarray(getattr(u, "_cur_button_states", []), dtype=int)
        target = np.asarray(getattr(u, "_target_button_states", []), dtype=int)
        if current.size and target.size and current.shape == target.shape:
            mismatch = int(np.count_nonzero(current != target))
            return mismatch == 0, float(mismatch), "buttons mismatched"
        return super().status(env)


class Puzzle4x4Task(OGBPuzzleTask):
    name = "puzzle4x4"
    dataset_name = "visual-puzzle-4x4-play-v0"
    blurb = "Press buttons in the 4x4 Lights Out grid until it matches the goal."


class Puzzle4x5Task(OGBPuzzleTask):
    name = "puzzle4x5"
    dataset_name = "visual-puzzle-4x5-play-v0"
    blurb = "Press buttons in the 4x5 Lights Out grid until it matches the goal."


class OGBAntTask(OGBTask):
    action_dim = 8
    default_budget = 200
    default_scale = 0.7
    default_goal = "native"
    action_help = [
        "Ant torque pairs: Q/W a0, A/G a1, E/C a2, D/F a3",
        "                 U/I a4, J/K a5, O/P a6, Z/X a7",
    ]
    bindings = [
        Binding("q", 0, -1.0), Binding("w", 0, +1.0),
        Binding("a", 1, -1.0), Binding("g", 1, +1.0),
        Binding("e", 2, -1.0), Binding("c", 2, +1.0),
        Binding("d", 3, -1.0), Binding("f", 3, +1.0),
        Binding("u", 4, -1.0), Binding("i", 4, +1.0),
        Binding("j", 5, -1.0), Binding("k", 5, +1.0),
        Binding("o", 6, -1.0), Binding("p", 6, +1.0),
        Binding("z", 7, -1.0), Binding("x", 7, +1.0),
    ]

    def status(self, env):
        info = getattr(env, "last_info", {})
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        goal = getattr(env.unwrapped, "cur_goal_xy", None)
        xy = info.get("xy")
        if goal is not None and xy is not None:
            d = float(np.linalg.norm(np.asarray(xy) - np.asarray(goal)))
            return success or d <= 0.5, d, "xy to goal"
        return success, float("nan"), "native success"


class AntmazeTeleportTask(OGBAntTask):
    name = "antmaze_teleport"
    dataset_name = "visual-antmaze-teleport-navigate-v0"
    blurb = "Drive the ant through the maze and teleport zones to the visual goal."


class AntmazeLargeTask(OGBAntTask):
    name = "antmaze_large"
    dataset_name = "visual-antmaze-large-stitch-v0"
    blurb = "Drive the ant through the large maze to the visual goal."


class AntsoccerTask(OGBAntTask):
    name = "antsoccer"
    dataset_name = "antsoccer-medium-stitch-v0"
    blurb = "Use ant torques to push the ball onto the target disk."

    def status(self, env):
        info = getattr(env, "last_info", {})
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        u = env.unwrapped
        if hasattr(u, "get_agent_ball_xy") and hasattr(u, "cur_goal_xy"):
            _, ball_xy = u.get_agent_ball_xy()
            d = float(np.linalg.norm(ball_xy - u.cur_goal_xy))
            return success or d <= 0.5, d, "ball to goal"
        return success, float("nan"), "native success"


class PowderworldTask(OGBTask):
    name = "powderworld"
    dataset_name = "powderworld-medium-play-v0"
    action_dim = 1
    default_budget = 200
    default_scale = 1.0
    default_goal = "native"
    blurb = "Paint and simulate powder elements until the grid matches the goal."
    action_help = [
        "Discrete actions 0-7. Each three presses choose element, x cell, y cell.",
        "Keys 1-8 choose action ids 0-7; Space sends action 0.",
    ]
    discrete_bindings = [
        DiscreteBinding("1", 0), DiscreteBinding("2", 1),
        DiscreteBinding("3", 2), DiscreteBinding("4", 3),
        DiscreteBinding("5", 4), DiscreteBinding("6", 5),
        DiscreteBinding("7", 6), DiscreteBinding("8", 7),
    ]

    def status(self, env):
        info = getattr(env, "last_info", {})
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        obs = resize_rgb(getattr(env, "last_obs"))
        goal = last_frame(info, "goal")
        diff = np.abs(obs.astype(np.int16) - goal.astype(np.int16))
        mismatch = int((diff > 8).any(axis=-1).sum())
        return success, float(mismatch), "pixel mismatches"


TASKS: dict[str, Task] = {
    t.name: t for t in (
        PushTTask(),
        TwoRoomTask(),
        ReacherTask(),
        CubeTask(),
        Puzzle4x4Task(),
        Puzzle4x5Task(),
        AntmazeTeleportTask(),
        PowderworldTask(),
        AntmazeLargeTask(),
        AntsoccerTask(),
    )
}


# Global one-shot controls are matched in the event loop *before* per-task
# action keys, so any task that binds one of these never reaches the action
# handler (e.g. "r" fired reset instead of an ant torque). "q" is exempt: the
# quit handler defers to it when a task binds it as an action key.
_RESERVED_PLAY_KEYS = {"r", "s", "t", "h", "n", "tab", "space", "escape",
                       "leftbracket", "rightbracket"}
for _task in TASKS.values():
    _clashes = sorted(set(_task.action_key_names()) & _RESERVED_PLAY_KEYS)
    if _clashes:
        raise ValueError(
            f"task {_task.name!r} binds reserved control key(s) {_clashes}; "
            "they are intercepted as global controls and never reach the action "
            "handler — choose different keys"
        )


# --------------------------------------------------------------------------- #
# Episode state                                                               #
# --------------------------------------------------------------------------- #
class Episode:
    """Holds the live env, goal image, and step bookkeeping for one attempt."""

    def __init__(self, task: Task, seed: int, goal_kind: str, steps_ahead: int, budget: int):
        self.task = task
        self.seed = seed
        self.budget = budget
        self.env = task.build(budget)
        _, self.info = self.env.reset(seed=seed)
        self.goal_img = task.prepare_goal(self.env, self.info, seed, steps_ahead, goal_kind)
        self.steps = 0
        self.solved = False
        self.last_action = task.display_action(task.noop_action())
        self.success, self.distance, self.dist_label = task.status(self.env)

    @property
    def obs_img(self):
        return last_frame(self.info, "pixels")

    def step(self, action: np.ndarray):
        if self.solved:
            return
        env_action = self.task.coerce_action(action)
        _, _, term, trunc, self.info = self.env.step(env_action)
        self.steps += 1
        self.last_action = self.task.display_action(env_action)
        self.success, self.distance, self.dist_label = self.task.status(self.env)
        if self.success or bool(term):
            self.solved = True

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Headless self-test (no window) — validates logic + saves a preview PNG       #
# --------------------------------------------------------------------------- #
def run_selftest(task_names: list[str], seed: int, goal_kind: str, steps_ahead: int):
    from PIL import Image

    out_paths = []
    for name in task_names:
        task = TASKS[name]
        kind = task.default_goal if goal_kind == "auto" else goal_kind
        ep = Episode(task, seed, kind, steps_ahead, task.default_budget)
        obs0 = ep.obs_img
        # Drive a few steps: for reacher, P-control toward the goal to prove
        # the success check fires end-to-end; otherwise random probing.
        rng = np.random.default_rng(seed)
        for _ in range(40):
            if name == "reacher":
                u = ep.env.unwrapped
                err = u.env.task.target_qpos - u.env.physics.data.qpos
                a = np.clip(8.0 * err, -1, 1).astype(np.float32)
            elif task.discrete_bindings:
                a = np.int64(rng.integers(0, len(task.discrete_bindings)))
            else:
                a = (0.4 * rng.uniform(-1, 1, size=task.action_dim)).astype(np.float32)
            ep.step(a)
            if ep.solved:
                break
        gap = np.full((IMG, 8, 3), 32, dtype=np.uint8)
        strip = np.hstack([obs0, gap, ep.obs_img, gap, ep.goal_img]).astype(np.uint8)
        path = f"/tmp/play_selftest_{name}.png"
        Image.fromarray(strip).save(path)
        out_paths.append(path)
        print(
            f"[{name:8s}] obs={obs0.shape} goal={ep.goal_img.shape} act_dim={task.action_dim} "
            f"goal={kind:7s} steps={ep.steps:3d} solved={ep.solved} "
            f"dist={ep.distance:.3f} ({ep.dist_label}) -> {path}"
        )
        ep.close()
    print("\nself-test OK. Panels are: start-obs | current-obs | goal.")
    return out_paths


# --------------------------------------------------------------------------- #
# Interactive pygame UI                                                        #
# --------------------------------------------------------------------------- #
def run_interactive(start_task: str, seed: int, goal_kind: str, steps_ahead: int, realtime: bool):
    try:
        import pygame
    except ImportError:
        sys.exit("pygame is required for the interactive demo: pip install pygame")

    SCALE = 2  # display upscaling of the 224x224 frames
    PANEL = IMG * SCALE
    MARGIN = 18
    TOP = 44
    HUD = 232
    W = MARGIN * 3 + PANEL * 2
    H = TOP + PANEL + HUD

    pygame.init()
    pygame.display.set_caption("World-Model Human Play")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont("menlo,consolas,monospace", 22, bold=True)
    f_med = pygame.font.SysFont("menlo,consolas,monospace", 16)
    f_small = pygame.font.SysFont("menlo,consolas,monospace", 14)
    pygame.key.set_repeat(220, 55)  # hold-to-repeat for turn-based stepping

    COL_BG = (18, 18, 22)
    COL_FG = (230, 230, 235)
    COL_DIM = (150, 150, 160)
    COL_OK = (90, 210, 120)
    COL_WARN = (235, 180, 70)
    COL_BAD = (235, 90, 90)
    COL_ACC = (110, 170, 255)

    task_order = list(TASKS.keys())
    state = {"task": start_task, "seed": seed, "show_help": False, "realtime": realtime}

    def resolve_kind(task: Task) -> str:
        return task.default_goal if goal_kind == "auto" else goal_kind

    def new_episode(task_name: str, seed_val: int) -> Episode:
        task = TASKS[task_name]
        return Episode(task, seed_val, resolve_kind(task), steps_ahead, task.default_budget)

    def loading_screen(msg: str):
        screen.fill(COL_BG)
        t = f_big.render(msg, True, COL_FG)
        screen.blit(t, (MARGIN, H // 2 - 16))
        pygame.display.flip()

    loading_screen("Building environment...")
    ep = new_episode(state["task"], state["seed"])
    key_set = {getattr(pygame, "K_" + key) for key in ep.task.action_key_names()}

    def build_action():
        pressed = pygame.key.get_pressed()
        return ep.task.action_from_pressed(pressed, pygame, state["scale"])

    def blit_frame(img: np.ndarray, x: int, y: int):
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        surf = pygame.transform.scale(surf, (PANEL, PANEL))
        screen.blit(surf, (x, y))

    def text(s, x, y, font=f_med, col=COL_FG):
        screen.blit(font.render(s, True, col), (x, y))

    state["scale"] = ep.task.default_scale
    running = True
    while running:
        do_step = False
        step_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                # Environment construction can take long enough for repeated
                # N/R/etc. events to queue up. Those controls are one-shot;
                # only action keys and Space should repeat while held.
                if not should_handle_keydown(event, key_set, pygame.K_SPACE):
                    continue
                if is_quit_key(k, key_set, pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_r:
                    ep.close()
                    loading_screen("Resetting...")
                    state["seed"] += 1
                    ep = new_episode(state["task"], state["seed"])
                    key_set = {getattr(pygame, "K_" + key) for key in ep.task.action_key_names()}
                    state["scale"] = ep.task.default_scale
                elif k in (pygame.K_n, pygame.K_TAB):
                    ep.close()
                    idx = (task_order.index(state["task"]) + 1) % len(task_order)
                    state["task"] = task_order[idx]
                    loading_screen(f"Loading {state['task']}...")
                    ep = new_episode(state["task"], state["seed"])
                    key_set = {getattr(pygame, "K_" + key) for key in ep.task.action_key_names()}
                    state["scale"] = ep.task.default_scale
                elif k == pygame.K_t:
                    state["realtime"] = not state["realtime"]
                elif k == pygame.K_h:
                    state["show_help"] = not state["show_help"]
                elif k == pygame.K_LEFTBRACKET:
                    state["scale"] = max(0.1, round(state["scale"] - 0.1, 2))
                elif k == pygame.K_RIGHTBRACKET:
                    state["scale"] = min(1.0, round(state["scale"] + 0.1, 2))
                elif k == pygame.K_s:
                    p = f"/tmp/play_{state['task']}_{ep.steps:03d}.png"
                    pygame.image.save(screen, p)
                    print(f"screenshot -> {p}")
                elif k == pygame.K_SPACE and not state["realtime"]:
                    do_step = True
                    step_action = ep.task.noop_action()
                elif k in key_set and not state["realtime"]:
                    do_step = True  # action sampled from held keys below

        if state["realtime"]:
            do_step = True

        if do_step and not ep.solved:
            act = step_action if step_action is not None else build_action()
            ep.step(act)

        # ----- draw ----- #
        screen.fill(COL_BG)
        task = ep.task
        over = ep.steps > ep.budget
        text(f"TASK  {task.name.upper()}", MARGIN, 12, f_big, COL_ACC)
        mode = "REAL-TIME" if state["realtime"] else "TURN-BASED"
        text(
            f"{mode}   goal:{resolve_kind(task)}   mag:{state['scale']:.1f}",
            MARGIN + 260, 16, f_med, COL_DIM,
        )

        blit_frame(ep.obs_img, MARGIN, TOP)
        blit_frame(ep.goal_img, MARGIN * 2 + PANEL, TOP)
        text("OBSERVATION (what the model sees)", MARGIN, TOP + PANEL + 4, f_small, COL_DIM)
        text("GOAL", MARGIN * 2 + PANEL, TOP + PANEL + 4, f_small, COL_DIM)

        hud_y = TOP + PANEL + 26
        # step / budget
        step_col = COL_BAD if over else COL_FG
        text(f"step {ep.steps} / {ep.budget}" + ("  (over budget)" if over else ""),
             MARGIN, hud_y, f_med, step_col)
        # distance + success
        if ep.solved:
            text("GOAL REACHED  ✓   press R for a new episode", MARGIN + 300, hud_y, f_med, COL_OK)
        else:
            dcol = COL_WARN
            text(f"{ep.dist_label}: {ep.distance:.3f}", MARGIN + 300, hud_y, f_med, dcol)

        # last action vector + bars
        ay = hud_y + 26
        text("action " + np.array2string(ep.last_action, precision=2, suppress_small=True,
             floatmode="fixed"), MARGIN, ay, f_med, COL_FG)
        bx = MARGIN + 360
        for i, v in enumerate(ep.last_action):
            cx = bx + i * 60
            pygame.draw.rect(screen, (40, 40, 48), (cx, ay, 50, 16))
            w = int(abs(v) * 25)
            x0 = cx + 25
            pygame.draw.rect(screen, COL_ACC, (x0 if v >= 0 else x0 - w, ay, max(1, w), 16))
            pygame.draw.line(screen, COL_DIM, (cx + 25, ay), (cx + 25, ay + 16))

        # blurb + key legend
        ly = ay + 28
        text(task.blurb, MARGIN, ly, f_small, COL_DIM)
        for j, line in enumerate(task.action_help):
            text(line, MARGIN, ly + 20 + j * 18, f_small, COL_FG)

        gy = ly + 20 + len(task.action_help) * 18 + 6
        quit_hint = "Esc quit" if pygame.K_q in key_set else "Esc/Q quit"
        text("R reset   N next-task   Space no-op step   T turn/real   [ ] magnitude   "
             f"S shot   H help   {quit_hint}", MARGIN, gy, f_small, COL_DIM)

        if state["show_help"]:
            text("Each keypress = one env.step (hold to repeat). The episode runs until you",
                 MARGIN, gy + 22, f_small, COL_ACC)
            text("reach the goal or you choose to reset — like the planner's eval_budget.",
                 MARGIN, gy + 40, f_small, COL_ACC)

        pygame.display.flip()
        clock.tick(30)

    ep.close()
    pygame.quit()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Play the world-model tasks as the AI would.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", default="pusht", choices=list(TASKS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal", default="auto", choices=["auto", "native", "rollout"],
                   help="goal source; 'auto' uses each task's default")
    p.add_argument("--steps-ahead", type=int, default=25,
                   help="rollout horizon used to build reachable goals")
    p.add_argument("--realtime", action="store_true",
                   help="step continuously while keys are held (default: turn-based)")
    p.add_argument("--list", action="store_true", help="list tasks and exit")
    p.add_argument("--selftest", action="store_true",
                   help="headless logic check (+ preview PNGs), no window")
    p.add_argument("--selftest-all", action="store_true",
                   help="run the headless self-test for every task")
    args = p.parse_args()

    if args.list:
        print("Available tasks:\n")
        for t in TASKS.values():
            print(f"  {t.name:9s} act_dim={t.action_dim}  budget={t.default_budget}  "
                  f"goal={t.default_goal}\n      {t.blurb}")
        return

    if args.selftest or args.selftest_all:
        names = list(TASKS.keys()) if args.selftest_all else [args.task]
        run_selftest(names, args.seed, args.goal, args.steps_ahead)
        return

    run_interactive(args.task, args.seed, args.goal, args.steps_ahead, args.realtime)


if __name__ == "__main__":
    main()
