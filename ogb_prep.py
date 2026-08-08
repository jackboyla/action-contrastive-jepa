"""Prepare generic OGBench image datasets for LeWM training.

The existing repo has first-class OGB Cube and Scene conversions. This module is
for one-off broader OGBench comparisons where the policy still expects RGB
pixels and vector actions in the stable-worldmodel HDF5 layout.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OGBTaskSpec:
    requested_name: str
    source_dataset: str
    output_name: str
    env_dataset: str
    eval_solver: str = "cem"
    discrete_action_n: int | None = None
    render_from_state: bool = False
    render_max_rows: int | None = None
    synthetic_pixels: str | None = None
    env_kwargs: dict[str, object] = field(default_factory=dict)


OGB_TASK_SPECS: dict[str, OGBTaskSpec] = {
    "puzzle-4x4-play-v0": OGBTaskSpec(
        requested_name="puzzle-4x4-play-v0",
        source_dataset="visual-puzzle-4x4-play-v0",
        output_name="ogbench/visual_puzzle_4x4_play",
        env_dataset="visual-puzzle-4x4-play-v0",
    ),
    "puzzle-4x5-play-v0": OGBTaskSpec(
        requested_name="puzzle-4x5-play-v0",
        source_dataset="visual-puzzle-4x5-play-v0",
        output_name="ogbench/visual_puzzle_4x5_play",
        env_dataset="visual-puzzle-4x5-play-v0",
    ),
    "antmaze-teleport-navigate-v0": OGBTaskSpec(
        requested_name="antmaze-teleport-navigate-v0",
        source_dataset="visual-antmaze-teleport-navigate-v0",
        output_name="ogbench/visual_antmaze_teleport_navigate",
        env_dataset="visual-antmaze-teleport-navigate-v0",
    ),
    "powderworld-medium-play-v0": OGBTaskSpec(
        requested_name="powderworld-medium-play-v0",
        source_dataset="powderworld-medium-play-v0",
        output_name="ogbench/powderworld_medium_play_rgb",
        env_dataset="powderworld-medium-play-v0",
        eval_solver="pgd",
        discrete_action_n=8,
    ),
    "antmaze-large-stitch-v0": OGBTaskSpec(
        requested_name="antmaze-large-stitch-v0",
        source_dataset="visual-antmaze-large-stitch-v0",
        output_name="ogbench/visual_antmaze_large_stitch",
        env_dataset="visual-antmaze-large-stitch-v0",
    ),
    "antsoccer-medium-stitch-v0": OGBTaskSpec(
        requested_name="antsoccer-medium-stitch-v0",
        source_dataset="antsoccer-medium-stitch-v0",
        output_name="ogbench/antsoccer_medium_stitch_topdown32_250k",
        env_dataset="antsoccer-medium-stitch-v0",
        render_max_rows=250_000,
        synthetic_pixels="antsoccer_topdown",
        env_kwargs={"ob_type": "states"},
    ),
}


def get_ogb_task_spec(task: str) -> OGBTaskSpec:
    if task not in OGB_TASK_SPECS:
        raise ValueError(
            f"Unknown OGB task {task!r}; expected one of {sorted(OGB_TASK_SPECS)}"
        )
    return OGB_TASK_SPECS[task]


def _episode_lengths(terminals):
    import numpy as np

    terminal_idx = np.flatnonzero(np.asarray(terminals, dtype=bool))
    if terminal_idx.size == 0:
        raise ValueError("OGBench dataset has no terminal markers")
    starts = np.concatenate([[0], terminal_idx[:-1] + 1]).astype(np.int64)
    lengths = (terminal_idx - starts + 1).astype(np.int64)
    if np.any(lengths <= 0):
        raise ValueError(f"Invalid OGB episode lengths: {lengths}")
    return starts, lengths


def _rgb_from_observations(observations):
    import numpy as np

    if observations.ndim != 4:
        raise ValueError(f"Expected image observations, got shape {observations.shape}")
    if observations.shape[-1] == 3:
        return observations
    if observations.shape[-1] == 6:
        # Powderworld stores RGB plus an action-overlay frame. The first three
        # channels are exactly the env's RGB render at the native 32x32 grid.
        return observations[..., :3]
    raise ValueError(f"Expected 3 or 6 image channels, got shape {observations.shape}")


def _one_hot_actions(actions, n: int):
    import numpy as np

    flat = np.asarray(actions, dtype=np.int64).reshape(-1)
    if np.any(flat < 0) or np.any(flat >= n):
        raise ValueError(f"Discrete actions out of range [0,{n}): {flat.min()}..{flat.max()}")
    out = np.zeros((flat.shape[0], n), dtype=np.float32)
    out[np.arange(flat.shape[0]), flat] = 1.0
    return out


def _continuous_actions(actions):
    import numpy as np

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _truncate_to_whole_episodes(dataset: dict, max_rows: int | None) -> dict:
    import numpy as np

    if max_rows is None:
        return dataset

    starts, lengths = _episode_lengths(dataset["terminals"])
    cumulative = np.cumsum(lengths)
    num_episodes = int(np.searchsorted(cumulative, max_rows, side="right"))
    if num_episodes == 0:
        num_episodes = 1
    row_count = int(cumulative[num_episodes - 1])
    if row_count >= int(cumulative[-1]):
        return dataset
    if int(starts[0]) != 0:
        raise ValueError("Cannot truncate OGB dataset with non-zero first episode start")

    print(
        f"Truncating {row_count}/{int(cumulative[-1])} rows "
        f"({num_episodes}/{len(lengths)} whole episodes)",
        flush=True,
    )
    return {key: value[:row_count] for key, value in dataset.items()}


def _render_frame_shape(spec: OGBTaskSpec) -> tuple[int, int, int]:
    height = int(spec.env_kwargs.get("height", 64))
    width = int(spec.env_kwargs.get("width", 64))
    if spec.synthetic_pixels == "antsoccer_topdown":
        height = width = 32
    return height, width, 3


def _antsoccer_topdown_pixels_from_qpos(qpos, size: int = 32):
    import numpy as np

    qpos = np.asarray(qpos)
    agent_xy = qpos[:, :2]
    ball_xy = qpos[:, -7:-5]
    coords = np.stack([agent_xy, ball_xy], axis=1)
    # OGB antsoccer medium coordinates are maze-cell positions roughly in
    # [0, 22]. Use a slightly wider fixed box so train/eval rasterization is
    # deterministic and does not depend on batch extrema.
    lo = np.array([-2.0, -2.0], dtype=np.float32)
    hi = np.array([24.0, 24.0], dtype=np.float32)
    xy = np.clip((coords - lo) / (hi - lo), 0.0, 1.0)
    px = np.rint(xy[..., 0] * (size - 1)).astype(np.int64)
    py = np.rint((1.0 - xy[..., 1]) * (size - 1)).astype(np.int64)

    pixels = np.full((qpos.shape[0], size, size, 3), 18, dtype=np.uint8)
    pixels[:, :, :, 1] = 22
    pixels[:, :, :, 2] = 26

    yy, xx = np.ogrid[:size, :size]
    colors = (
        np.array([80, 180, 255], dtype=np.uint8),
        np.array([255, 110, 80], dtype=np.uint8),
    )
    radii = (2, 2)
    for idx in range(qpos.shape[0]):
        for obj_idx in range(2):
            mask = (xx - px[idx, obj_idx]) ** 2 + (yy - py[idx, obj_idx]) ** 2 <= radii[obj_idx] ** 2
            pixels[idx, mask] = colors[obj_idx]
    return pixels


def _synthetic_pixels_from_state(spec: OGBTaskSpec, dataset: dict):
    if spec.synthetic_pixels == "antsoccer_topdown":
        return _antsoccer_topdown_pixels_from_qpos(dataset["qpos"])
    raise ValueError(f"Unknown synthetic pixel source: {spec.synthetic_pixels}")


def _write_rendered_pixels_from_qpos_qvel(
    spec: OGBTaskSpec,
    dataset: dict,
    pixel_dataset,
) -> None:
    import numpy as np
    import ogbench

    if "qpos" not in dataset or "qvel" not in dataset:
        raise KeyError(
            f"{spec.source_dataset} is state-based and requires qpos/qvel for "
            "pixel rendering, but the OGB dataset did not provide them"
        )

    env = ogbench.make_env_and_datasets(
        spec.env_dataset,
        env_only=True,
        **spec.env_kwargs,
    )
    observations = dataset["observations"]
    qpos = dataset["qpos"]
    qvel = dataset["qvel"]
    expected_shape = _render_frame_shape(spec)
    total = int(observations.shape[0])
    render_batch_size = 4096
    try:
        env.reset(seed=0)
        for start in range(0, total, render_batch_size):
            end = min(start + render_batch_size, total)
            batch = np.empty((end - start, *expected_shape), dtype=np.uint8)
            for offset, idx in enumerate(range(start, end)):
                env.unwrapped.set_state(qpos[idx], qvel[idx])
                frame = env.render()
                if frame is None:
                    frame = env.unwrapped.get_ob()
                if frame.shape[:3] != expected_shape:
                    raise ValueError(
                        f"Expected rendered frame shape {expected_shape}, got {frame.shape}"
                    )
                batch[offset] = np.asarray(frame[..., :3], dtype=np.uint8)
            pixel_dataset[start:end] = batch
            if end % 10000 < render_batch_size or end == total:
                print(
                    f"rendered {end}/{total} state frames "
                    f"for {spec.source_dataset}",
                    flush=True,
                )
    finally:
        env.close()


def _wait_for_existing(output_path: Path, lock_dir: Path, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if output_path.exists():
            print(f"Prepared OGB HDF5 appeared while waiting: {output_path}", flush=True)
            return
        if not lock_dir.exists():
            break
        print(f"Waiting for OGB conversion lock: {lock_dir}", flush=True)
        time.sleep(60)
    if output_path.exists():
        return
    raise TimeoutError(f"Timed out waiting for {output_path}")


def prepare_ogb_hdf5(
    *,
    task: str,
    cache_dir: str | Path,
    force: bool = False,
    lock_timeout_s: int = 12 * 60 * 60,
) -> Path:
    """Download/convert one OGBench task to stable-worldmodel HDF5."""

    import h5py
    import numpy as np
    import ogbench

    spec = get_ogb_task_spec(task)
    cache_root = Path(cache_dir)
    output_path = cache_root / f"{spec.output_name}.h5"
    lock_dir = cache_root / f"{spec.output_name}.lock"

    if output_path.exists() and not force:
        print(f"OGB HDF5 exists, skipping: {output_path}", flush=True)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
        have_lock = True
    except FileExistsError:
        have_lock = False

    if not have_lock:
        _wait_for_existing(output_path, lock_dir, lock_timeout_s)
        return output_path

    partial_path = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.partial")
    try:
        if output_path.exists() and not force:
            return output_path
        if partial_path.exists():
            partial_path.unlink()

        dataset_dir = Path("/tmp/ogbench")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading OGB dataset {spec.source_dataset}", flush=True)
        train_dataset, _ = ogbench.make_env_and_datasets(
            spec.source_dataset,
            dataset_dir=str(dataset_dir),
            compact_dataset=False,
            dataset_only=True,
            add_info=True,
        )
        train_dataset = _truncate_to_whole_episodes(train_dataset, spec.render_max_rows)

        observations = train_dataset["observations"]
        if spec.render_from_state:
            pixels = None
        elif spec.synthetic_pixels:
            pixels = _synthetic_pixels_from_state(spec, train_dataset)
        else:
            pixels = _rgb_from_observations(observations)

        actions = (
            _one_hot_actions(train_dataset["actions"], spec.discrete_action_n)
            if spec.discrete_action_n is not None
            else _continuous_actions(train_dataset["actions"])
        )
        terminals = np.asarray(train_dataset["terminals"], dtype=bool)
        starts, lengths = _episode_lengths(terminals)
        total = int(lengths.sum())
        offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)

        pixel_rows = total if spec.render_from_state else pixels.shape[0]
        if total != pixel_rows or total != actions.shape[0]:
            raise ValueError(
                f"Row mismatch for {spec.source_dataset}: total={total}, "
                f"pixels={pixel_rows}, actions={actions.shape[0]}"
            )

        pixel_shape = _render_frame_shape(spec) if spec.render_from_state else pixels.shape[1:]
        pixel_chunks = (min(256, total), *pixel_shape)
        vector_chunks = (min(4096, total),)
        action_chunks = (min(4096, total), actions.shape[1])

        print(f"Writing {partial_path} rows={total}", flush=True)
        with h5py.File(partial_path, "w") as h5:
            h5.attrs["requested_dataset"] = spec.requested_name
            h5.attrs["source_dataset"] = spec.source_dataset
            h5.attrs["env_dataset"] = spec.env_dataset
            h5.attrs["pixel_source"] = (
                "rendered_qpos_qvel"
                if spec.render_from_state
                else spec.synthetic_pixels or "observations"
            )
            h5.attrs["discrete_action_n"] = (
                -1 if spec.discrete_action_n is None else spec.discrete_action_n
            )
            h5.create_dataset("ep_len", data=lengths.astype(np.int64))
            h5.create_dataset("ep_offset", data=offsets)
            pixel_ds = h5.create_dataset(
                "pixels",
                shape=(total, *pixel_shape),
                dtype=np.uint8,
                chunks=pixel_chunks,
            )
            if spec.render_from_state:
                _write_rendered_pixels_from_qpos_qvel(spec, train_dataset, pixel_ds)
            else:
                pixel_ds[...] = pixels
            h5.create_dataset(
                "action",
                data=actions.astype(np.float32, copy=False),
                dtype=np.float32,
                chunks=action_chunks,
            )
            h5.create_dataset("episode_idx", shape=(total,), dtype="int64", chunks=vector_chunks)
            h5.create_dataset("step_idx", shape=(total,), dtype="int64", chunks=vector_chunks)
            h5.create_dataset("terminal", data=terminals, dtype="bool", chunks=vector_chunks)

            for optional_key in ("qpos", "qvel", "button_states"):
                if optional_key in train_dataset:
                    values = np.asarray(train_dataset[optional_key])
                    h5.create_dataset(
                        optional_key,
                        data=values,
                        chunks=(min(4096, total), *values.shape[1:]),
                    )

            cursor = 0
            for ep_idx, length in enumerate(lengths):
                dst = slice(cursor, cursor + int(length))
                h5["episode_idx"][dst] = ep_idx
                h5["step_idx"][dst] = np.arange(length, dtype=np.int64)
                cursor += int(length)

        os.replace(partial_path, output_path)
        print(f"Prepared OGB HDF5: {output_path}", flush=True)
        return output_path
    finally:
        if partial_path.exists():
            partial_path.unlink()
        try:
            lock_dir.rmdir()
        except OSError:
            pass
