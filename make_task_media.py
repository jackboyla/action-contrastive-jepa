"""Extract still + gif media for each task dataset (for README).

Reads pixel observations from each dataset on the volume (and downloads the
OGBench visual-scene dataset), extracts a representative episode, and writes a
still PNG + a looping GIF to the volume under media/. Then download locally.

Run: nohup .venv/bin/modal run make_task_media.py::main > /tmp/media.log 2>&1 &
"""

import modal

from modal_app import STABLEWM_HOME, image, volume

app = modal.App("task-media", image=image)


def _save_media(pixels, out_stem, n_gif=40, size=160):
    """pixels: (T,H,W,C) uint8 -> writes <out_stem>_still.png and <out_stem>.gif"""
    import numpy as np
    from PIL import Image
    from pathlib import Path

    pixels = np.asarray(pixels)
    if pixels.ndim == 4 and pixels.shape[1] in (1, 3) and pixels.shape[-1] not in (1, 3):
        pixels = np.transpose(pixels, (0, 2, 3, 1))  # CHW -> HWC
    if pixels.dtype != np.uint8:
        p = pixels.astype("float32")
        p = (p - p.min()) / (p.ptp() + 1e-8) * 255.0
        pixels = p.astype("uint8")
    if pixels.shape[-1] > 3:
        pixels = pixels[..., :3]
    if pixels.shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=-1)

    T = pixels.shape[0]
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)

    # still: a mid-episode frame (more informative than frame 0)
    Image.fromarray(pixels[min(T // 3, T - 1)]).resize((size, size)).save(f"{out_stem}_still.png")

    # gif: evenly subsample to n_gif frames
    idx = np.linspace(0, T - 1, min(n_gif, T)).astype(int)
    frames = [Image.fromarray(pixels[i]).resize((size, size)) for i in idx]
    frames[0].save(
        f"{out_stem}.gif", save_all=True, append_images=frames[1:],
        duration=80, loop=0, optimize=True,
    )
    print(f"  wrote {out_stem}_still.png and {out_stem}.gif  (T={T})", flush=True)


def _ogb_frame(env, obs):
    """Return a displayable RGB frame from either pixel or state OGB observations."""
    import numpy as np

    obs = np.asarray(obs)
    if obs.ndim == 3 and obs.shape[-1] <= 3:
        return obs[..., :3]
    return env.render()


def _save_ogb_env_media(dataset_name, out_stem, seed=0, n_frames=80):
    import ogbench
    import numpy as np

    env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        frames.append(_ogb_frame(env, obs))
        rng = np.random.default_rng(seed)
        for _ in range(n_frames - 1):
            if hasattr(env.action_space, "n"):
                action = int(rng.integers(env.action_space.n))
            else:
                low = np.asarray(env.action_space.low, dtype=np.float32)
                high = np.asarray(env.action_space.high, dtype=np.float32)
                action = rng.uniform(low, high).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(_ogb_frame(env, obs))
            if terminated or truncated:
                break
        _save_media(np.asarray(frames), out_stem)
    finally:
        env.close()


def _save_scene_highres_still(dataset, frame_idx, out_path, size=1024):
    """Render a Scene dataset state at print/README resolution."""
    import gymnasium as gym
    import numpy as np
    import ogbench.manipspace  # noqa: F401  (registers scene-v0)
    from PIL import Image
    from pathlib import Path

    qpos = np.asarray(dataset["qpos"][frame_idx], dtype=np.float64)
    qvel = np.asarray(dataset["qvel"][frame_idx], dtype=np.float64)
    button_states = np.asarray(dataset["button_states"][frame_idx], dtype=np.int64)

    env = gym.make(
        "scene-v0",
        ob_type="pixels",
        width=size,
        height=size,
        visualize_info=False,
        terminate_at_goal=True,
    )
    try:
        env.reset(seed=0)
        env.unwrapped.set_state(qpos, qvel, button_states)
        frame = env.unwrapped.render(camera="front_pixels")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(out_path)
        print(f"  wrote high-res Scene still {out_path} ({size}x{size})", flush=True)
    finally:
        env.close()


def _first_episode_pixels(f):
    """Return pixels for the first episode given various boundary schemas."""
    import numpy as np
    keys = list(f.keys())
    pix_key = next((k for k in ("pixels", "observations", "image", "rgb") if k in keys), None)
    if pix_key is None:
        raise KeyError(f"no pixel key in {keys}")
    pix = f[pix_key]
    # determine episode 0 slice
    if "ep_offset" in keys and "ep_len" in keys:
        s = int(f["ep_offset"][0]); e = s + int(f["ep_len"][0])
    elif "ep_idx" in keys:
        ep = f["ep_idx"][:]; e = int(np.searchsorted(ep, ep[0] + 1)); s = 0
    elif "episode_idx" in keys:
        ep = f["episode_idx"][:]; e = int(np.searchsorted(ep, ep[0] + 1)); s = 0
    elif "terminals" in keys:
        term = f["terminals"][:]; e = int(np.argmax(term > 0)) + 1 if (term > 0).any() else min(200, pix.shape[0]); s = 0
    else:
        s, e = 0, min(200, pix.shape[0])
    e = min(e, s + 250)  # cap
    return pix[s:e]


@app.function(volumes={str(STABLEWM_HOME): volume}, gpu="L4", timeout=3600)
def build():
    import hdf5plugin  # noqa: F401  (registers the HDF5 compression filters)
    import h5py
    from pathlib import Path

    base = Path(STABLEWM_HOME)
    media = base / "media"
    media.mkdir(parents=True, exist_ok=True)

    tasks = {
        "pusht": base / "pusht_expert_train.h5",
        "tworoom": base / "tworoom.h5",
        "reacher": base / "dmc" / "reacher_random.h5",
        "cube": base / "ogbench" / "cube_single_expert.h5",
    }
    for name, path in tasks.items():
        print(f"=== {name}: {path} ===", flush=True)
        if not path.exists():
            print(f"  MISSING {path}", flush=True); continue
        try:
            with h5py.File(path, "r") as f:
                print(f"  keys: {list(f.keys())[:20]}", flush=True)
                pix = _first_episode_pixels(f)
                _save_media(pix, str(media / name))
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}", flush=True)

    # Scene: pre-rendered visual dataset
    print("=== scene: visual-scene-play-v0 ===", flush=True)
    try:
        import ogbench, numpy as np
        _, tr, _ = ogbench.make_env_and_datasets(
            "visual-scene-play-v0", compact_dataset=False, add_info=True
        )
        obs = np.asarray(tr["observations"])
        term = np.asarray(tr["terminals"])
        e = int(np.argmax(term > 0)) + 1 if (term > 0).any() else 200
        n = min(e, 250)
        _save_media(obs[:n], str(media / "scene"))
        _save_scene_highres_still(tr, min(n // 3, n - 1), media / "scene_still.png")
    except Exception as e:
        print(f"  FAILED scene: {type(e).__name__}: {e}", flush=True)

    ogb_play_tasks = {
        "puzzle4x4": "visual-puzzle-4x4-play-v0",
        "puzzle4x5": "visual-puzzle-4x5-play-v0",
        "antmaze_teleport": "visual-antmaze-teleport-navigate-v0",
        "powderworld": "powderworld-medium-play-v0",
        "antmaze_large": "visual-antmaze-large-stitch-v0",
        "antsoccer": "antsoccer-medium-stitch-v0",
    }
    for name, dataset_name in ogb_play_tasks.items():
        print(f"=== {name}: {dataset_name} ===", flush=True)
        try:
            _save_ogb_env_media(dataset_name, str(media / name))
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}", flush=True)

    volume.commit()
    print("=== done; media files: ===", flush=True)
    for p in sorted(media.glob("*")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)", flush=True)


@app.local_entrypoint()
def main():
    build.remote()
