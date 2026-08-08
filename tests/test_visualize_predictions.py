import numpy as np
import torch
from PIL import Image

from visualize_predictions import clear_rollout_artifacts, save_rollout_gif


def test_save_rollout_gif_writes_infinite_loop_extension(tmp_path):
    path = tmp_path / "rollout.gif"
    target = torch.zeros(2, 3, 8, 8)
    pred_frames = torch.ones(2, 3, 8, 8)
    target[1] = 0.5
    pred_frames[1] = 0.25

    save_rollout_gif(
        target=target,
        pred_frames=pred_frames,
        mse=np.array([0.1, 0.2]),
        path=path,
        cell_size=16,
        fps=3,
    )

    image = Image.open(path)
    assert image.n_frames == 2
    assert image.info["loop"] == 0
    assert 300 <= image.info["duration"] <= 360


def test_clear_rollout_artifacts_removes_generated_images_only(tmp_path):
    generated_png = tmp_path / "rollout_00.png"
    generated_gif = tmp_path / "rollout_01.gif"
    keep = tmp_path / "horizon_errors.png"
    generated_png.write_text("old")
    generated_gif.write_text("old")
    keep.write_text("keep")

    clear_rollout_artifacts(tmp_path)

    assert not generated_png.exists()
    assert not generated_gif.exists()
    assert keep.exists()
