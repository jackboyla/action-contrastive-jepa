import json
import sys
from pathlib import Path

import lightning as pl
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    get_cuda_memory_metrics,
    JsonlMetricsCallback,
    ProgressBarCallback,
    ResumableCheckpointCallback,
    resolve_resume_checkpoint,
)


class TinyRegression(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = (self.layer(x) - y).pow(2).mean()
        self.log("fit/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = (self.layer(x) - y).pow(2).mean()
        self.log("validate/loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def make_loader():
    x = torch.arange(8, dtype=torch.float32).view(-1, 1)
    y = 2 * x
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=2, shuffle=False
    )


def test_resolve_resume_checkpoint_uses_state_file(tmp_path):
    ckpt = tmp_path / "custom.ckpt"
    ckpt.write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_state.json").write_text(
        json.dumps({"latest_checkpoint": "custom.ckpt"})
    )

    assert resolve_resume_checkpoint(tmp_path, "model").resolve() == ckpt.resolve()


def test_resolve_resume_checkpoint_refuses_nonempty_restart(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{}\n")

    with pytest.raises(RuntimeError, match="Refusing to start fresh"):
        resolve_resume_checkpoint(tmp_path, "model")


def test_cuda_memory_metrics_ignore_cpu_device():
    assert get_cuda_memory_metrics(torch.device("cpu")) == {}


def test_progress_bar_callback_emits_progress(tmp_path, capsys):
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=2,
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        callbacks=[ProgressBarCallback(refresh_every_n_steps=1, min_interval_s=0.0)],
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        TinyRegression(),
        train_dataloaders=make_loader(),
        val_dataloaders=make_loader(),
    )

    out = capsys.readouterr().out
    assert "[train] starting" in out
    assert "step" in out and "it/s" in out
    # ETA is rendered as elapsed<remaining
    assert "<" in out
    assert "[val]" in out
    assert "loss" in out


def test_resumable_checkpoint_smoke_resume(tmp_path):
    callbacks = [
        JsonlMetricsCallback(tmp_path, log_every_n_steps=1, log_to_trainer=False),
        ResumableCheckpointCallback(
            tmp_path,
            filename="tiny",
            step_interval=1,
            monitor="validate/loss",
            save_best=True,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(TinyRegression(), train_dataloaders=make_loader(), val_dataloaders=make_loader())

    latest = tmp_path / "tiny_weights.ckpt"
    assert latest.is_file()
    assert (tmp_path / "tiny_best_weights.ckpt").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()

    trainer = pl.Trainer(
        max_steps=3,
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        callbacks=[
            ResumableCheckpointCallback(
                tmp_path,
                filename="tiny",
                step_interval=1,
                monitor="validate/loss",
                save_best=True,
            )
        ],
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        TinyRegression(),
        train_dataloaders=make_loader(),
        val_dataloaders=make_loader(),
        ckpt_path=str(latest),
    )

    assert trainer.global_step == 3
