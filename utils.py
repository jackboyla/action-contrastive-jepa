import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from stable_pretraining import data as dt

from progress import format_duration, format_rate, render_bar


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class StopAfterEpochCallback(Callback):
    """Request a graceful stop after a completed 1-indexed epoch."""

    def __init__(self, stop_after_epoch: int):
        super().__init__()
        if stop_after_epoch < 1:
            raise ValueError("stop_after_epoch must be >= 1")
        self.stop_after_epoch = int(stop_after_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch >= self.stop_after_epoch:
            if trainer.is_global_zero:
                print(
                    f"Stopping after epoch {completed_epoch} "
                    f"(runtime.stop_after_epoch={self.stop_after_epoch}).",
                    flush=True,
                )
            trainer.should_stop = True


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True, default=_json_default))
        f.write("\n")
        f.flush()


def get_scalar_metrics(values: dict, prefix: str | None = None) -> dict[str, float]:
    metrics = {}
    for key, value in values.items():
        if prefix and not key.startswith(prefix):
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().float().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        elif not isinstance(value, (int, float)):
            continue
        metrics[str(key)] = float(value)
    return metrics


def infer_batch_size(batch) -> int | None:
    if torch.is_tensor(batch):
        return int(batch.shape[0]) if batch.ndim > 0 else None
    if isinstance(batch, dict):
        for value in batch.values():
            size = infer_batch_size(value)
            if size is not None:
                return size
    if isinstance(batch, (list, tuple)):
        for value in batch:
            size = infer_batch_size(value)
            if size is not None:
                return size
    return None


def get_cuda_memory_metrics(device=None) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            return {}
        device = device.index if device.index is not None else torch.cuda.current_device()

    total_bytes = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    gib = 1024**3
    return {
        "gpu/memory_allocated_gib": allocated / gib,
        "gpu/memory_reserved_gib": reserved / gib,
        "gpu/max_memory_allocated_gib": max_allocated / gib,
        "gpu/max_memory_reserved_gib": max_reserved / gib,
        "gpu/memory_total_gib": total_bytes / gib,
        "gpu/memory_allocated_pct": allocated / total_bytes,
        "gpu/memory_reserved_pct": reserved / total_bytes,
        "gpu/max_memory_allocated_pct": max_allocated / total_bytes,
        "gpu/max_memory_reserved_pct": max_reserved / total_bytes,
    }


def resolve_resume_checkpoint(run_dir, output_model_name, resume_cfg=None):
    """Resolve the checkpoint to resume from, or fail before overwriting progress."""

    run_dir = Path(run_dir)
    resume_cfg = resume_cfg or {}
    mode = str(resume_cfg.get("mode", "auto")).lower()
    explicit_path = resume_cfg.get("ckpt_path")
    prevent_restart = bool(resume_cfg.get("prevent_restart_existing", True))

    if mode not in {"auto", "must", "never"}:
        raise ValueError(f"resume.mode must be one of auto, must, never; got {mode!r}")

    def existing(path):
        path = Path(path)
        return path if path.is_file() else None

    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = run_dir / path
        path = existing(path)
        if path is None:
            raise FileNotFoundError(f"resume.ckpt_path does not exist: {explicit_path}")
        return path

    latest_from_state = None
    state_path = run_dir / "checkpoint_state.json"
    if state_path.is_file():
        try:
            with state_path.open() as f:
                state = json.load(f)
            latest = state.get("latest_checkpoint")
            if latest:
                latest_from_state = Path(latest)
                if not latest_from_state.is_absolute():
                    latest_from_state = run_dir / latest_from_state
        except (OSError, json.JSONDecodeError):
            latest_from_state = None

    candidates = [
        latest_from_state,
        run_dir / f"{output_model_name}_weights.ckpt",
        run_dir / "latest.ckpt",
        run_dir / "last.ckpt",
    ]
    resume_path = next((path for path in candidates if path and path.is_file()), None)

    if mode == "must" and resume_path is None:
        raise FileNotFoundError(f"resume.mode=must but no checkpoint exists in {run_dir}")

    if mode == "never":
        if prevent_restart and resume_path is not None:
            raise RuntimeError(
                f"Refusing to start fresh because a checkpoint exists at {resume_path}. "
                "Set resume.mode=auto to continue or resume.prevent_restart_existing=false."
            )
        return None

    if resume_path is not None:
        return resume_path

    if prevent_restart and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to start fresh in non-empty run directory without a checkpoint: {run_dir}. "
            "Use a new subdir, set resume.mode=never with "
            "resume.prevent_restart_existing=false, or provide resume.ckpt_path."
        )

    return None


class JsonlMetricsCallback(Callback):
    """Write local JSONL metrics/events and mirror timing metrics to the trainer logger."""

    def __init__(
        self,
        dirpath,
        log_every_n_steps: int = 50,
        log_to_trainer: bool = True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.log_every_n_steps = max(int(log_every_n_steps or 1), 1)
        self.log_to_trainer = log_to_trainer
        self.metrics_path = self.dirpath / "metrics.jsonl"
        self.events_path = self.dirpath / "events.jsonl"
        self._last_batch_end = None
        self._batch_start = None
        self._ema_step_time = None

    def _event(self, trainer, name: str, **payload) -> None:
        if not trainer.is_global_zero:
            return
        append_jsonl(
            self.events_path,
            {
                "time": time.time(),
                "event": name,
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                **payload,
            },
        )

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            self._event(
                trainer,
                "fit_start",
                max_epochs=trainer.max_epochs,
                max_steps=trainer.max_steps,
                estimated_stepping_batches=getattr(
                    trainer, "estimated_stepping_batches", None
                ),
            )
        self._last_batch_end = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._event(trainer, "train_epoch_start")

    def on_validation_epoch_start(self, trainer, pl_module):
        self._event(trainer, "validation_epoch_start")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = get_scalar_metrics(trainer.callback_metrics)
        self._event(trainer, "validation_epoch_end", metrics=metrics)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step <= 0 or step % self.log_every_n_steps != 0:
            self._last_batch_end = time.perf_counter()
            return

        now = time.perf_counter()
        batch_time = now - self._batch_start if self._batch_start else None
        data_time = (
            self._batch_start - self._last_batch_end
            if self._batch_start and self._last_batch_end
            else None
        )
        self._last_batch_end = now

        if batch_time is not None:
            step_time = batch_time
            if data_time is not None:
                step_time += max(data_time, 0.0)
            if self._ema_step_time is None:
                self._ema_step_time = step_time
            else:
                self._ema_step_time = 0.9 * self._ema_step_time + 0.1 * step_time
        else:
            step_time = None

        batch_size = infer_batch_size(batch) or 0
        throughput = (
            float(batch_size) / batch_time
            if batch_size and batch_time and batch_time > 0
            else None
        )
        wall_throughput = (
            float(batch_size) / step_time
            if batch_size and step_time and step_time > 0
            else None
        )

        total_steps = getattr(trainer, "estimated_stepping_batches", None)
        eta_seconds = None
        if total_steps and total_steps != float("inf") and self._ema_step_time:
            eta_seconds = max(int(total_steps) - step, 0) * self._ema_step_time

        lr_values = []
        optimizers = trainer.optimizers or []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                lr_values.append(float(group.get("lr", 0.0)))

        metrics = {
            "time": time.time(),
            "epoch": int(trainer.current_epoch),
            "global_step": step,
            "batch_idx": int(batch_idx),
            "batch_size": int(batch_size),
            "timing/batch_time_s": batch_time,
            "timing/data_time_s": data_time,
            "timing/step_time_s": step_time,
            "timing/gpu_samples_per_s": throughput,
            "timing/gpu_steps_per_s": (1.0 / batch_time) if batch_time else None,
            "timing/samples_per_s": wall_throughput,
            "timing/steps_per_s": (1.0 / step_time) if step_time else None,
            "timing/eta_s": eta_seconds,
        }
        metrics.update(get_cuda_memory_metrics(getattr(pl_module, "device", None)))
        if lr_values:
            metrics["optim/lr"] = lr_values[0]
            metrics["optim/lr_min"] = min(lr_values)
            metrics["optim/lr_max"] = max(lr_values)

        if hasattr(pl_module, "_last_grad_norm"):
            metrics["optim/grad_norm"] = float(pl_module._last_grad_norm)

        if isinstance(outputs, dict):
            for key, value in get_scalar_metrics(outputs).items():
                if (
                    "loss" in key
                    or key.endswith("_mse")
                    or key.endswith("_std")
                    or key.endswith("_weight")
                    or key.endswith("_weight_effective")
                ):
                    metrics[f"train_step/{key}"] = value

        metrics = {key: value for key, value in metrics.items() if value is not None}

        if trainer.is_global_zero:
            append_jsonl(self.metrics_path, metrics)

        if self.log_to_trainer and trainer.logger is not None:
            logger_metrics = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and key not in {"time", "epoch"}
            }
            trainer.logger.log_metrics(logger_metrics, step=step)

    def on_train_epoch_end(self, trainer, pl_module):
        self._event(trainer, "train_epoch_end")

    def on_exception(self, trainer, pl_module, exception):
        self._event(trainer, "exception", exception=repr(exception))

    def on_fit_end(self, trainer, pl_module):
        self._event(trainer, "fit_end")


class ProgressBarCallback(Callback):
    """Print verbose, log-friendly progress bars with ETA for fit/validation.

    Emits standalone lines (no carriage returns) so progress and ETA are visible
    in non-interactive logs (e.g. Modal) as well as in a terminal. Intended to
    replace Lightning's default TQDM bar, which renders poorly in captured logs.
    """

    def __init__(
        self,
        refresh_every_n_steps: int = 25,
        bar_width: int = 24,
        min_interval_s: float = 5.0,
    ):
        super().__init__()
        self.refresh_every_n_steps = max(int(refresh_every_n_steps or 1), 1)
        self.bar_width = int(bar_width)
        self.min_interval_s = float(min_interval_s)
        self._train_start = None
        self._train_start_step = 0
        self._last_train_emit = 0.0
        self._val_start = None
        self._val_count = 0
        self._last_val_emit = 0.0

    # ---------------------------------------------------------------- helpers
    def _total_steps(self, trainer):
        total = getattr(trainer, "estimated_stepping_batches", None)
        if total is None or total == float("inf"):
            return None
        try:
            return int(total)
        except (TypeError, ValueError, OverflowError):
            return None

    def _loss_value(self, trainer, outputs):
        if isinstance(outputs, dict):
            value = outputs.get("loss")
            if torch.is_tensor(value) and value.numel() == 1:
                return float(value.detach().cpu())
            if isinstance(value, (int, float)):
                return float(value)
        metrics = trainer.callback_metrics or {}
        for key in ("fit/loss", "train/loss", "validate/loss", "loss"):
            value = metrics.get(key)
            if torch.is_tensor(value) and value.numel() == 1:
                return float(value.detach().cpu())
            if isinstance(value, (int, float)):
                return float(value)
        return None

    # ------------------------------------------------------------------ train
    def on_train_start(self, trainer, pl_module):
        self._train_start = time.perf_counter()
        self._train_start_step = int(trainer.global_step)
        self._last_train_emit = 0.0
        if trainer.is_global_zero:
            total = self._total_steps(trainer)
            print(
                f"[train] starting at step {trainer.global_step} | "
                f"max_epochs={trainer.max_epochs} | "
                f"total_steps={total if total is not None else '?'}",
                flush=True,
            )

    def _render_train(self, trainer, outputs):
        step = int(trainer.global_step)
        total = self._total_steps(trainer)
        elapsed = time.perf_counter() - self._train_start if self._train_start else 0.0
        done = step - self._train_start_step
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        epoch = int(trainer.current_epoch) + 1
        max_epochs = trainer.max_epochs or "?"
        if total:
            fraction = step / total
            remaining = (total - step) / rate if rate > 0 else None
            head = (
                f"[train] e{epoch}/{max_epochs} {render_bar(fraction, self.bar_width)} "
                f"{fraction * 100:5.1f}% step {step:,}/{total:,}"
            )
            timing = f"{format_duration(elapsed)}<{format_duration(remaining)}"
        else:
            head = f"[train] e{epoch}/{max_epochs} step {step:,}"
            timing = format_duration(elapsed)
        line = f"{head} | {format_rate(rate)} | {timing}"
        loss = self._loss_value(trainer, outputs)
        if loss is not None:
            line += f" | loss {loss:.4f}"
        return line

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        epoch_batches = trainer.num_training_batches
        is_epoch_last = (
            epoch_batches not in (None, float("inf"))
            and (batch_idx + 1) >= int(epoch_batches)
        )
        due = step > 0 and step % self.refresh_every_n_steps == 0
        if not (due or is_epoch_last):
            return
        now = time.perf_counter()
        if not is_epoch_last and now - self._last_train_emit < self.min_interval_s:
            return
        self._last_train_emit = now
        print(self._render_train(trainer, outputs), flush=True)

    # ------------------------------------------------------------- validation
    def on_validation_start(self, trainer, pl_module):
        self._val_start = time.perf_counter()
        self._val_count = 0
        self._last_val_emit = 0.0

    def _val_total(self, trainer):
        total = trainer.num_val_batches
        if isinstance(total, (list, tuple)):
            finite = [t for t in total if isinstance(t, (int, float)) and t != float("inf")]
            total = sum(finite) if finite else None
        if total in (None, 0) or total == float("inf"):
            return None
        try:
            return int(total)
        except (TypeError, ValueError):
            return None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        self._val_count += 1
        total = self._val_total(trainer)
        is_last = bool(total) and self._val_count >= total
        due = self._val_count % self.refresh_every_n_steps == 0
        if not (is_last or due):
            return
        now = time.perf_counter()
        if not is_last and now - self._last_val_emit < self.min_interval_s:
            return
        self._last_val_emit = now
        elapsed = time.perf_counter() - self._val_start if self._val_start else 0.0
        rate = self._val_count / elapsed if elapsed > 0 and self._val_count else 0.0
        if total:
            fraction = self._val_count / total
            remaining = (total - self._val_count) / rate if rate > 0 else None
            line = (
                f"[val] {render_bar(fraction, self.bar_width)} {fraction * 100:5.1f}% "
                f"{self._val_count}/{total} | {format_rate(rate)} | "
                f"{format_duration(elapsed)}<{format_duration(remaining)}"
            )
        else:
            line = (
                f"[val] {self._val_count} batches | {format_rate(rate)} | "
                f"{format_duration(elapsed)}"
            )
        print(line, flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        metrics = get_scalar_metrics(trainer.callback_metrics)
        shown = {
            key: value
            for key, value in metrics.items()
            if "loss" in key or key.endswith("_mse")
        }
        if shown:
            summary = " | ".join(f"{key} {value:.4f}" for key, value in sorted(shown.items()))
            print(
                f"[val] epoch {int(trainer.current_epoch) + 1} done | {summary}",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(self._render_train(trainer, None) + " | epoch end", flush=True)


class ResumableCheckpointCallback(Callback):
    """Save atomic latest/best checkpoints during long training runs."""

    def __init__(
        self,
        dirpath,
        filename="model",
        step_interval: int = 0,
        time_interval_seconds: float = 0,
        monitor: str = "validate/loss",
        mode: str = "min",
        save_best: bool = True,
        keep_last_n: int = 0,
        save_on_train_epoch_end: bool = True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.step_interval = int(step_interval or 0)
        self.time_interval_seconds = float(time_interval_seconds or 0)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.keep_last_n = int(keep_last_n or 0)
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.latest_path = self.dirpath / f"{self.filename}_weights.ckpt"
        self.best_path = self.dirpath / f"{self.filename}_best_weights.ckpt"
        self.state_path = self.dirpath / "checkpoint_state.json"
        self.snapshot_dir = self.dirpath / "checkpoints"
        self.best_score = None
        self.best_step = None
        self._last_saved_step = -1
        self._last_save_time = time.time()
        if self.mode not in {"min", "max"}:
            raise ValueError(f"checkpoint.mode must be min or max, got {self.mode!r}")

    @property
    def state_key(self):
        return f"{self.__class__.__qualname__}.{self.filename}"

    def state_dict(self):
        return {
            "best_score": self.best_score,
            "best_step": self.best_step,
            "last_saved_step": self._last_saved_step,
            "last_save_time": self._last_save_time,
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict.get("best_score")
        self.best_step = state_dict.get("best_step")
        self._last_saved_step = state_dict.get("last_saved_step", -1)
        last_save_time = state_dict.get("last_save_time")
        self._last_save_time = float(last_save_time) if last_save_time else time.time()

    def _is_better(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return value < float(self.best_score)
        return value > float(self.best_score)

    def _save_checkpoint(self, trainer, path: Path, reason: str) -> None:
        if not trainer.is_global_zero:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        trainer.save_checkpoint(str(tmp))
        tmp.replace(path)
        self._last_saved_step = int(trainer.global_step)
        self._last_save_time = time.time()
        self._write_state(trainer, reason=reason)
        print(f"Saved checkpoint ({reason}): {path}", flush=True)

    def _write_state(self, trainer, reason: str) -> None:
        write_json_atomic(
            self.state_path,
            {
                "time": time.time(),
                "reason": reason,
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "latest_checkpoint": self.latest_path.name,
                "best_checkpoint": self.best_path.name if self.best_path.exists() else None,
                "best_score": self.best_score,
                "best_step": self.best_step,
                "monitor": self.monitor,
                "mode": self.mode,
            },
        )

    def _maybe_save_snapshot(self, trainer) -> None:
        if self.keep_last_n <= 0 or not self.latest_path.is_file():
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot_dir / f"step_{int(trainer.global_step)}.ckpt"
        try:
            if snapshot.exists():
                snapshot.unlink()
            os.link(self.latest_path, snapshot)
        except OSError:
            shutil.copy2(self.latest_path, snapshot)
        snapshots = sorted(
            self.snapshot_dir.glob("step_*.ckpt"), key=lambda path: path.stat().st_mtime
        )
        for old in snapshots[:-self.keep_last_n]:
            old.unlink(missing_ok=True)

    def _should_save_on_step(self, trainer) -> bool:
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_saved_step:
            return False
        if self.step_interval > 0 and step % self.step_interval == 0:
            return True
        if (
            self.time_interval_seconds > 0
            and time.time() - self._last_save_time >= self.time_interval_seconds
        ):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if not trainer.is_global_zero or not self._should_save_on_step(trainer):
            return

        self._save_checkpoint(trainer, self.latest_path, reason="train_step")
        self._maybe_save_snapshot(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.save_on_train_epoch_end and trainer.is_global_zero:
            self._save_checkpoint(trainer, self.latest_path, reason="train_epoch_end")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.save_best or not trainer.is_global_zero:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        if torch.is_tensor(metric):
            metric = metric.detach().float().cpu().item()
        metric = float(metric)
        if self._is_better(metric):
            self.best_score = metric
            self.best_step = int(trainer.global_step)
            self._save_checkpoint(trainer, self.best_path, reason=f"best_{self.monitor}")
            self._write_state(trainer, reason=f"best_{self.monitor}")

    def on_exception(self, trainer, pl_module, exception):
        if trainer.is_global_zero:
            try:
                if int(trainer.global_step) <= 0 and self.latest_path.exists():
                    print(
                        "Skipping exception checkpoint at global_step=0 to avoid "
                        f"overwriting existing checkpoint: {self.latest_path}",
                        flush=True,
                    )
                    return
                self._save_checkpoint(trainer, self.latest_path, reason="exception")
            except Exception as save_error:
                print(f"Error saving exception checkpoint: {save_error}", flush=True)

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._save_checkpoint(trainer, self.latest_path, reason="fit_end")


class RNGStateCallback(Callback):
    """Persist practical RNG state in Lightning checkpoints."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get("rng_state")
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])


class MinEpochEarlyStopping(EarlyStopping):
    """Early stopping with a minimum epoch guard."""

    def __init__(self, *args, min_epochs: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = int(min_epochs or 0)

    def _run_early_stopping_check(self, trainer):
        if int(trainer.current_epoch) + 1 < self.min_epochs:
            return
        return super()._run_early_stopping_check(trainer)
