"""Train an auxiliary frozen-latent decoder for LeWM visualizations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

from project_paths import configure_stablewm_home

configure_stablewm_home()

import hydra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import v2 as transforms

from decoder import build_decoder, build_decoder_from_checkpoint
from progress import ProgressPrinter


ROOT = Path(__file__).resolve().parent


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_eval_cfg(task: str) -> DictConfig:
    path = ROOT / "config" / "eval" / f"{task}.yaml"
    if not path.exists():
        choices = sorted(p.stem for p in (ROOT / "config" / "eval").glob("*.yaml"))
        raise FileNotFoundError(f"unknown task '{task}'. Available tasks: {choices}")
    return OmegaConf.load(path)


def make_input_transform(img_size: int):
    stats = spt.data.dataset_stats.ImageNet
    return transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**stats),
            transforms.Resize(size=(img_size, img_size), antialias=True),
        ]
    )


def make_target_transform(img_size: int, target_space: str):
    if target_space == "rgb":
        return transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(size=(img_size, img_size), antialias=True),
            ]
        )
    if target_space == "imagenet_normalized":
        return make_input_transform(img_size)
    raise ValueError("loss.target_space must be 'rgb' or 'imagenet_normalized'")


def prepare_pixels(
    pixels: torch.Tensor,
    *,
    img_size: int,
    target_space: str,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pixels.ndim == 5:
        pixels = pixels[:, 0]
    inputs = make_input_transform(img_size)(pixels).to(device)
    targets = make_target_transform(img_size, target_space)(pixels).to(device)
    return inputs, targets


def encode_pixels(
    world_model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return world_model.encode({"pixels": inputs[:, None]})["emb"][:, 0].detach()


def resolve_amp(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type != "cuda":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision in {"16", "fp16", "float16"}:
        return torch.float16, True
    return None, False


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def to_rgb(x: torch.Tensor, target_space: str) -> torch.Tensor:
    if target_space == "imagenet_normalized":
        return denormalize_imagenet(x.float())
    if target_space == "rgb":
        return x.float().clamp(0.0, 1.0)
    raise ValueError("target_space must be 'rgb' or 'imagenet_normalized'")


def loss_fn(
    pred: torch.Tensor, target: torch.Tensor, cfg: DictConfig
) -> dict[str, torch.Tensor]:
    per_pixel_mse = (pred - target).pow(2).flatten(start_dim=1)
    topk_fraction = float(cfg.loss.topk_fraction)
    topk_weight = float(cfg.loss.topk_mse_weight)
    if topk_weight > 0 and topk_fraction > 0:
        k = max(1, int(per_pixel_mse.size(1) * topk_fraction))
        topk_mse = per_pixel_mse.topk(k=k, dim=1).values.mean()
    else:
        topk_mse = per_pixel_mse.new_zeros(())
    mse = per_pixel_mse.mean()
    l1 = F.l1_loss(pred, target)
    loss = (
        float(cfg.loss.l1_weight) * l1
        + float(cfg.loss.mse_weight) * mse
        + topk_weight * topk_mse
    )
    return {
        "loss": loss,
        "l1": l1,
        "mse": mse,
        "topk_mse": topk_mse,
    }


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, steps: int):
    if not cfg.scheduler.enabled:
        return None
    if cfg.scheduler.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps)
        )
    raise ValueError(f"unknown scheduler type: {cfg.scheduler.type}")


def run_epoch(
    *,
    world_model: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: DictConfig,
    device: torch.device | str,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    split: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    training = optimizer is not None
    decoder.train(training)
    totals = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "topk_mse": 0.0}
    total_items = 0
    rows = []

    limit_key = "limit_train_batches" if training else "limit_val_batches"
    limit_batches = cfg.trainer.get(limit_key)

    total_batches = len(loader)
    if limit_batches is not None:
        total_batches = min(total_batches, int(limit_batches))
    progress = None
    if bool(cfg.trainer.get("progress_bar", True)):
        progress = ProgressPrinter(
            total_batches,
            label=f"decoder {split} e{epoch}",
            min_interval_s=float(cfg.trainer.get("progress_refresh_s", 2.0)),
        )

    for step, batch in enumerate(loader):
        if limit_batches is not None and step >= int(limit_batches):
            break

        inputs, targets = prepare_pixels(
            batch["pixels"],
            img_size=int(cfg.img_size),
            target_space=str(cfg.loss.target_space),
            device=device,
        )
        z = encode_pixels(world_model, inputs)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            pred = decoder(z)
            losses = loss_fn(pred, targets, cfg)
            loss = losses["loss"]

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.trainer.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(), float(cfg.trainer.gradient_clip_val)
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.trainer.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(), float(cfg.trainer.gradient_clip_val)
                    )
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = targets.size(0)
        total_items += batch_size
        detached = {key: value.detach().float().item() for key, value in losses.items()}
        for key, value in detached.items():
            totals[key] += value * batch_size
        row = {
            "split": split,
            "epoch": epoch,
            "step": step + 1,
            "items": total_items,
            "lr": (
                None
                if optimizer is None
                else float(optimizer.param_groups[0]["lr"])
            ),
            **detached,
        }
        rows.append(row)

        if progress is not None:
            progress.update(1, suffix=f"loss {totals['loss'] / total_items:.5f}")

    if progress is not None:
        progress.close(suffix=f"loss {totals['loss'] / max(total_items, 1):.5f}")

    denom = max(total_items, 1)
    return {f"{split}_{key}": value / denom for key, value in totals.items()}, rows


def image_from_tensor(
    tensor: torch.Tensor, *, target_space: str, size: int = 128
) -> Image.Image:
    rgb = to_rgb(tensor[None], target_space)[0]
    array = rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255).astype(np.uint8))
    return image.resize((size, size), Image.Resampling.BILINEAR)


@torch.inference_mode()
def save_reconstruction_preview(
    *,
    world_model: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    cfg: DictConfig,
    device: torch.device | str,
    path: Path,
    max_items: int = 6,
) -> None:
    batch = next(iter(loader))
    inputs, targets = prepare_pixels(
        batch["pixels"][:max_items],
        img_size=int(cfg.img_size),
        target_space=str(cfg.loss.target_space),
        device=device,
    )
    recon = decoder(encode_pixels(world_model, inputs))
    n_items = targets.size(0)

    cell = 128
    label_w = 90
    header_h = 28
    canvas = Image.new(
        "RGB", (label_w + n_items * cell, header_h + 2 * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for i in range(n_items):
        draw.text((label_w + i * cell + 6, 8), f"sample {i}", fill=(30, 30, 30))
        canvas.paste(
            image_from_tensor(targets[i], target_space=str(cfg.loss.target_space), size=cell),
            (label_w + i * cell, header_h),
        )
        canvas.paste(
            image_from_tensor(recon[i], target_space=str(cfg.loss.target_space), size=cell),
            (label_w + i * cell, header_h + cell),
        )
    draw.text((8, header_h + cell // 2), "target", fill=(30, 30, 30))
    draw.text((8, header_h + cell + cell // 2), "decoder", fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_checkpoint(
    *,
    decoder: torch.nn.Module,
    cfg: DictConfig,
    output_path: Path,
    latent_dim: int,
    dataset_name: str,
    metrics: dict[str, float],
    epoch: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "decoder": decoder.state_dict(),
        "latent_dim": latent_dim,
        "img_size": int(cfg.img_size),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "target_space": str(cfg.loss.target_space),
        "policy": str(cfg.policy),
        "dataset_name": dataset_name,
        "metrics": metrics,
        "epoch": epoch,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(checkpoint, output_path)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join("" if row[k] is None else str(row[k]) for k in fields) + "\n")


def save_training_curves(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, key in zip(axes.ravel(), ["loss", "topk_mse", "mse", "l1"]):
        for split in ["train", "val"]:
            split_rows = [row for row in rows if row["split"] == split]
            if split_rows:
                ax.plot(
                    range(1, len(split_rows) + 1),
                    [row[key] for row in split_rows],
                    label=split,
                )
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="./config/decoder", config_name="train")
def run(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    eval_cfg = load_eval_cfg(str(cfg.task))
    dataset_name = str(cfg.dataset_name or eval_cfg.eval.dataset_name)
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    output_dir = (
        Path(str(cfg.output_dir)).expanduser().resolve()
        if cfg.output_dir is not None
        else Path(
            cache_dir,
            "decoders",
            str(cfg.task),
            str(cfg.policy).replace("/", "__"),
        )
    )
    output_path = output_dir / f"{cfg.output_name}.pt"
    device = torch.device(resolve_device(str(cfg.device)))

    print(f"Loading frozen world model '{cfg.policy}' on {device}", flush=True)
    world_model = swm.policy.AutoCostModel(str(cfg.policy), cache_dir=cache_dir)
    world_model = world_model.to(device).eval()
    world_model.requires_grad_(False)
    if hasattr(world_model, "interpolate_pos_encoding"):
        world_model.interpolate_pos_encoding = True

    print(f"Loading dataset '{dataset_name}' from {cache_dir}", flush=True)
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels"],
        cache_dir=cache_dir,
    )

    if cfg.max_samples is not None and int(cfg.max_samples) < len(dataset):
        rng = np.random.default_rng(int(cfg.seed))
        indices = np.sort(
            rng.choice(len(dataset), size=int(cfg.max_samples), replace=False)
        )
        dataset = Subset(dataset, indices.tolist())

    train_len = int(len(dataset) * float(cfg.train_split))
    val_len = len(dataset) - train_len
    if train_len <= 0 or val_len <= 0:
        raise ValueError(
            "decoder dataset split is empty; increase max_samples or adjust train_split"
        )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    if int(loader_kwargs.get("num_workers", 0)) == 0:
        loader_kwargs["persistent_workers"] = False
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    sample = next(iter(train_loader))
    inputs, _ = prepare_pixels(
        sample["pixels"][:1],
        img_size=int(cfg.img_size),
        target_space=str(cfg.loss.target_space),
        device=device,
    )
    latent_dim = int(encode_pixels(world_model, inputs).size(-1))
    decoder = build_decoder(
        architecture=str(cfg.model.architecture),
        latent_dim=latent_dim,
        img_size=int(cfg.img_size),
        patch_size=int(cfg.model.patch_size),
        dim=int(cfg.model.dim),
        heads=int(cfg.model.heads),
        depth=int(cfg.model.depth),
    ).to(device)
    start_epoch = 1
    if cfg.trainer.resume_checkpoint is not None:
        resume_path = Path(str(cfg.trainer.resume_checkpoint)).expanduser()
        resumed_decoder, checkpoint = build_decoder_from_checkpoint(resume_path)
        decoder.load_state_dict(resumed_decoder.state_dict())
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Loaded decoder weights from {resume_path}", flush=True)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    train_batches = len(train_loader)
    if cfg.trainer.limit_train_batches is not None:
        train_batches = min(train_batches, int(cfg.trainer.limit_train_batches))
    total_train_steps = max(1, int(cfg.trainer.max_epochs) * train_batches)
    scheduler = build_scheduler(optimizer, cfg, total_train_steps)
    amp_dtype, use_grad_scaler = resolve_amp(device, str(cfg.trainer.precision))
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    best_val = float("inf")
    best_metrics = {}
    metric_rows = []
    for epoch in range(start_epoch, int(cfg.trainer.max_epochs) + 1):
        train_metrics, train_rows = run_epoch(
            world_model=world_model,
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            split="train",
        )
        val_metrics, val_rows = run_epoch(
            world_model=world_model,
            decoder=decoder,
            loader=val_loader,
            optimizer=None,
            cfg=cfg,
            device=device,
            amp_dtype=amp_dtype,
            scaler=None,
            scheduler=None,
            epoch=epoch,
            split="val",
        )
        metrics = {**train_metrics, **val_metrics}
        print(f"epoch={epoch} metrics={json.dumps(metrics)}", flush=True)
        epoch_rows = train_rows + val_rows
        metric_rows.extend(epoch_rows)
        append_jsonl(output_dir / "metrics.jsonl", epoch_rows)
        write_tsv(output_dir / "metrics.tsv", metric_rows)
        save_training_curves(metric_rows, output_dir / "training_curves.png")

        save_checkpoint(
            decoder=decoder,
            cfg=cfg,
            output_path=output_dir / f"{cfg.output_name}_last.pt",
            latent_dim=latent_dim,
            dataset_name=dataset_name,
            metrics=metrics,
            epoch=epoch,
        )
        if epoch % max(1, int(cfg.trainer.save_every_epochs)) == 0:
            save_checkpoint(
                decoder=decoder,
                cfg=cfg,
                output_path=output_dir / f"{cfg.output_name}_epoch_{epoch}.pt",
                latent_dim=latent_dim,
                dataset_name=dataset_name,
                metrics=metrics,
                epoch=epoch,
            )
            save_reconstruction_preview(
                world_model=world_model,
                decoder=decoder,
                loader=val_loader,
                cfg=cfg,
                device=device,
                path=output_dir / f"reconstruction_epoch_{epoch}.png",
            )
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            best_metrics = metrics
            save_checkpoint(
                decoder=decoder,
                cfg=cfg,
                output_path=output_path,
                latent_dim=latent_dim,
                dataset_name=dataset_name,
                metrics=metrics,
                epoch=epoch,
            )

    save_reconstruction_preview(
        world_model=world_model,
        decoder=decoder,
        loader=val_loader,
        cfg=cfg,
        device=device,
        path=output_dir / "reconstruction_preview.png",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved best decoder to {output_path}", flush=True)


if __name__ == "__main__":
    run()
