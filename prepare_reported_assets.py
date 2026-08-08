"""Prepare released LeWM datasets and checkpoints in STABLEWM_HOME."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from project_paths import configure_stablewm_home

configure_stablewm_home()

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from huggingface_hub import hf_hub_download, hf_hub_url

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


@dataclass(frozen=True)
class ReportedAsset:
    task: str
    dataset_repo: str
    dataset_file: str
    dataset_target: str
    model_repo: str
    checkpoint_target: str


ASSETS: dict[str, ReportedAsset] = {
    "tworoom": ReportedAsset(
        task="tworoom",
        dataset_repo="quentinll/lewm-tworooms",
        dataset_file="tworoom.tar.zst",
        dataset_target="tworoom.h5",
        model_repo="quentinll/lewm-tworooms",
        checkpoint_target="tworoom/lewm_object.ckpt",
    ),
    "pusht": ReportedAsset(
        task="pusht",
        dataset_repo="quentinll/lewm-pusht",
        dataset_file="pusht_expert_train.h5.zst",
        dataset_target="pusht_expert_train.h5",
        model_repo="quentinll/lewm-pusht",
        checkpoint_target="pusht/lewm_object.ckpt",
    ),
    "reacher": ReportedAsset(
        task="reacher",
        dataset_repo="quentinll/lewm-reacher",
        dataset_file="reacher.tar.zst",
        dataset_target="dmc/reacher_random.h5",
        model_repo="quentinll/lewm-reacher",
        checkpoint_target="reacher/lewm_object.ckpt",
    ),
    "cube": ReportedAsset(
        task="cube",
        dataset_repo="quentinll/lewm-cube",
        dataset_file="cube_single_expert.tar.zst",
        dataset_target="ogbench/cube_single_expert.h5",
        model_repo="quentinll/lewm-cube",
        checkpoint_target="cube/lewm_object.ckpt",
    ),
}


def parse_tasks(value: str | None) -> list[str]:
    if not value:
        return list(ASSETS)
    tasks = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    unknown = sorted(set(tasks) - set(ASSETS))
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; expected one of {sorted(ASSETS)}")
    return tasks


def run_pipeline(left: list[str], right: list[str], *, stdout_path: Path | None = None) -> None:
    print("Running:", " ".join(left), "|", " ".join(right), flush=True)
    stdout = None if stdout_path is None else stdout_path.open("wb")
    try:
        p1 = subprocess.Popen(left, stdout=subprocess.PIPE)
        assert p1.stdout is not None
        p2 = subprocess.Popen(right, stdin=p1.stdout, stdout=stdout)
        p1.stdout.close()
        rc2 = p2.wait()
        rc1 = p1.wait()
    finally:
        if stdout is not None:
            stdout.close()

    if rc1 != 0:
        raise subprocess.CalledProcessError(rc1, left)
    if rc2 != 0:
        raise subprocess.CalledProcessError(rc2, right)


def curl_cmd(url: str) -> list[str]:
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-delay",
        "5",
    ]
    token = os.environ.get("HF_TOKEN")
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    cmd.append(url)
    return cmd


def find_extracted_h5(root: Path) -> Path:
    candidates = [path for path in root.rglob("*.h5") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no .h5 file found under {root}")
    return max(candidates, key=lambda path: path.stat().st_size)


def prepare_dataset(asset: ReportedAsset, cache_dir: Path, *, force: bool) -> None:
    target = cache_dir / asset.dataset_target
    if target.exists() and not force:
        print(f"Dataset exists, skipping: {target}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(
        asset.dataset_repo,
        asset.dataset_file,
        repo_type="dataset",
    )
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    if asset.dataset_file.endswith(".h5.zst"):
        run_pipeline(curl_cmd(url), ["zstd", "-T0", "-dc"], stdout_path=partial)
        partial.replace(target)
        print(f"Prepared dataset: {target}", flush=True)
        return

    extract_dir = cache_dir / "_extract" / asset.task
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    try:
        run_pipeline(
            curl_cmd(url),
            ["tar", "-I", "zstd", "-xf", "-", "-C", str(extract_dir)],
        )
        extracted_h5 = find_extracted_h5(extract_dir)
        if target.exists():
            target.unlink()
        shutil.move(str(extracted_h5), str(target))
        print(f"Prepared dataset: {target}", flush=True)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def clean_cfg_section(section: dict) -> dict:
    return {key: value for key, value in section.items() if key != "_target_"}


def remap_encoder_key(key: str) -> str:
    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.(.*)", key)
    if not match:
        return key
    idx, rest = match.groups()
    rest = rest.replace("attention.attention.query", "attention.q_proj")
    rest = rest.replace("attention.attention.key", "attention.k_proj")
    rest = rest.replace("attention.attention.value", "attention.v_proj")
    rest = rest.replace("attention.output.dense", "attention.o_proj")
    rest = rest.replace("intermediate.dense", "mlp.fc1")
    rest = rest.replace("output.dense", "mlp.fc2")
    return f"encoder.layers.{idx}.{rest}"


def build_model(cfg: dict) -> JEPA:
    encoder_cfg = clean_cfg_section(cfg["encoder"])
    encoder = spt.backbone.utils.vit_hf(**encoder_cfg)

    def mlp(name: str) -> MLP:
        section = clean_cfg_section(cfg[name])
        section["norm_fn"] = torch.nn.BatchNorm1d
        return MLP(**section)

    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**clean_cfg_section(cfg["predictor"])),
        action_encoder=Embedder(**clean_cfg_section(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )


def prepare_checkpoint(asset: ReportedAsset, cache_dir: Path, *, force: bool) -> None:
    target = cache_dir / asset.checkpoint_target
    if target.exists() and not force:
        print(f"Checkpoint exists, skipping: {target}", flush=True)
        return

    local_dir = cache_dir / "_hf_models" / asset.task
    local_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(
        hf_hub_download(
            asset.model_repo,
            "config.json",
            repo_type="model",
            local_dir=local_dir,
        )
    )
    weights_path = Path(
        hf_hub_download(
            asset.model_repo,
            "weights.pt",
            repo_type="model",
            local_dir=local_dir,
        )
    )

    cfg = json.loads(config_path.read_text())
    model = build_model(cfg)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = {remap_encoder_key(key): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, target)
    print(f"Prepared checkpoint: {target}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(ASSETS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    cache_dir = Path(swm.data.utils.get_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing reported assets under {cache_dir}", flush=True)

    for task in parse_tasks(args.tasks):
        asset = ASSETS[task]
        print(f"=== Preparing {task} ===", flush=True)
        if args.datasets:
            prepare_dataset(asset, cache_dir, force=bool(args.force))
        if args.checkpoints:
            prepare_checkpoint(asset, cache_dir, force=bool(args.force))


if __name__ == "__main__":
    main(sys.argv[1:])
