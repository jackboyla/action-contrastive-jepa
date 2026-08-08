"""Modal entrypoints for remote LeWM training and evaluation."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import modal

from ogb_prep import get_ogb_task_spec, prepare_ogb_hdf5


APP_NAME = "multi-future-lewm"
VOLUME_NAME = "multi-future-lewm-cache"
REPO_DIR = Path("/workspace/long-horizon-world-model")
STABLEWM_HOME = REPO_DIR / ".stable_worldmodel"
LOCAL_STABLEWM_HOME = Path("/tmp/stable_worldmodel")

TRAIN_GPU = os.environ.get("MODAL_TRAIN_GPU", "A100-40GB")
DECODER_GPU = os.environ.get("MODAL_DECODER_GPU", "L4")
EVAL_GPU = os.environ.get("MODAL_EVAL_GPU", "L4")
VISUALIZE_GPU = os.environ.get("MODAL_VISUALIZE_GPU", "L4")
CPU_COUNT = float(os.environ.get("MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "32768"))
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60)))
# Auto-retry on failures, including spot-worker preemption. Training/eval resume
# from the latest checkpoint (resume.mode: auto), so a retried attempt continues
# where the preempted run left off instead of restarting from scratch.
RETRIES = int(os.environ.get("MODAL_RETRIES", "3"))
OUTPUT_COMMIT_INTERVAL_SECONDS = int(
    os.environ.get("MODAL_OUTPUT_COMMIT_INTERVAL_SECONDS", "300")
)
REPORTED_LEWM_TASKS = ("tworoom", "pusht", "reacher", "cube")
SCENE_OFFICIAL_POLICIES = {
    "action_nce": "scene/lewm_masked_action_nce_e10_s3072",
    "sigreg": "scene/lewm_sigreg_e10_s3072",
}
SCENE_OFFICIAL_TASKS = (1, 2, 3, 4, 5)
SCENE_OFFICIAL_CHUNK_STARTS = (0, 10, 20, 30, 40)
OGB_COMPARISON_METHODS = {
    "action_nce": "lewm_masked_action_nce",
    "sigreg": "lewm",
}
DEFAULT_OGB_COMPARISON_ROOT = "ogb_broad"
OGB_EVAL_SUBDIR_ROOTS = {
    "antsoccer-medium-stitch-v0": "ogb_broad_antsoccer_topdown1",
}
LOCAL_TRAIN_DATASETS = {
    "pusht": ("pusht_expert_train.h5",),
    "reacher": ("dmc/reacher_random.h5",),
    "tworoom": ("tworoom.h5",),
    "ogb": ("ogbench/cube_single_expert.h5",),
    "scene": ("ogbench/visual_scene_play.h5",),
}
MODAL_SECRETS = []
if os.environ.get("WANDB_API_KEY"):
    MODAL_SECRETS.append(
        modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "curl",
        "git",
        "libegl1",
        "libglib2.0-0",
        "libgl1",
        "libopengl0",
        "libosmesa6",
        "swig",
        "zstd",
    )
    .run_commands(
        'python -m pip install "pip<24" "setuptools==65.5.0" "wheel<0.39" "packaging<22"',
        "python -m pip install gym==0.21.0",
        (
            'python -m pip install modal "stable-worldmodel[train,env]==0.0.6" '
            '"stable-pretraining==0.1.6" "dm-control==1.0.41" '
            '"mujoco==3.8.1" "huggingface_hub[hf_transfer]" '
            "hf_transfer wandb"
        ),
    )
    .workdir(str(REPO_DIR))
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HYDRA_FULL_ERROR": "1",
            "MUJOCO_GL": "egl",
            "STABLEWM_HOME": str(STABLEWM_HOME),
        }
    )
    .add_local_dir(
        ".",
        remote_path=str(REPO_DIR),
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            ".stable_worldmodel",
            "wandb",
            "outputs",
            "multirun",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


def _parse_overrides(overrides: str | None) -> list[str]:
    return shlex.split(overrides or "")


def _parse_tasks(tasks: str | None) -> list[str]:
    if not tasks:
        return list(REPORTED_LEWM_TASKS)
    parsed = [item.strip() for item in tasks.replace(",", " ").split() if item.strip()]
    if not parsed:
        return list(REPORTED_LEWM_TASKS)
    return parsed


def _parse_csv_ints(values: str | None, default: tuple[int, ...]) -> list[int]:
    if not values:
        return list(default)
    parsed = [
        int(item.strip())
        for item in values.replace(",", " ").split()
        if item.strip()
    ]
    return parsed or list(default)


def _parse_scene_official_methods(methods: str | None) -> list[str]:
    if not methods:
        return list(SCENE_OFFICIAL_POLICIES)
    parsed = [
        item.strip()
        for item in methods.replace(",", " ").split()
        if item.strip()
    ]
    if not parsed:
        return list(SCENE_OFFICIAL_POLICIES)
    unknown = sorted(set(parsed) - set(SCENE_OFFICIAL_POLICIES))
    if unknown:
        raise ValueError(f"Unknown official Scene method(s): {unknown}")
    return parsed


def _parse_ogb_methods(methods: str | None) -> list[str]:
    if not methods:
        return list(OGB_COMPARISON_METHODS)
    parsed = [
        item.strip()
        for item in methods.replace(",", " ").split()
        if item.strip()
    ]
    if not parsed:
        return list(OGB_COMPARISON_METHODS)
    unknown = sorted(set(parsed) - set(OGB_COMPARISON_METHODS))
    if unknown:
        raise ValueError(f"Unknown OGB comparison method(s): {unknown}")
    return parsed


def _modal_call_marker_path(namespace: str, key: str) -> Path:
    safe_key = "".join(char if char.isalnum() or char in "._-" else "__" for char in key)
    return STABLEWM_HOME / "modal_call_ids" / namespace / f"{safe_key}.txt"


def _attach_or_spawn_call(namespace: str, key: str, fn, **kwargs):
    marker = _modal_call_marker_path(namespace, key)
    volume.reload()
    if marker.exists():
        call_id = marker.read_text(encoding="utf-8").strip()
        if call_id:
            call = modal.FunctionCall.from_id(call_id)
            print(f"Reusing Modal call namespace={namespace} key={key}: {call}", flush=True)
            return call

    call = fn.spawn(**kwargs)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{call.object_id}\n", encoding="utf-8")
    volume.commit()
    return call


def _slugify_ogb_task(task: str) -> str:
    return (
        task.removesuffix("-v0")
        .replace("-", "_")
        .replace("/", "_")
    )


def _scene_official_chunk_filename(
    method: str,
    task_id: int,
    start: int,
    chunk_size: int,
) -> str:
    end = start + chunk_size - 1
    return (
        f"scene_official_{method}_s3072_task{task_id}_"
        f"eps{start}_{end}_n{chunk_size}.txt"
    )


def _scene_official_chunk_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "==== RESULTS ====" in text and "metrics:" in text


def _append_default(overrides: list[str], override: str) -> list[str]:
    key = override.split("=", 1)[0]
    if not any(item.split("=", 1)[0] == key for item in overrides):
        overrides.append(override)
    return overrides


def _default_decoder_path(task: str, policy: str, output_name: str = "decoder") -> str:
    policy_name = policy.replace("/", "__")
    return str(STABLEWM_HOME / "decoders" / task / policy_name / f"{output_name}.pt")


def _run_subprocess(
    cmd: list[str],
    *,
    env_overrides: dict[str, str | Path] | None = None,
    commit_interval_seconds: int | None = None,
) -> dict[str, object]:
    env = os.environ.copy()
    env.update(
        {
            "MUJOCO_GL": "egl",
            "STABLEWM_HOME": str(STABLEWM_HOME),
        }
    )
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})

    print("Running:", shlex.join(cmd), flush=True)
    if commit_interval_seconds and commit_interval_seconds > 0:
        process = subprocess.Popen(cmd, cwd=REPO_DIR, env=env)
        while True:
            try:
                returncode = process.wait(timeout=commit_interval_seconds)
                break
            except subprocess.TimeoutExpired:
                print(
                    f"Committing Modal Volume after {commit_interval_seconds}s",
                    flush=True,
                )
                volume.commit()
    else:
        completed = subprocess.run(cmd, cwd=REPO_DIR, env=env, check=False)
        returncode = completed.returncode

    volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    return {"cmd": cmd, "returncode": returncode}


def _stage_volume_files_to_local(relative_paths: tuple[str, ...]) -> None:
    LOCAL_STABLEWM_HOME.mkdir(parents=True, exist_ok=True)
    for relative_path in relative_paths:
        src = STABLEWM_HOME / relative_path
        dst = LOCAL_STABLEWM_HOME / relative_path
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"Local dataset exists, skipping copy: {dst}", flush=True)
            continue
        print(f"Staging dataset to local disk: {src} -> {dst}", flush=True)
        partial = dst.with_suffix(dst.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        shutil.copy2(src, partial)
        partial.replace(dst)


def _copy_local_run_to_volume(subdir: str) -> None:
    src = LOCAL_STABLEWM_HOME / subdir
    dst = STABLEWM_HOME / subdir
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_symlink():
        print(f"Trained run is volume-backed via symlink: {src} -> {dst}", flush=True)
        volume.commit()
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying trained run back to Modal Volume: {src} -> {dst}", flush=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    volume.commit()


def _link_local_run_to_volume(subdir: str) -> None:
    local_run = LOCAL_STABLEWM_HOME / subdir
    volume_run = STABLEWM_HOME / subdir

    volume_run.mkdir(parents=True, exist_ok=True)
    local_run.parent.mkdir(parents=True, exist_ok=True)

    if local_run.is_symlink():
        local_run.unlink()
    elif local_run.exists():
        print(f"Moving existing local run cache to Modal Volume: {local_run}", flush=True)
        shutil.copytree(local_run, volume_run, dirs_exist_ok=True)
        shutil.rmtree(local_run)

    print(f"Linking local run directory to Modal Volume: {local_run} -> {volume_run}", flush=True)
    local_run.symlink_to(volume_run, target_is_directory=True)
    volume.commit()


def _run_train_impl(
    config_name: str = "lewm_mh",
    data: str = "pusht",
    subdir: str | None = None,
    overrides: str | None = None,
    disable_wandb: bool = False,
    prepare_assets: bool = False,
    prepare_tasks: str | None = None,
    force_prepare_assets: bool = False,
    ogb_task: str | None = None,
    force_prepare_ogb_dataset: bool = False,
    stage_local_data: bool = True,
) -> dict[str, object]:
    """Run train.py on Modal."""

    volume.reload()

    if prepare_assets:
        prepare_cmd = [
            "python",
            "prepare_reported_assets.py",
            f"--tasks={prepare_tasks or data}",
        ]
        if force_prepare_assets:
            prepare_cmd.append("--force")
        prepare_cmd.extend(["--datasets", "--no-checkpoints"])
        _run_subprocess(prepare_cmd)

    override_list = _parse_overrides(overrides)
    ogb_dataset_paths = None
    if ogb_task:
        spec = get_ogb_task_spec(ogb_task)
        prepare_ogb_hdf5(
            task=ogb_task,
            cache_dir=STABLEWM_HOME,
            force=force_prepare_ogb_dataset,
        )
        volume.commit()
        _append_default(override_list, f"data.dataset.name={spec.output_name}")
        ogb_dataset_paths = (f"{spec.output_name}.h5",)

    if subdir:
        _append_default(override_list, f"subdir={subdir}")
    if disable_wandb:
        _append_default(override_list, "wandb.enabled=false")

    env_overrides = None
    commit_interval_seconds = None
    local_train_datasets = ogb_dataset_paths or LOCAL_TRAIN_DATASETS.get(data)
    if stage_local_data and local_train_datasets:
        _stage_volume_files_to_local(local_train_datasets)
        env_overrides = {"STABLEWM_HOME": LOCAL_STABLEWM_HOME}
        commit_interval_seconds = OUTPUT_COMMIT_INTERVAL_SECONDS
        if subdir:
            _link_local_run_to_volume(subdir)

    cmd = [
        "python",
        "train.py",
        f"--config-name={config_name}",
        f"data={data}",
        *override_list,
    ]
    result = _run_subprocess(
        cmd,
        env_overrides=env_overrides,
        commit_interval_seconds=commit_interval_seconds,
    )
    if env_overrides is not None and subdir:
        _copy_local_run_to_volume(subdir)
    return result


@app.function(
    gpu=TRAIN_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
    secrets=MODAL_SECRETS,
)
def run_train(
    config_name: str = "lewm_mh",
    data: str = "pusht",
    subdir: str | None = None,
    overrides: str | None = None,
    disable_wandb: bool = False,
    prepare_assets: bool = False,
    prepare_tasks: str | None = None,
    force_prepare_assets: bool = False,
    ogb_task: str | None = None,
    force_prepare_ogb_dataset: bool = False,
    stage_local_data: bool = True,
) -> dict[str, object]:
    """Run train.py on Modal."""

    return _run_train_impl(
        config_name=config_name,
        data=data,
        subdir=subdir,
        overrides=overrides,
        disable_wandb=disable_wandb,
        prepare_assets=prepare_assets,
        prepare_tasks=prepare_tasks,
        force_prepare_assets=force_prepare_assets,
        ogb_task=ogb_task,
        force_prepare_ogb_dataset=force_prepare_ogb_dataset,
        stage_local_data=stage_local_data,
    )


@app.function(
    gpu=TRAIN_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
    secrets=MODAL_SECRETS,
)
def run_train_then_evaluate(
    config_name: str = "lewm_mh",
    data: str = "pusht",
    subdir: str | None = None,
    overrides: str | None = None,
    disable_wandb: bool = False,
    prepare_assets: bool = False,
    prepare_tasks: str | None = None,
    force_prepare_assets: bool = False,
    ogb_task: str | None = None,
    force_prepare_ogb_dataset: bool = False,
    stage_local_data: bool = True,
    eval_config_name: str = "pusht",
    eval_policy: str = "",
    eval_overrides: str | None = None,
) -> dict[str, object]:
    """Train, then evaluate a checkpoint from the trained policy directory."""

    train_result = _run_train_impl(
        config_name=config_name,
        data=data,
        subdir=subdir,
        overrides=overrides,
        disable_wandb=disable_wandb,
        prepare_assets=prepare_assets,
        prepare_tasks=prepare_tasks,
        force_prepare_assets=force_prepare_assets,
        ogb_task=ogb_task,
        force_prepare_ogb_dataset=force_prepare_ogb_dataset,
        stage_local_data=stage_local_data,
    )

    policy = eval_policy or (subdir or "")
    if not policy:
        raise ValueError("eval_policy or subdir must be provided for train_then_evaluate")

    eval_override_list = _parse_overrides(eval_overrides)
    if ogb_task:
        spec = get_ogb_task_spec(ogb_task)
        _append_default(eval_override_list, f"eval.dataset_name={spec.output_name}")
        _append_default(eval_override_list, f"eval.ogb_dataset_name={spec.env_dataset}")
        if spec.eval_solver != "cem":
            _append_default(eval_override_list, f"solver={spec.eval_solver}")
        if spec.synthetic_pixels:
            _append_default(eval_override_list, f"eval.synthetic_pixels={spec.synthetic_pixels}")
        for key, value in spec.env_kwargs.items():
            _append_default(eval_override_list, f"+eval.env_kwargs.{key}={value}")

    eval_cmd = [
        "python",
        "eval.py",
        f"--config-name={eval_config_name}",
        f"policy={policy}",
        *eval_override_list,
    ]
    eval_result = _run_subprocess(eval_cmd)
    return {"train": train_result, "eval": eval_result}


@app.function(
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_ogb_comparison_task(
    task: str,
    methods: str = "action_nce,sigreg",
    seed: int = 3072,
    eval_seed: int = 42,
    eval_num: int = 50,
    eval_env_batch_size: int = 5,
    force_prepare_ogb_dataset: bool = False,
    subdir_root: str = DEFAULT_OGB_COMPARISON_ROOT,
) -> dict[str, object]:
    """Prepare one generic OGB task once, then run both comparison methods."""

    spec = get_ogb_task_spec(task)
    task_slug = _slugify_ogb_task(task)
    method_list = _parse_ogb_methods(methods)

    volume.reload()
    prepare_ogb_hdf5(
        task=task,
        cache_dir=STABLEWM_HOME,
        force=force_prepare_ogb_dataset,
    )
    volume.commit()

    calls = []
    for method in method_list:
        config_name = OGB_COMPARISON_METHODS[method]
        subdir = f"{subdir_root}/{task_slug}/{method}_s{seed}"
        output_filename = f"{task_slug}_{method}_s{seed}_n{eval_num}.txt"
        train_overrides = f"seed={seed} early_stopping.enabled=false"
        eval_overrides = (
            f"eval.num_eval={eval_num} "
            f"eval.env_batch_size={eval_env_batch_size} "
            f"seed={eval_seed} "
            "output.save_video=false "
            f"output.filename={output_filename}"
        )
        call = _attach_or_spawn_call(
            "ogb_comparison_lane",
            f"{subdir}|eval_seed={eval_seed}|eval_num={eval_num}",
            run_train_then_evaluate,
            config_name=config_name,
            data="ogb_generic",
            subdir=subdir,
            overrides=train_overrides,
            disable_wandb=False,
            prepare_assets=False,
            prepare_tasks=None,
            force_prepare_assets=False,
            ogb_task=task,
            force_prepare_ogb_dataset=False,
            stage_local_data=True,
            eval_config_name="ogb_generic",
            eval_policy=subdir,
            eval_overrides=eval_overrides,
        )
        print(
            f"Spawned OGB comparison method={method} task={task} "
            f"subdir={subdir}: {call}",
            flush=True,
        )
        calls.append((method, call))

    results = []
    for method, call in calls:
        print(f"Waiting for OGB comparison method={method} task={task}", flush=True)
        result = call.get()
        print(
            f"Completed OGB comparison method={method} task={task}: {result}",
            flush=True,
        )
        results.append({"method": method, "result": result})

    return {
        "task": task,
        "source_dataset": spec.source_dataset,
        "output_dataset": spec.output_name,
        "methods": method_list,
        "subdir_root": subdir_root,
        "results": results,
    }


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_ogb_comparison_supervisor(
    tasks: str,
    methods: str = "action_nce,sigreg",
    seed: int = 3072,
    eval_seed: int = 42,
    eval_num: int = 50,
    eval_env_batch_size: int = 5,
    force_prepare_ogb_dataset: bool = False,
    subdir_root: str = DEFAULT_OGB_COMPARISON_ROOT,
) -> dict[str, object]:
    """Modal-resident supervisor for broad OGB task comparisons."""

    task_list = [item.strip() for item in tasks.replace(",", " ").split() if item.strip()]
    if not task_list:
        raise ValueError("At least one OGB task is required")

    calls = []
    for task in task_list:
        call = _attach_or_spawn_call(
            "ogb_comparison_task",
            f"{subdir_root}|{task}|methods={methods}|seed={seed}|eval_seed={eval_seed}|eval_num={eval_num}",
            run_ogb_comparison_task,
            task=task,
            methods=methods,
            seed=seed,
            eval_seed=eval_seed,
            eval_num=eval_num,
            eval_env_batch_size=eval_env_batch_size,
            force_prepare_ogb_dataset=force_prepare_ogb_dataset,
            subdir_root=subdir_root,
        )
        print(f"Spawned OGB task supervisor task={task}: {call}", flush=True)
        calls.append((task, call))

    results = []
    for task, call in calls:
        print(f"Waiting for OGB task supervisor task={task}", flush=True)
        result = call.get()
        print(f"Completed OGB task supervisor task={task}: {result}", flush=True)
        results.append(result)

    return {
        "tasks": task_list,
        "methods": _parse_ogb_methods(methods),
        "subdir_root": subdir_root,
        "results": results,
    }


@app.function(
    cpu=1.0,
    memory=4096,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_ogb_eval_matrix_task(
    task: str,
    methods: str = "action_nce,sigreg",
    seed: int = 3072,
    eval_seed: int = 42,
    eval_num: int = 50,
    eval_env_batch_size: int = 5,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    subdir_root: str = "ogb_broad_clean1",
    force_prepare_ogb_dataset: bool = False,
    cem_num_samples: int = 64,
    cem_n_steps: int = 10,
    cem_topk: int = 8,
    pgd_num_samples: int = 64,
    pgd_n_steps: int = 10,
) -> dict[str, object]:
    """Run corrected OGB trajectory evals for one task from existing checkpoints."""

    spec = get_ogb_task_spec(task)
    task_slug = _slugify_ogb_task(task)
    method_list = _parse_ogb_methods(methods)
    policy_root = OGB_EVAL_SUBDIR_ROOTS.get(task, subdir_root)

    volume.reload()
    prepare_ogb_hdf5(
        task=task,
        cache_dir=STABLEWM_HOME,
        force=force_prepare_ogb_dataset,
    )
    volume.commit()

    calls = []
    for method in method_list:
        policy = f"{policy_root}/{task_slug}/{method}_s{seed}"
        output_filename = (
            f"{task_slug}_{method}_s{seed}_traj_g{goal_offset_steps}"
            f"_b{eval_budget}_n{eval_num}.txt"
        )
        eval_override_list = [
            "eval.protocol=ogb_trajectory",
            f"eval.dataset_name={spec.output_name}",
            f"eval.ogb_dataset_name={spec.env_dataset}",
            f"eval.num_eval={eval_num}",
            f"eval.env_batch_size={eval_env_batch_size}",
            f"eval.goal_offset_steps={goal_offset_steps}",
            f"eval.eval_budget={eval_budget}",
            f"seed={eval_seed}",
            "output.save_video=false",
            f"output.filename={output_filename}",
        ]
        if spec.eval_solver != "cem":
            eval_override_list.extend(
                [
                    f"solver={spec.eval_solver}",
                    f"solver.num_samples={pgd_num_samples}",
                    f"solver.n_steps={pgd_n_steps}",
                ]
            )
        else:
            eval_override_list.extend(
                [
                    f"solver.num_samples={cem_num_samples}",
                    f"solver.n_steps={cem_n_steps}",
                    f"solver.topk={cem_topk}",
                ]
            )
        if spec.synthetic_pixels:
            eval_override_list.append(f"eval.synthetic_pixels={spec.synthetic_pixels}")
        for key, value in spec.env_kwargs.items():
            eval_override_list.append(f"+eval.env_kwargs.{key}={value}")

        overrides = " ".join(shlex.quote(item) for item in eval_override_list)
        call = _attach_or_spawn_call(
            "ogb_eval_matrix_lane",
            (
                f"{policy}|protocol=trajectory|seed={eval_seed}|n={eval_num}|"
                f"g={goal_offset_steps}|b={eval_budget}"
            ),
            run_eval,
            config_name="ogb_generic",
            policy=policy,
            overrides=overrides,
        )
        print(
            f"Spawned OGB trajectory eval method={method} task={task} "
            f"policy={policy}: {call}",
            flush=True,
        )
        calls.append((method, call, output_filename))

    results = []
    for method, call, output_filename in calls:
        print(f"Waiting for OGB trajectory eval method={method} task={task}", flush=True)
        result = call.get()
        print(
            f"Completed OGB trajectory eval method={method} task={task}: {result}",
            flush=True,
        )
        results.append(
            {"method": method, "output_filename": output_filename, "result": result}
        )

    return {
        "task": task,
        "source_dataset": spec.source_dataset,
        "output_dataset": spec.output_name,
        "methods": method_list,
        "policy_root": policy_root,
        "results": results,
    }


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_ogb_eval_matrix_supervisor(
    tasks: str,
    methods: str = "action_nce,sigreg",
    seed: int = 3072,
    eval_seed: int = 42,
    eval_num: int = 50,
    eval_env_batch_size: int = 5,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    subdir_root: str = "ogb_broad_clean1",
    force_prepare_ogb_dataset: bool = False,
    cem_num_samples: int = 64,
    cem_n_steps: int = 10,
    cem_topk: int = 8,
    pgd_num_samples: int = 64,
    pgd_n_steps: int = 10,
) -> dict[str, object]:
    """Modal-resident supervisor for corrected broad OGB trajectory evals."""

    task_list = [item.strip() for item in tasks.replace(",", " ").split() if item.strip()]
    if not task_list:
        raise ValueError("At least one OGB task is required")

    calls = []
    for task in task_list:
        call = _attach_or_spawn_call(
            "ogb_eval_matrix_task",
            (
                f"{subdir_root}|{task}|methods={methods}|seed={seed}|"
                f"eval_seed={eval_seed}|n={eval_num}|g={goal_offset_steps}|"
                f"b={eval_budget}"
            ),
            run_ogb_eval_matrix_task,
            task=task,
            methods=methods,
            seed=seed,
            eval_seed=eval_seed,
            eval_num=eval_num,
            eval_env_batch_size=eval_env_batch_size,
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget,
            subdir_root=subdir_root,
            force_prepare_ogb_dataset=force_prepare_ogb_dataset,
            cem_num_samples=cem_num_samples,
            cem_n_steps=cem_n_steps,
            cem_topk=cem_topk,
            pgd_num_samples=pgd_num_samples,
            pgd_n_steps=pgd_n_steps,
        )
        print(f"Spawned OGB trajectory eval task={task}: {call}", flush=True)
        calls.append((task, call))

    results = []
    for task, call in calls:
        print(f"Waiting for OGB trajectory eval task={task}", flush=True)
        result = call.get()
        print(f"Completed OGB trajectory eval task={task}: {result}", flush=True)
        results.append(result)

    return {
        "tasks": task_list,
        "methods": _parse_ogb_methods(methods),
        "subdir_root": subdir_root,
        "results": results,
    }


@app.function(
    gpu=DECODER_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_decoder_train(
    task: str = "pusht",
    policy: str = "pusht/lewm",
    overrides: str | None = None,
) -> dict[str, object]:
    """Run train_decoder.py on Modal."""

    volume.reload()
    cmd = [
        "python",
        "train_decoder.py",
        f"task={task}",
        f"policy={policy}",
        *_parse_overrides(overrides),
    ]
    return _run_subprocess(cmd)


@app.function(
    gpu=EVAL_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_eval(
    config_name: str = "pusht",
    policy: str = "random",
    overrides: str | None = None,
) -> dict[str, object]:
    """Run eval.py on Modal."""

    volume.reload()
    cmd = [
        "python",
        "eval.py",
        f"--config-name={config_name}",
        f"policy={policy}",
        *_parse_overrides(overrides),
    ]
    return _run_subprocess(
        cmd,
        commit_interval_seconds=OUTPUT_COMMIT_INTERVAL_SECONDS,
    )


@app.function(
    gpu=EVAL_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_scene_official_chunk_lane(
    method: str,
    task_id: int,
    starts: str | None = None,
    chunk_size: int = 10,
    max_attempts: int = 3,
) -> dict[str, object]:
    """Run one official Scene method/task lane entirely inside Modal.

    The lane writes one result file per episode chunk. If Modal retries the
    function after a worker failure, already committed chunks are skipped.
    """

    if method not in SCENE_OFFICIAL_POLICIES:
        raise ValueError(f"Unknown official Scene method: {method}")
    task_id = int(task_id)
    chunk_size = int(chunk_size)
    max_attempts = int(max_attempts)
    chunk_starts = _parse_csv_ints(starts, SCENE_OFFICIAL_CHUNK_STARTS)
    policy = SCENE_OFFICIAL_POLICIES[method]

    volume.reload()
    completed: list[str] = []
    skipped: list[str] = []

    for start in chunk_starts:
        filename = _scene_official_chunk_filename(method, task_id, start, chunk_size)
        result_path = STABLEWM_HOME / "scene" / filename
        if _scene_official_chunk_complete(result_path):
            print(f"Skipping completed official Scene chunk: {filename}", flush=True)
            skipped.append(filename)
            continue

        overrides = (
            f"eval.num_eval={chunk_size} "
            f"eval.episode_start={start} "
            f"eval.task_ids=[{task_id}] "
            "eval.env_batch_size=1 "
            f"output.filename={filename} "
            "output.save_video=false"
        )
        cmd = [
            "python",
            "eval.py",
            "--config-name=scene_official",
            f"policy={policy}",
            *_parse_overrides(overrides),
        ]

        for attempt in range(1, max_attempts + 1):
            print(
                f"Official Scene lane method={method} task={task_id} "
                f"chunk={start}-{start + chunk_size - 1} attempt={attempt}",
                flush=True,
            )
            try:
                _run_subprocess(
                    cmd,
                    commit_interval_seconds=OUTPUT_COMMIT_INTERVAL_SECONDS,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"Official Scene chunk subprocess failed rc={exc.returncode}: "
                    f"{filename}",
                    flush=True,
                )

            volume.reload()
            if _scene_official_chunk_complete(result_path):
                print(f"Verified official Scene chunk: {filename}", flush=True)
                completed.append(filename)
                break

            volume.commit()
            if attempt < max_attempts:
                print(f"Retrying missing official Scene chunk: {filename}", flush=True)
                time.sleep(60)
        else:
            raise RuntimeError(
                f"Failed to produce official Scene chunk after {max_attempts} "
                f"attempts: {filename}"
            )

    return {
        "method": method,
        "task_id": task_id,
        "starts": chunk_starts,
        "completed": completed,
        "skipped": skipped,
    }


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_scene_official_chunk_supervisor(
    methods: str = "action_nce,sigreg",
    tasks: str = "1,2,3,4,5",
    starts: str = "0,10,20,30,40",
    chunk_size: int = 10,
    max_attempts: int = 3,
) -> dict[str, object]:
    """Remote supervisor for all official Scene chunk lanes.

    The local client triggers only this function. This supervisor then runs
    inside Modal, spawns the GPU lane functions, and waits for them there.
    """

    method_list = _parse_scene_official_methods(methods)
    task_ids = _parse_csv_ints(tasks, SCENE_OFFICIAL_TASKS)
    start_values = _parse_csv_ints(starts, SCENE_OFFICIAL_CHUNK_STARTS)
    start_spec = ",".join(str(value) for value in start_values)

    calls = []
    for method in method_list:
        for task_id in task_ids:
            call = run_scene_official_chunk_lane.spawn(
                method=method,
                task_id=int(task_id),
                starts=start_spec,
                chunk_size=int(chunk_size),
                max_attempts=int(max_attempts),
            )
            print(
                f"Remote supervisor spawned official Scene lane "
                f"method={method} task={task_id}: {call}",
                flush=True,
            )
            calls.append((method, int(task_id), call))

    results = []
    for method, task_id, call in calls:
        print(
            f"Remote supervisor waiting for official Scene lane "
            f"method={method} task={task_id}",
            flush=True,
        )
        result = call.get()
        print(
            f"Remote supervisor completed official Scene lane "
            f"method={method} task={task_id}: {result}",
            flush=True,
        )
        results.append(result)

    return {"methods": method_list, "tasks": task_ids, "starts": start_values, "results": results}


@app.function(
    gpu=EVAL_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_probe(
    policy: str,
    dataset: str = "pusht_expert_train",
    n: int = 4000,
    overrides: str | None = None,
) -> dict[str, object]:
    """Run probe.py (linear state-decoding probe) on Modal."""

    volume.reload()
    cmd = [
        "python",
        "probe.py",
        "--policy",
        policy,
        "--dataset",
        dataset,
        "--n",
        str(n),
        *_parse_overrides(overrides),
    ]
    return _run_subprocess(cmd)


@app.function(
    gpu=EVAL_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_probe_features(
    policy: str,
    dataset: str = "pusht_expert_train",
    n: int = 4000,
    overrides: str | None = None,
) -> dict[str, object]:
    """Run probe_features.py (CLS vs patch-token orientation decodability) on Modal."""

    volume.reload()
    cmd = [
        "python",
        "probe_features.py",
        "--policy",
        policy,
        "--dataset",
        dataset,
        "--n",
        str(n),
        *_parse_overrides(overrides),
    ]
    return _run_subprocess(cmd)


@app.function(
    gpu=EVAL_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_surprise_diagnostics(
    policy: str,
    dataset: str,
    n: int = 4096,
    overrides: str | None = None,
) -> dict[str, object]:
    """Run latent surprise/counterfactual diagnostics on Modal."""

    volume.reload()
    cmd = [
        "python",
        "surprise_diagnostics.py",
        "--policy",
        policy,
        "--dataset",
        dataset,
        "--n",
        str(n),
        *_parse_overrides(overrides),
    ]
    return _run_subprocess(cmd)


@app.function(
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_prepare_reported_assets(
    tasks: str = ",".join(REPORTED_LEWM_TASKS),
    force: bool = False,
    datasets: bool = True,
    checkpoints: bool = True,
) -> dict[str, object]:
    """Download/extract reported datasets and convert HF checkpoints on Modal."""

    volume.reload()
    cmd = [
        "python",
        "prepare_reported_assets.py",
        f"--tasks={tasks}",
    ]
    if force:
        cmd.append("--force")
    cmd.append("--datasets" if datasets else "--no-datasets")
    cmd.append("--checkpoints" if checkpoints else "--no-checkpoints")
    return _run_subprocess(cmd)


@app.function(
    gpu=VISUALIZE_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    volumes={str(STABLEWM_HOME): volume},
)
def run_visualize_predictions(
    task: str = "pusht",
    policy: str = "pusht/lewm",
    overrides: str | None = None,
) -> dict[str, object]:
    """Run visualize_predictions.py on Modal."""

    volume.reload()
    cmd = [
        "python",
        "visualize_predictions.py",
        f"task={task}",
        f"policy={policy}",
        *_parse_overrides(overrides),
    ]
    return _run_subprocess(cmd)


@app.local_entrypoint()
def train(
    config_name: str = "lewm_mh",
    data: str = "pusht",
    subdir: str = "",
    overrides: str = "",
    disable_wandb: bool = False,
    prepare_assets: bool = False,
    prepare_tasks: str = "",
    force_prepare_assets: bool = False,
    ogb_task: str = "",
    force_prepare_ogb_dataset: bool = False,
    stage_local_data: bool = True,
) -> None:
    """Launch a training job on Modal.

    Keep this entrypoint blocking and use the Modal CLI's global
    ``run --detach`` option for disconnect protection. Do not add an
    entrypoint-level ``--no-wait`` path; returning after ``spawn`` can stop the
    app before the remote work actually runs.
    """

    return run_train.remote(
        config_name=config_name,
        data=data,
        subdir=subdir or None,
        overrides=overrides,
        disable_wandb=disable_wandb,
        prepare_assets=prepare_assets,
        prepare_tasks=prepare_tasks or None,
        force_prepare_assets=force_prepare_assets,
        ogb_task=ogb_task or None,
        force_prepare_ogb_dataset=force_prepare_ogb_dataset,
        stage_local_data=stage_local_data,
    )


@app.local_entrypoint()
def train_then_evaluate(
    config_name: str = "lewm_mh",
    data: str = "pusht",
    subdir: str = "",
    overrides: str = "",
    disable_wandb: bool = False,
    prepare_assets: bool = False,
    prepare_tasks: str = "",
    force_prepare_assets: bool = False,
    ogb_task: str = "",
    force_prepare_ogb_dataset: bool = False,
    stage_local_data: bool = True,
    eval_config_name: str = "pusht",
    eval_policy: str = "",
    eval_overrides: str = "",
) -> None:
    """Launch a train-then-evaluate job on Modal.

    Use ``modal run --detach`` at the CLI layer for long jobs; this entrypoint
    intentionally blocks on the remote function.
    """

    kwargs = {
        "config_name": config_name,
        "data": data,
        "subdir": subdir or None,
        "overrides": overrides,
        "disable_wandb": disable_wandb,
        "prepare_assets": prepare_assets,
        "prepare_tasks": prepare_tasks or None,
        "force_prepare_assets": force_prepare_assets,
        "ogb_task": ogb_task or None,
        "force_prepare_ogb_dataset": force_prepare_ogb_dataset,
        "stage_local_data": stage_local_data,
        "eval_config_name": eval_config_name,
        "eval_policy": eval_policy,
        "eval_overrides": eval_overrides,
    }
    return run_train_then_evaluate.remote(**kwargs)


@app.local_entrypoint()
def decoder(
    task: str = "pusht",
    policy: str = "pusht/lewm",
    overrides: str = "",
) -> None:
    """Launch an auxiliary decoder training job on Modal."""

    call = run_decoder_train.spawn(
        task=task,
        policy=policy,
        overrides=overrides,
    )
    call.get()


@app.local_entrypoint()
def evaluate(
    config_name: str = "pusht",
    policy: str = "random",
    overrides: str = "",
) -> None:
    """Launch an evaluation job on Modal.

    This entrypoint intentionally blocks on ``run_eval.remote``. Use the Modal
    CLI's global ``run --detach`` option so the app survives a local disconnect;
    do not use an entrypoint-level ``--no-wait`` spawn path.
    """

    run_eval.remote(config_name=config_name, policy=policy, overrides=overrides)


@app.local_entrypoint()
def scene_official_chunks(
    methods: str = "action_nce,sigreg",
    tasks: str = "1,2,3,4,5",
    starts: str = "0,10,20,30,40",
    chunk_size: int = 10,
    max_attempts: int = 3,
) -> None:
    """Launch a Modal-resident official Scene chunk supervisor."""

    run_scene_official_chunk_supervisor.remote(
        methods=methods,
        tasks=tasks,
        starts=starts,
        chunk_size=chunk_size,
        max_attempts=max_attempts,
    )


@app.local_entrypoint()
def ogb_comparison(
    tasks: str,
    methods: str = "action_nce,sigreg",
    seed: int = 3072,
    eval_seed: int = 42,
    eval_num: int = 50,
    eval_env_batch_size: int = 5,
    force_prepare_ogb_dataset: bool = False,
    subdir_root: str = DEFAULT_OGB_COMPARISON_ROOT,
) -> None:
    """Launch broad OGB train/eval comparisons via Modal-resident supervision."""

    run_ogb_comparison_supervisor.remote(
        tasks=tasks,
        methods=methods,
        seed=seed,
        eval_seed=eval_seed,
        eval_num=eval_num,
        eval_env_batch_size=eval_env_batch_size,
        force_prepare_ogb_dataset=force_prepare_ogb_dataset,
        subdir_root=subdir_root,
    )


@app.local_entrypoint()
def ogb_eval_matrix(
    tasks: str,
    methods: str = "action_nce,sigreg",
    seed: int = 3072,
    eval_seed: int = 42,
    eval_num: int = 50,
    eval_env_batch_size: int = 5,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    subdir_root: str = "ogb_broad_clean1",
    force_prepare_ogb_dataset: bool = False,
    cem_num_samples: int = 64,
    cem_n_steps: int = 10,
    cem_topk: int = 8,
    pgd_num_samples: int = 64,
    pgd_n_steps: int = 10,
) -> None:
    """Launch corrected broad OGB trajectory evals from existing checkpoints."""

    run_ogb_eval_matrix_supervisor.remote(
        tasks=tasks,
        methods=methods,
        seed=seed,
        eval_seed=eval_seed,
        eval_num=eval_num,
        eval_env_batch_size=eval_env_batch_size,
        goal_offset_steps=goal_offset_steps,
        eval_budget=eval_budget,
        subdir_root=subdir_root,
        force_prepare_ogb_dataset=force_prepare_ogb_dataset,
        cem_num_samples=cem_num_samples,
        cem_n_steps=cem_n_steps,
        cem_topk=cem_topk,
        pgd_num_samples=pgd_num_samples,
        pgd_n_steps=pgd_n_steps,
    )


@app.local_entrypoint()
def probe(
    policy: str = "pusht/lewm_masked",
    dataset: str = "pusht_expert_train",
    n: int = 4000,
    overrides: str = "",
) -> None:
    """Launch a linear state-decoding probe on Modal. Results print to logs as
    `PROBE_R2 ...` lines (read them with `modal app logs`)."""

    run_probe.remote(policy=policy, dataset=dataset, n=n, overrides=overrides)


@app.local_entrypoint()
def probe_features(
    policy: str = "pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10",
    dataset: str = "pusht_expert_train",
    n: int = 4000,
    overrides: str = "",
) -> None:
    """Compare CLS-latent vs patch-token orientation decodability on Modal.
    Results print as `PROBE_R2 <rep> state[d] = ...` lines."""

    run_probe_features.remote(policy=policy, dataset=dataset, n=n, overrides=overrides)


@app.local_entrypoint()
def surprise_diagnostics(
    policy: str,
    dataset: str,
    n: int = 4096,
    overrides: str = "",
) -> None:
    """Launch latent surprise/counterfactual diagnostics on Modal."""

    run_surprise_diagnostics.remote(
        policy=policy,
        dataset=dataset,
        n=n,
        overrides=overrides,
    )


@app.local_entrypoint()
def visualize(
    task: str = "pusht",
    policy: str = "pusht/lewm",
    overrides: str = "",
) -> None:
    """Launch a prediction visualization job on Modal."""

    call = run_visualize_predictions.spawn(
        task=task,
        policy=policy,
        overrides=overrides,
    )
    call.get()


@app.local_entrypoint()
def prepare_reported_assets(
    tasks: str = ",".join(REPORTED_LEWM_TASKS),
    force: bool = False,
    datasets: bool = True,
    checkpoints: bool = True,
) -> None:
    """Prepare reported LeWM datasets/checkpoints in the Modal Volume."""

    call = run_prepare_reported_assets.spawn(
        tasks=tasks,
        force=force,
        datasets=datasets,
        checkpoints=checkpoints,
    )
    call.get()


@app.local_entrypoint()
def reproduce_checkpoint(
    task: str = "pusht",
    policy: str = "pusht/lewm",
    eval_overrides: str = "",
    decoder_overrides: str = "max_samples=50000 trainer.max_epochs=3",
    visualize_overrides: str = "num_rollouts=8",
    train_decoder: bool = True,
    run_eval_job: bool = True,
    run_visualize_job: bool = True,
) -> None:
    """Evaluate a checkpoint and optionally build decoded prediction figures."""

    if run_eval_job:
        eval_call = run_eval.spawn(
            config_name=task,
            policy=policy,
            overrides=eval_overrides,
        )
        eval_call.get()

    if train_decoder:
        decoder_call = run_decoder_train.spawn(
            task=task,
            policy=policy,
            overrides=decoder_overrides,
        )
        decoder_call.get()

    if run_visualize_job:
        visualize_override_list = _parse_overrides(visualize_overrides)
        if train_decoder:
            _append_default(
                visualize_override_list,
                f"decoder.path={_default_decoder_path(task, policy)}",
            )
        visualize_call = run_visualize_predictions.spawn(
            task=task,
            policy=policy,
            overrides=shlex.join(visualize_override_list),
        )
        visualize_call.get()


@app.local_entrypoint()
def reproduce_reported_results(
    tasks: str = ",".join(REPORTED_LEWM_TASKS),
    policy_template: str = "{task}/lewm",
    eval_overrides: str = "output.save_video=false eval.env_batch_size=10",
    decoder_overrides: str = "max_samples=50000 trainer.max_epochs=3",
    visualize_overrides: str = "num_rollouts=8",
    prepare_assets: bool = True,
    force_prepare_assets: bool = False,
    train_decoder: bool = True,
    run_eval_job: bool = True,
    run_visualize_job: bool = True,
) -> None:
    """Reproduce released LeWM checkpoint results and train decoders per task."""

    task_list = _parse_tasks(tasks)
    task_spec = ",".join(task_list)

    if prepare_assets:
        prepare_call = run_prepare_reported_assets.spawn(
            tasks=task_spec,
            force=force_prepare_assets,
            datasets=True,
            checkpoints=True,
        )
        prepare_call.get()

    for task in task_list:
        policy = policy_template.format(task=task)
        print(f"=== Reproducing task={task} policy={policy} ===", flush=True)

        if run_eval_job:
            eval_call = run_eval.spawn(
                config_name=task,
                policy=policy,
                overrides=eval_overrides,
            )
            eval_call.get()

        if train_decoder:
            decoder_call = run_decoder_train.spawn(
                task=task,
                policy=policy,
                overrides=decoder_overrides,
            )
            decoder_call.get()

        if run_visualize_job:
            visualize_override_list = _parse_overrides(visualize_overrides)
            if train_decoder:
                _append_default(
                    visualize_override_list,
                    f"decoder.path={_default_decoder_path(task, policy)}",
                )
            visualize_call = run_visualize_predictions.spawn(
                task=task,
                policy=policy,
                overrides=shlex.join(visualize_override_list),
            )
            visualize_call.get()


@app.local_entrypoint()
def upload(local_path: str, remote_path: str = "/") -> None:
    """Upload local datasets or checkpoints into the Modal STABLEWM_HOME volume."""

    local = Path(local_path).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(local)

    remote = remote_path if remote_path.startswith("/") else f"/{remote_path}"
    with volume.batch_upload() as batch:
        if local.is_dir():
            batch.put_directory(local, remote)
        else:
            if remote.endswith("/"):
                remote = f"{remote}{local.name}"
            batch.put_file(local, remote)

    print(f"Uploaded {local} to modal volume {VOLUME_NAME}:{remote}")
