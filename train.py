import os
import re
import subprocess
import types
from functools import partial
from pathlib import Path

from project_paths import configure_stablewm_home

configure_stablewm_home()

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
import copy

from module import ARPredictor, ConvDecoder, Embedder, FutureQueryPredictor, HorizonInverseDynamics, InverseDynamics, MLP, SIGReg, VarianceReg
from utils import (
    JsonlMetricsCallback,
    MinEpochEarlyStopping,
    ModelObjectCallBack,
    ProgressBarCallback,
    RNGStateCallback,
    ResumableCheckpointCallback,
    StopAfterEpochCallback,
    get_column_normalizer,
    get_img_preprocessor,
    resolve_resume_checkpoint,
    write_json_atomic,
)


def _run_git_command(args):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def get_git_metadata():
    status = _run_git_command(["status", "--short"])
    return {
        "commit": _run_git_command(["rev-parse", "HEAD"]),
        "branch": _run_git_command(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def boolish(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "on"}


def wandb_enabled(cfg):
    enabled = cfg.wandb.get("enabled", "auto")
    if isinstance(enabled, str) and enabled.lower() == "auto":
        return bool(os.environ.get("WANDB_API_KEY"))
    return boolish(enabled)


def build_wandb_logger(cfg, run_dir: Path, run_metadata: dict):
    if not wandb_enabled(cfg):
        print(
            "WandB disabled. Set WANDB_API_KEY or wandb.enabled=true to enable cloud logging.",
            flush=True,
        )
        return None

    wandb_kwargs = OmegaConf.to_container(cfg.wandb.config, resolve=True) or {}
    wandb_kwargs = {key: value for key, value in wandb_kwargs.items() if value is not None}
    wandb_kwargs.setdefault("save_dir", str(run_dir / "wandb"))
    wandb_kwargs.setdefault("id", str(cfg.get("subdir") or run_dir.name))
    wandb_kwargs["id"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(wandb_kwargs["id"]))
    wandb_kwargs.setdefault("resume", "allow")

    if not os.environ.get("WANDB_API_KEY") and wandb_kwargs.get("mode") != "offline":
        print(
            "WANDB_API_KEY is not set; using WandB offline mode for this run.",
            flush=True,
        )
        wandb_kwargs["mode"] = "offline"

    def init_logger(kwargs):
        logger = WandbLogger(**kwargs)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config.update(
            {"run_metadata": run_metadata}, allow_val_change=True
        )
        return logger

    try:
        return init_logger(wandb_kwargs)
    except Exception as exc:
        if "entity" in wandb_kwargs:
            entity = wandb_kwargs.pop("entity")
            print(
                f"WandB init failed for entity={entity!r}; retrying with default account. "
                f"Original error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                return init_logger(wandb_kwargs)
            except Exception as retry_exc:
                exc = retry_exc
        print(
            f"WandB init failed; continuing with local JSONL logs only. "
            f"Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def make_loader_kwargs(loader_cfg):
    kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
    num_workers = int(kwargs.get("num_workers", 0) or 0)
    if num_workers <= 0:
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)
        kwargs["persistent_workers"] = False
    return kwargs


def configure_torch_runtime(cfg):
    runtime_cfg = cfg.get("runtime", {})
    torch.multiprocessing.set_sharing_strategy(
        str(runtime_cfg.get("sharing_strategy", "file_system"))
    )
    matmul_precision = runtime_cfg.get("float32_matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))


def check_resume_config_compatibility(run_dir: Path, cfg, resume_path: Path | None):
    if resume_path is None or bool(cfg.resume.get("allow_config_mismatch", False)):
        return
    previous_config_path = run_dir / "config.yaml"
    if not previous_config_path.is_file():
        return

    previous = OmegaConf.load(previous_config_path)
    current = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    keys_to_compare = [
        "data",
        "encoder_scale",
        "img_size",
        "patch_size",
        "wm",
        "predictor",
        "loss",
        "optimizer",
        "output_model_name",
    ]
    mismatches = []
    for key in keys_to_compare:
        if OmegaConf.select(previous, key) != OmegaConf.select(current, key):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            f"Refusing to resume {resume_path} because config keys differ: "
            f"{', '.join(mismatches)}. Set resume.allow_config_mismatch=true only "
            "if this is intentional."
        )


def attach_gradient_norm_logger(module, every_n_steps: int):
    every_n_steps = int(every_n_steps or 0)
    if every_n_steps <= 0:
        return

    def after_manual_backward(self):
        step = int(getattr(self.trainer, "global_step", 0))
        if step <= 0 or step % every_n_steps != 0:
            return
        total = 0.0
        for param in self.parameters(with_callbacks=False):
            if param.grad is None:
                continue
            norm = param.grad.detach().float().norm(2)
            total += float(norm.item() ** 2)
        self._last_grad_norm = total ** 0.5

    module.after_manual_backward = types.MethodType(after_manual_backward, module)


def dump_run_metadata(run_dir: Path, cfg, resume_path: Path | None, metadata: dict):
    payload = {
        "run_dir": str(run_dir),
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "wandb_enabled": wandb_enabled(cfg),
        "metadata": metadata,
    }
    write_json_atomic(run_dir / "run_metadata.json", payload)


def build_scheduler_section(cfg):
    scheduler_cfg = OmegaConf.to_container(
        cfg.get("scheduler", {"type": "LinearWarmupCosineAnnealingLR"}), resolve=True
    )
    scheduler_cfg = dict(scheduler_cfg or {})
    interval = scheduler_cfg.pop("interval", "step")
    frequency = int(scheduler_cfg.pop("frequency", 1) or 1)
    monitor = scheduler_cfg.pop("monitor", None)
    return scheduler_cfg, interval, frequency, monitor


def _representation_diagnostics(emb):
    """Gradient-free latent-collapse monitors for one batch (cheap, no autograd).

    ``emb``: (B, T, D). Returns detached scalar tensors added to the step output:

    * ``emb_std`` -- mean per-dimension std across the batch+time axis. Goes to
      ~0 when the encoder collapses to a constant.
    * ``effective_rank`` -- participation ratio of the latent covariance, in
      [1, D]. ~1 means the representation has collapsed onto a single direction
      (very low rank); ~D means it is isotropic / whitened. Catches *subspace*
      collapse that ``emb_std`` misses -- a latent can keep a healthy std while
      quietly losing rank into the few directions an inverse head reads out.

    Effective rank uses only the DxD covariance, no eigendecomposition:
    ``PR = tr(C)^2 / ||C||_F^2 = (sum_i var_i)^2 / sum_ij C_ij^2``. The cost is a
    single (D, N) x (N, D) matmul under ``no_grad`` -- negligible next to the
    encoder forward pass, and it never touches the training graph.
    """
    with torch.no_grad():
        z = emb.detach().float().flatten(0, -2)              # (N, D), fp32
        emb_std = z.std(dim=0).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        denom = max(1, zc.shape[0] - 1)
        # Keep the covariance in fp32 even under bf16-mixed autocast; the rank
        # ratio is sensitive to Frobenius-norm precision.
        if z.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                cov = (zc.transpose(0, 1) @ zc) / denom      # (D, D)
        else:
            cov = (zc.transpose(0, 1) @ zc) / denom          # (D, D)
        tr = torch.diagonal(cov).sum()
        fro2 = cov.pow(2).sum()
        eps = torch.finfo(cov.dtype).tiny
        effective_rank = (tr * tr + eps) / (fro2 + eps)
    return {"emb_std": emb_std, "effective_rank": effective_rank}


def _inverse_margin_diagnostics(inverse_loss, target_act, loss_type, inverse_acc=None):
    """How much room the inverse-dynamics term is actually using (diagnostic).

    The inverse task only resists collapse insofar as it beats the trivial
    solution; once this margin is ~0 the anti-collapse signal has gone slack --
    the mechanism behind environment-dependent MTM collapse (e.g. Reacher, where
    the frameskip-5 action block is underdetermined from the latent endpoints, so
    inverse MSE sits at the mean-action floor while the encoder shrinks). Always
    detached; never added to the training loss. ``inverse_margin`` is headroom
    above the trivial baseline, in natural units per loss type:

    * ``mse`` -- baseline is the mean-action MSE (variance of the action
      targets). ``inverse_margin = baseline - loss``, in action-variance units;
      ~0 means the head does no better than the constant batch-mean action.
    * ``action_nce`` -- baseline is chance discrimination accuracy ``1/N`` over
      the N in-batch candidates. ``inverse_margin = acc - 1/N`` (needs
      ``inverse_acc``), in accuracy units (0..1); ~0 means chance-level
      discrimination. NB: the cross-entropy *loss* floor for this distance-
      softmax InfoNCE is >= log(N) and drifts with the action geometry (the
      negatives are real actions, not encoder outputs), so accuracy-over-chance
      is the clean, bounded canary -- not ``log(N) - loss``.
    """
    with torch.no_grad():
        tgt = target_act.detach()
        if loss_type in {"action_nce", "nce", "discrimination"}:
            n = max(2, tgt.shape[0] * tgt.shape[1])
            baseline = torch.as_tensor(1.0 / n, device=tgt.device, dtype=torch.float32)
            out = {"inverse_baseline": baseline}
            if inverse_acc is not None:
                out["inverse_margin"] = inverse_acc.detach().float() - baseline
            return out
        # mse regression
        tgt = tgt.float()
        mean_act = tgt.mean(dim=(0, 1), keepdim=True)
        baseline = (tgt - mean_act).pow(2).mean()
        margin = baseline - inverse_loss.detach().float()
    return {"inverse_baseline": baseline, "inverse_margin": margin}


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    output.update(_representation_diagnostics(emb))
    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in {"emb_std", "effective_rank"}
    }
    self.log_dict(
        losses_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output


def get_horizon_weights(cfg, horizon, device):
    weights_cfg = cfg.loss.get("horizon_weights", {})
    schedule = weights_cfg.get("schedule", "uniform") if weights_cfg else "uniform"
    steps = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)

    if schedule == "uniform":
        weights = torch.ones_like(steps)
    elif schedule in {"inverse", "reciprocal"}:
        weights = 1.0 / steps
    elif schedule in {"discount", "exponential"}:
        gamma = float(weights_cfg.get("gamma", 0.95))
        weights = gamma ** (steps - 1)
    else:
        raise ValueError(f"unknown horizon weight schedule: {schedule}")

    if weights_cfg and weights_cfg.get("normalize", False):
        weights = weights / weights.mean().clamp_min(1e-8)

    return weights


def mh_lejepa_forward(self, batch, stage, cfg):
    """encode observations and directly predict a future latent trajectory."""

    ctx_len = cfg.wm.history_size
    horizon = cfg.wm.horizon
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    expected_steps = ctx_len + horizon
    if emb.size(1) < expected_steps:
        raise ValueError(
            f"direct_horizon requires at least {expected_steps} steps, got {emb.size(1)}"
        )

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    fut_act = act_emb[:, ctx_len - 1 : ctx_len - 1 + horizon]

    tgt_emb = emb[:, ctx_len : ctx_len + horizon]
    pred_emb = self.model.predict_future(ctx_emb, ctx_act, fut_act)

    horizon_mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
    weights = get_horizon_weights(cfg, horizon, emb.device)

    output["pred_loss"] = (horizon_mse * weights.view(1, -1)).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    for h in range(horizon):
        output[f"horizon_{h + 1}_mse"] = horizon_mse[:, h].mean()

    output.update(_representation_diagnostics(emb))

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k.endswith("_mse") or k in {"emb_std", "effective_rank"}
    }
    self.log_dict(
        metrics_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output

def action_discrimination_loss(pred_act, tgt_act, temperature=0.1):
    """Classify each transition's true action among in-batch action negatives.

    The inverse head still predicts an action vector. Instead of MSE regression,
    we score every candidate action in the batch by negative squared distance to
    the prediction and apply cross-entropy. A collapsed encoder gives the same
    inverse prediction for every transition, so it cannot identify distinct true
    actions except by chance.
    """

    pred = pred_act.flatten(0, 1).float()
    tgt = tgt_act.flatten(0, 1).float()
    if pred.size(0) < 2:
        raise ValueError("action_discrimination_loss needs at least two candidates")
    temp = max(float(temperature), 1e-8)
    logits = -(pred[:, None, :] - tgt[None, :, :]).pow(2).mean(dim=-1) / temp
    labels = torch.arange(pred.size(0), device=pred.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def state_transition_discrimination_loss(
    pred_emb,
    tgt_emb,
    anchor_emb,
    *,
    mode="delta",
    temperature=0.1,
    hard_negatives=0,
):
    """Classify each predicted future state among in-batch future-state negatives.

    Unlike inverse action discrimination, this contrastive target is aligned
    with the planner's final-state matching problem. In ``delta`` mode it
    compares predicted and actual latent changes, which emphasizes the state
    variables that changed under the action instead of static scene/pose terms.
    ``hard_negatives`` keeps the diagonal positive but restricts negatives to
    the currently highest-scoring wrong targets, so the loss focuses on states
    the representation already confuses instead of being dominated by easy
    in-batch negatives.
    """

    pred = pred_emb.float()
    tgt = tgt_emb.float()
    anchor = anchor_emb[:, : pred.size(1)].float()
    mode = str(mode)
    if mode == "delta":
        pred = pred - anchor
        tgt = tgt - anchor
    elif mode != "absolute":
        raise ValueError(f"unknown state_nce mode: {mode!r} (expected 'delta' or 'absolute')")

    pred = F.normalize(pred.flatten(0, 1), dim=-1)
    tgt = F.normalize(tgt.flatten(0, 1), dim=-1)
    if pred.size(0) < 2:
        raise ValueError("state_transition_discrimination_loss needs at least two candidates")

    temp = max(float(temperature), 1e-8)
    logits = pred @ tgt.transpose(0, 1) / temp
    labels = torch.arange(pred.size(0), device=pred.device)
    hard_negatives = int(hard_negatives or 0)
    if hard_negatives > 0 and hard_negatives < pred.size(0) - 1:
        neg_logits = logits.masked_fill(
            torch.eye(logits.size(0), dtype=torch.bool, device=logits.device),
            -torch.inf,
        )
        hard_logits = neg_logits.topk(hard_negatives, dim=1).values
        logits_for_loss = torch.cat([logits.diag().unsqueeze(1), hard_logits], dim=1)
        labels_for_loss = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
    else:
        logits_for_loss = logits
        labels_for_loss = labels
    loss = F.cross_entropy(logits_for_loss, labels_for_loss)
    with torch.no_grad():
        acc = (logits_for_loss.argmax(dim=1) == labels_for_loss).float().mean()
    return loss, acc


def masked_transition_forward(self, batch, stage, cfg):
    """Masked transition modeling: mask one element of (z_t, a_t, z_{t+1}) and
    predict it from the other two.

    Two masked tasks share the LeWM encoder:
      * mask z_{t+1} -> forward dynamics  (z_t, a_t) -> z_{t+1}   [the LeWM predictor]
      * mask a_t     -> inverse dynamics  (z_t, z_{t+1}) -> a_t   [anti-collapse]

    There is no SIGReg / variance / covariance term. Collapse (f(o) = c) drives
    the forward loss to zero but is unattractive because it makes the
    inverse-dynamics task unsolvable: a constant encoder cannot reveal which
    action produced a transition, so distinct states emerge from the requirement
    to predict distinguishable action-conditioned transitions.
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    masked_cfg = cfg.loss.get("masked", {})
    mask_mode = str(masked_cfg.get("mask_mode", "both"))
    inv_weight = float(masked_cfg.get("inverse_weight", 1.0))
    inv_warmup_epochs = int(masked_cfg.get("inverse_warmup_epochs", 0) or 0)
    current_epoch = max(0, int(getattr(self, "current_epoch", 0) or 0))
    if inv_warmup_epochs > 0:
        inv_weight_scale = min(1.0, current_epoch / float(inv_warmup_epochs))
    else:
        inv_weight_scale = 1.0
    effective_inv_weight = inv_weight * inv_weight_scale

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]  # (B, T, A_emb)
    action = batch["action"]  # (B, T, A) the normalized raw action fed to the encoder

    # --- mask z_{t+1}: forward dynamics, identical to LeWM's predictor path ---
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)
    tgt_emb = emb[:, n_preds : n_preds + pred_emb.size(1)]
    forward_loss = (pred_emb - tgt_emb).pow(2).mean()

    # --- mask a_t: inverse dynamics over every transition z_t -> z_{t+1} ---
    # action[:, t] is the action carrying state t -> t+1, so the last action is
    # dropped (its target state is outside the window).
    z_t = emb[:, :-1]
    z_next = emb[:, 1:]
    tgt_act = action[:, : emb.size(1) - 1]
    pred_act = self.model.predict_action(z_t, z_next)
    inverse_loss_type = str(masked_cfg.get("inverse_loss_type", "mse"))
    if inverse_loss_type == "mse":
        inverse_loss = (pred_act - tgt_act).pow(2).mean()
        inverse_acc = None
    elif inverse_loss_type in {"action_nce", "nce", "discrimination"}:
        inverse_loss, inverse_acc = action_discrimination_loss(
            pred_act,
            tgt_act,
            temperature=float(masked_cfg.get("action_nce_temperature", 0.1)),
        )
    else:
        raise ValueError(
            f"unknown inverse_loss_type: {inverse_loss_type!r} "
            "(expected 'mse' or 'action_nce')"
        )

    if mask_mode == "both":
        # Expectation of the random mask: train both heads every step (lower
        # variance, the recommended default for a clean signal).
        loss = forward_loss + effective_inv_weight * inverse_loss
    elif mask_mode == "random":
        # Literal masked-modeling: pick one task per batch. torch.where routes the
        # gradient through only the chosen branch while both metrics stay logged.
        forward_prob = float(masked_cfg.get("forward_prob", 0.5))
        pick_forward = torch.rand((), device=emb.device) < forward_prob
        loss = torch.where(pick_forward, forward_loss, effective_inv_weight * inverse_loss)
    else:
        raise ValueError(f"unknown mask_mode: {mask_mode!r} (expected 'both' or 'random')")

    state_nce_cfg = masked_cfg.get("state_nce", {})
    state_nce_weight = float(state_nce_cfg.get("weight", 0.0))
    if state_nce_weight > 0.0:
        state_nce_loss, state_nce_acc = state_transition_discrimination_loss(
            pred_emb,
            tgt_emb,
            ctx_emb,
            mode=state_nce_cfg.get("mode", "delta"),
            temperature=float(state_nce_cfg.get("temperature", 0.1)),
            hard_negatives=int(state_nce_cfg.get("hard_negatives", 0) or 0),
        )
        loss = loss + state_nce_weight * state_nce_loss
        output["state_nce_loss"] = state_nce_loss
        output["state_nce_acc"] = state_nce_acc
        output["state_nce_weight"] = torch.as_tensor(
            state_nce_weight, device=emb.device
        )

    sigreg_weight = float(cfg.loss.get("sigreg", {}).get("weight", 0.0))
    if sigreg_weight > 0.0:
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = loss + sigreg_weight * sigreg_loss
        output["sigreg_loss"] = sigreg_loss

    # Optional per-dim variance floor (VICReg variance term). Pushes lazy dims live
    # without constraining distribution shape. Zero cost when all dims are above gamma.
    varreg_cfg = cfg.loss.get("variance_reg", {})
    varreg_weight = float(varreg_cfg.get("weight", 0.0))
    if varreg_weight > 0.0:
        varreg_gamma = float(varreg_cfg.get("gamma", 0.1))
        variance_reg_loss = VarianceReg(gamma=varreg_gamma)(emb)
        loss = loss + varreg_weight * variance_reg_loss
        output["variance_reg_loss"] = variance_reg_loss

    # Optional reconstruction auxiliary. Decoding the planning latent back to the
    # frame forces it to retain observation detail (e.g. block orientation) that
    # the dynamics/inverse objective discards. Grounded target = the frame itself.
    recon_cfg = cfg.loss.get("reconstruction", {})
    recon_weight = float(recon_cfg.get("weight", 0.0))
    if recon_weight > 0.0:
        out_size = int(recon_cfg.get("out_size", 64))
        recon = self.model.reconstruct(emb)
        recon_target = self.model.recon_target(batch, out_size)
        recon_loss = (recon - recon_target).pow(2).mean()
        loss = loss + recon_weight * recon_loss
        output["recon_loss"] = recon_loss

    output["forward_loss"] = forward_loss
    output["inverse_loss"] = inverse_loss
    output.update(_inverse_margin_diagnostics(inverse_loss, tgt_act, inverse_loss_type, inverse_acc))
    if inverse_acc is not None:
        output["inverse_acc"] = inverse_acc
    output["inverse_weight_effective"] = torch.as_tensor(
        effective_inv_weight, device=emb.device
    )
    output["loss"] = loss
    # SIGReg-free collapse canaries (gradient-free): emb_std -> 0 on a constant
    # encoder; effective_rank -> 1 when the latent loses rank into a low-D
    # subspace even while emb_std still looks healthy. Paired with inverse_margin
    # (floor minus achieved inverse loss): margin -> 0 is the slack-anti-collapse
    # signal behind Reacher-style collapse. See _representation_diagnostics.
    output.update(_representation_diagnostics(emb))

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in {"emb_std", "effective_rank", "inverse_margin", "inverse_baseline", "inverse_weight_effective", "inverse_acc", "state_nce_acc", "state_nce_weight"}
    }
    self.log_dict(
        metrics_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output


def masked_horizon_forward(self, batch, stage, cfg):
    """Multi-horizon masked transition modeling (the "beat PushT" variant).

    Same masked principle, but the forward task predicts the full H-step latent
    *trajectory* via the direct-horizon FutureQueryPredictor instead of a single
    step, while the inverse-dynamics term (anti-collapse) runs over every
    transition. No SIGReg.

    Motivation: on tasks where the goal variable is a slowly / indirectly
    controlled object (e.g. the PushT block, which moves only on contact),
    single-step prediction + single-step inverse are dominated by the densely
    action-coupled agent and under-encode the object. Predicting H steps ahead
    forces the encoder to represent the object well enough to roll its
    contact-driven future forward, which is exactly what goal-matching planning
    needs. Planning uses the direct-horizon rollout, identical to LeWM-Direct-H.
    """

    ctx_len = cfg.wm.history_size
    horizon = cfg.wm.horizon
    masked_cfg = cfg.loss.get("masked", {})
    inv_weight = float(masked_cfg.get("inverse_weight", 1.0))
    inv_warmup_epochs = int(masked_cfg.get("inverse_warmup_epochs", 0) or 0)
    current_epoch = max(0, int(getattr(self, "current_epoch", 0) or 0))
    if inv_warmup_epochs > 0:
        inv_weight_scale = min(1.0, current_epoch / float(inv_warmup_epochs))
    else:
        inv_weight_scale = 1.0
    effective_inv_weight = inv_weight * inv_weight_scale

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]  # (B, C+H, D)
    act_emb = output["act_emb"]
    action = batch["action"]

    expected_steps = ctx_len + horizon
    if emb.size(1) < expected_steps:
        raise ValueError(
            f"masked_horizon requires at least {expected_steps} steps, got {emb.size(1)}"
        )

    # --- mask the future trajectory: H-step forward prediction ---
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    fut_act = act_emb[:, ctx_len - 1 : ctx_len - 1 + horizon]
    tgt_emb = emb[:, ctx_len : ctx_len + horizon]
    pred_emb = self.model.predict_future(ctx_emb, ctx_act, fut_act)
    horizon_mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
    weights = get_horizon_weights(cfg, horizon, emb.device)
    forward_loss = (horizon_mse * weights.view(1, -1)).mean()

    # --- mask the action: inverse dynamics over every transition (anti-collapse) ---
    z_t = emb[:, :-1]
    z_next = emb[:, 1:]
    tgt_act = action[:, : emb.size(1) - 1]
    pred_act = self.model.predict_action(z_t, z_next)
    inverse_loss = (pred_act - tgt_act).pow(2).mean()
    output.update(_inverse_margin_diagnostics(inverse_loss, tgt_act, "mse"))

    loss = forward_loss + effective_inv_weight * inverse_loss

    output["forward_loss"] = forward_loss
    output["inverse_loss"] = inverse_loss
    output["inverse_weight_effective"] = torch.as_tensor(
        effective_inv_weight, device=emb.device
    )
    output["loss"] = loss
    output.update(_representation_diagnostics(emb))
    for h in range(horizon):
        output[f"horizon_{h + 1}_mse"] = horizon_mse[:, h].mean()

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k.endswith("_mse") or k in {"emb_std", "effective_rank", "inverse_margin", "inverse_baseline", "inverse_weight_effective"}
    }
    self.log_dict(
        metrics_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output


def byol_wm_forward(self, batch, stage, cfg):
    """BYOL-style temporal world model with stochastic horizon.

    Online encoder produces z_t. An EMA (momentum) copy of the encoder
    produces the target z_{t+k} — no gradient through the target path.
    The FutureQueryPredictor maps (z_t, a_{t:t+k}) → z_{t+k}_pred and is
    trained to match the EMA target via MSE.

    Anti-collapse mechanism: EMA asymmetry. If the online encoder collapses
    (z = const), the EMA target lags behind and continues to produce diverse
    representations for different observations, keeping the prediction loss
    non-trivially positive. No SIGReg, no VICReg, no inverse dynamics.

    Stochastic horizon k ~ Uniform[min_k, max_k] forces the encoder to
    represent state variables that only matter for multi-step prediction
    (e.g. block orientation on PushT, joint angles on Reacher) — variables
    that a single-step objective can ignore because they barely change per
    step but accumulate meaningful differences over k steps.
    """
    ctx_len = cfg.wm.history_size
    byol_cfg = cfg.loss.get("byol", {})
    momentum = float(byol_cfg.get("momentum", 0.996))
    min_k = int(byol_cfg.get("min_k", 1))
    max_k = int(byol_cfg.get("max_k", cfg.wm.horizon))

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Online encoding — gradient flows through this path
    output = self.model.encode(batch)
    emb = output["emb"]       # (B, T, D)
    act_emb = output["act_emb"]

    # EMA target encoding — no gradient
    target_output = self.model.encode_target(batch)
    target_emb = target_output["target_emb"]  # (B, T, D)

    # Sample one horizon per batch step
    k = int(torch.randint(min_k, max_k + 1, ()).item())

    # Predict z_{ctx+k} from online context via FutureQueryPredictor
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    fut_act = act_emb[:, ctx_len - 1 : ctx_len - 1 + k]
    pred_emb = self.model.predict_future(ctx_emb, ctx_act, fut_act)  # (B, k, D)

    pred_at_k = pred_emb[:, k - 1]                        # (B, D)
    tgt_at_k  = target_emb[:, ctx_len + k - 1].detach()   # (B, D)
    forward_loss = F.mse_loss(pred_at_k, tgt_at_k)

    output["forward_loss"] = forward_loss
    output["loss"] = forward_loss
    output.update(_representation_diagnostics(emb))
    output["horizon_k"] = torch.as_tensor(float(k), device=emb.device)

    # EMA update after each training step
    if stage == "fit":
        self.model.ema_update(momentum)

    metrics_dict = {
        f"{stage}/{key}": v.detach()
        for key, v in output.items()
        if "loss" in key or key in {"emb_std", "effective_rank", "horizon_k"}
    }
    self.log_dict(
        metrics_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output


def _endpoint_inverse_losses(
    model,
    emb,
    action,
    min_gap,
    max_gap,
    *,
    loss_type="mse",
    temperature=0.1,
):
    """Compute inverse-dynamics loss for endpoint pairs at each gap in [min_gap, max_gap].

    For each gap k, predicts a_t from (z_t, z_{t+k}, e_k) across all valid t positions.
    Returns list of (gap, loss, acc, candidate_count) tuples.
    """
    losses = []
    loss_type = str(loss_type)
    for gap in range(min_gap, max_gap + 1):
        if emb.size(1) <= gap:
            raise ValueError(
                f"cannot form inverse endpoint pairs with gap={gap} "
                f"from only {emb.size(1)} encoded steps"
            )
        z_t = emb[:, : emb.size(1) - gap]
        z_future = emb[:, gap:]
        tgt_act = action[:, : emb.size(1) - gap]
        pred_act = model.predict_action(z_t, z_future, horizon=gap)
        if loss_type == "mse":
            loss = (pred_act - tgt_act).pow(2).mean()
            acc = None
        elif loss_type in {"action_nce", "nce", "discrimination"}:
            loss, acc = action_discrimination_loss(
                pred_act,
                tgt_act,
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"unknown endpoint inverse_loss_type: {loss_type!r} "
                "(expected 'mse' or 'action_nce')"
            )
        losses.append((gap, loss, acc, tgt_act.flatten(0, 1).size(0)))
    return losses


def _endpoint_inverse_diagnostics(inverse_loss, gap_losses, target_act, loss_type):
    """Diagnostics for endpoint inverse losses, including endpoint Action-NCE."""
    loss_type = str(loss_type)
    if loss_type in {"action_nce", "nce", "discrimination"}:
        accs = [acc for _, _, acc, _ in gap_losses if acc is not None]
        counts = [count for _, _, _, count in gap_losses if count is not None]
        if accs:
            inverse_acc = torch.stack(accs).mean()
            baseline = torch.as_tensor(
                sum(1.0 / float(count) for count in counts) / len(counts),
                device=inverse_loss.device,
            )
            return {
                "inverse_acc": inverse_acc.detach(),
                "inverse_baseline": baseline,
                "inverse_margin": inverse_acc.detach() - baseline,
            }
    return _inverse_margin_diagnostics(inverse_loss, target_act, loss_type)


def masked_endpoint_inverse_forward(self, batch, stage, cfg):
    """Standard one-step MTM forward plus multi-step endpoint inverse dynamics.

    This isolates the action-discrimination hypothesis: keep the planner-facing
    AR forward task identical to `masked_transition`, but replace the inverse
    head with (z_t, z_{t+k}, e_k) -> a_t for k in [min_gap, max_gap].
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    masked_cfg = cfg.loss.get("masked", {})
    inv_weight = float(masked_cfg.get("inverse_weight", 1.0))
    inv_warmup_epochs = int(masked_cfg.get("inverse_warmup_epochs", 0) or 0)
    current_epoch = max(0, int(getattr(self, "current_epoch", 0) or 0))
    inv_weight_scale = (
        min(1.0, current_epoch / float(inv_warmup_epochs))
        if inv_warmup_epochs > 0
        else 1.0
    )
    effective_inv_weight = inv_weight * inv_weight_scale

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    action = batch["action"]

    # Standard one-step AR forward loss, unchanged from masked_transition.
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)
    tgt_emb = emb[:, n_preds : n_preds + pred_emb.size(1)]
    forward_loss = (pred_emb - tgt_emb).pow(2).mean()

    min_gap = int(masked_cfg.get("inverse_min_horizon", 1))
    max_gap = min(
        int(masked_cfg.get("inverse_max_horizon", cfg.wm.get("horizon", 1))),
        int(cfg.wm.get("horizon", 1)),
        emb.size(1) - 1,
    )
    if min_gap > max_gap:
        raise ValueError(
            f"inverse_min_horizon={min_gap} exceeds max valid gap {max_gap}"
        )

    train_mode = str(masked_cfg.get("inverse_train_mode", "random"))
    eval_mode = str(masked_cfg.get("inverse_eval_mode", "all"))
    inverse_mode = train_mode if stage == "fit" else eval_mode
    inverse_loss_type = str(masked_cfg.get("inverse_loss_type", "mse"))
    action_nce_temperature = float(masked_cfg.get("action_nce_temperature", 0.1))
    if inverse_mode == "random":
        gap = int(torch.randint(min_gap, max_gap + 1, (1,), device=emb.device).item())
        gap_losses = _endpoint_inverse_losses(
            self.model,
            emb,
            action,
            gap,
            gap,
            loss_type=inverse_loss_type,
            temperature=action_nce_temperature,
        )
        inverse_loss = gap_losses[0][1]
        output["inverse_horizon"] = torch.as_tensor(float(gap), device=emb.device)
    elif inverse_mode == "all":
        gap_losses = _endpoint_inverse_losses(
            self.model,
            emb,
            action,
            min_gap,
            max_gap,
            loss_type=inverse_loss_type,
            temperature=action_nce_temperature,
        )
        inverse_loss = torch.stack([loss for _, loss, _, _ in gap_losses]).mean()
    else:
        raise ValueError(
            f"unknown inverse mode: {inverse_mode!r} (expected 'random' or 'all')"
        )

    output.update(
        _endpoint_inverse_diagnostics(
            inverse_loss,
            gap_losses,
            action,
            inverse_loss_type,
        )
    )
    loss = forward_loss + effective_inv_weight * inverse_loss

    output["forward_loss"] = forward_loss
    output["inverse_loss"] = inverse_loss
    output["inverse_weight_effective"] = torch.as_tensor(
        effective_inv_weight, device=emb.device
    )
    output["loss"] = loss
    output.update(_representation_diagnostics(emb))
    for gap, gap_loss, _, _ in gap_losses:
        output[f"inverse_h{gap}_loss"] = gap_loss

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in {"emb_std", "effective_rank", "inverse_margin", "inverse_baseline", "inverse_weight_effective", "inverse_horizon", "inverse_acc"}
    }
    self.log_dict(
        metrics_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output


def masked_multi_step_forward(self, batch, stage, cfg):
    """Multi-scale masked trajectory modeling (ms-mtm).

    Two complementary objectives:
      1. H-step forward prediction: predict z_{ctx+1}...z_{ctx+H} from context.
      2. Endpoint inverse dynamics: predict a_t from (z_t, z_{t+k}, e_k) at
         random gap k ~ Uniform{min_gap, max_gap}. At large k, recovering the
         first action requires encoding slow state variables (block orientation,
         joint configuration) that 1-step inverse dynamics can ignore.

    No SIGReg. Anti-collapse is task-derived from both objectives.
    """
    ctx_len = cfg.wm.history_size
    horizon = cfg.wm.horizon
    masked_cfg = cfg.loss.get("masked", {})
    inv_weight = float(masked_cfg.get("inverse_weight", 1.0))
    inv_warmup_epochs = int(masked_cfg.get("inverse_warmup_epochs", 0) or 0)
    current_epoch = max(0, int(getattr(self, "current_epoch", 0) or 0))
    inv_weight_scale = min(1.0, current_epoch / float(inv_warmup_epochs)) if inv_warmup_epochs > 0 else 1.0
    effective_inv_weight = inv_weight * inv_weight_scale

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]        # (B, ctx+H, D)
    act_emb = output["act_emb"]
    action = batch["action"]

    expected_steps = ctx_len + horizon
    if emb.size(1) < expected_steps:
        raise ValueError(
            f"masked_multi_step requires at least {expected_steps} steps, got {emb.size(1)}"
        )

    # --- H-step forward prediction ---
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    fut_act = act_emb[:, ctx_len - 1 : ctx_len - 1 + horizon]
    tgt_emb = emb[:, ctx_len : ctx_len + horizon]
    pred_emb = self.model.predict_future(ctx_emb, ctx_act, fut_act)
    horizon_mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)  # (B, H)
    weights = get_horizon_weights(cfg, horizon, emb.device)
    forward_loss = (horizon_mse * weights.view(1, -1)).mean()

    # --- Endpoint inverse dynamics at random gap k ---
    min_gap = int(masked_cfg.get("inverse_min_horizon", 1))
    max_gap = min(int(masked_cfg.get("inverse_max_horizon", horizon)), horizon, emb.size(1) - 1)

    train_mode = str(masked_cfg.get("inverse_train_mode", "random"))
    eval_mode = str(masked_cfg.get("inverse_eval_mode", "all"))
    inverse_mode = train_mode if stage == "fit" else eval_mode
    inverse_loss_type = str(masked_cfg.get("inverse_loss_type", "mse"))
    action_nce_temperature = float(masked_cfg.get("action_nce_temperature", 0.1))

    if inverse_mode == "random":
        k = int(torch.randint(min_gap, max_gap + 1, (1,), device=emb.device).item())
        gap_losses = _endpoint_inverse_losses(
            self.model,
            emb,
            action,
            k,
            k,
            loss_type=inverse_loss_type,
            temperature=action_nce_temperature,
        )
        inverse_loss = gap_losses[0][1]
        output["inverse_horizon"] = torch.as_tensor(float(k), device=emb.device)
    else:  # "all"
        gap_losses = _endpoint_inverse_losses(
            self.model,
            emb,
            action,
            min_gap,
            max_gap,
            loss_type=inverse_loss_type,
            temperature=action_nce_temperature,
        )
        inverse_loss = torch.stack([loss for _, loss, _, _ in gap_losses]).mean()

    output.update(
        _endpoint_inverse_diagnostics(
            inverse_loss,
            gap_losses,
            action,
            inverse_loss_type,
        )
    )
    loss = forward_loss + effective_inv_weight * inverse_loss

    output["forward_loss"] = forward_loss
    output["inverse_loss"] = inverse_loss
    output["inverse_weight_effective"] = torch.as_tensor(effective_inv_weight, device=emb.device)
    output["loss"] = loss
    output.update(_representation_diagnostics(emb))
    for h in range(horizon):
        output[f"horizon_{h + 1}_mse"] = horizon_mse[:, h].mean()
    for gap, gap_loss, _, _ in gap_losses:
        output[f"inverse_h{gap}_loss"] = gap_loss

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k.endswith("_mse") or k in {"emb_std", "effective_rank", "inverse_margin", "inverse_baseline", "inverse_weight_effective", "inverse_horizon", "inverse_acc"}
    }
    self.log_dict(metrics_dict, on_step=(stage == "fit"), on_epoch=(stage != "fit"), sync_dist=True)
    return output


def ac_cpc_forward(self, batch, stage, cfg):
    """Action-conditioned contrastive predictive coding (AC-CPC).

    SIGReg-free, MSE-free anti-collapse. The forward predictor (z_t, a_t) -> z_hat
    is trained with InfoNCE: among a pool of candidate true futures (this
    transition's real z_{t+1} as the positive, plus the real z_{t+1} of *other*
    trajectories in the batch as negatives), the prediction must identify its own.

    Why this avoids collapse without SIGReg or an inverse head: a constant encoder
    maps every future to the same point, so the predictor cannot tell the true
    future from a negative and InfoNCE is pinned at chance (loss ~ log N). The only
    way to drive the loss down is to make futures *distinguishable*, i.e. encode
    whatever varies across trajectories (for PushT, the block pose). This also
    breaks the MSE self-prediction circularity that hurts masked/LeWM on PushT:
    MSE forward regresses the encoder onto its *own* latent, so a latent that
    ignores the block still earns a low target error; here prediction is graded
    against *other* trajectories' real futures, so ignoring the block is punished
    directly by mis-identification.

    Latents are unit-normalized in the encoder/predictor (the config sets
    normalize_latents=True), so logits are cosine similarities / temperature and
    the planner's MSE criterion matches the trained cosine geometry.
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    cpc_cfg = cfg.loss.get("cpc", {})
    temperature = float(cpc_cfg.get("temperature", 0.1))

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D), unit-norm
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]                        # (B, P, D) true futures
    pred_emb = self.model.predict(ctx_emb, ctx_act)   # (B, P, D) predicted futures

    B, P, D = pred_emb.shape
    pred_flat = pred_emb.reshape(B * P, D)
    tgt_flat = tgt_emb.reshape(B * P, D)

    # logits[i, j] = <pred_i, tgt_j> / temperature (cosine sim; both unit-norm).
    logits = (pred_flat @ tgt_flat.t()) / temperature  # (N, N), N = B*P
    N = B * P
    labels = torch.arange(N, device=emb.device)

    # Negatives are futures from *other* trajectories only: mask out same-trajectory
    # non-positive candidates so temporally adjacent frames of the same trajectory
    # (legitimately similar futures) are not penalized as false negatives.
    traj_id = torch.arange(B, device=emb.device).repeat_interleave(P)  # (N,)
    same_traj = traj_id.unsqueeze(0) == traj_id.unsqueeze(1)           # (N, N)
    diagonal = torch.eye(N, dtype=torch.bool, device=emb.device)
    logits = logits.masked_fill(same_traj & ~diagonal, float("-inf"))

    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        cpc_acc = (logits.argmax(dim=1) == labels).float().mean()

    output["cpc_loss"] = loss
    output["loss"] = loss
    output["cpc_acc"] = cpc_acc
    output["cpc_temperature"] = torch.as_tensor(temperature, device=emb.device)
    # SIGReg-free collapse monitor: mean per-dim std of the latent. With unit-norm
    # latents this stays ~1/sqrt(D) unless the encoder collapses to a point.
    output.update(_representation_diagnostics(emb))

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in {"emb_std", "effective_rank", "cpc_acc", "cpc_temperature"}
    }
    self.log_dict(
        metrics_dict,
        on_step=(stage == "fit"),
        on_epoch=(stage != "fit"),
        sync_dist=True,
    )
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    configure_torch_runtime(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    use_gpu_image_preprocess = bool(cfg.runtime.get("gpu_image_preprocess", False))
    transforms = []
    if not use_gpu_image_preprocess:
        transforms.append(
            get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
        )
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    loader_kwargs = make_loader_kwargs(cfg.loader)
    train = torch.utils.data.DataLoader(
        train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
    )
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    prediction_mode = cfg.wm.get("prediction_mode", "autoregressive")

    predictor = None
    future_predictor = None
    inverse_predictor = None
    target_encoder = None
    target_projector = None
    decoder = None
    if prediction_mode in (
        "autoregressive",
        "masked_transition",
        "masked_endpoint_inverse",
        "ac_cpc",
    ):
        # masked_transition and ac_cpc reuse the LeWM autoregressive predictor
        # verbatim for the single-step forward task, so the planner is unchanged;
        # only the training objective (InfoNCE vs MSE+SIGReg) differs.
        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        if prediction_mode == "masked_transition":
            inverse_predictor = InverseDynamics(
                latent_dim=embed_dim,
                action_dim=effective_act_dim,
                **cfg.inverse,
            )
        elif prediction_mode == "masked_endpoint_inverse":
            inverse_predictor = HorizonInverseDynamics(
                latent_dim=embed_dim,
                action_dim=effective_act_dim,
                max_horizon=cfg.wm.horizon,
                **cfg.inverse,
            )
    elif prediction_mode in ("direct_horizon", "masked_horizon", "masked_multi_step"):
        future_predictor = FutureQueryPredictor(
            num_context=cfg.wm.history_size,
            horizon=cfg.wm.horizon,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        if prediction_mode == "masked_horizon":
            inverse_predictor = InverseDynamics(
                latent_dim=embed_dim,
                action_dim=effective_act_dim,
                **cfg.inverse,
            )
        elif prediction_mode == "masked_multi_step":
            inverse_predictor = HorizonInverseDynamics(
                latent_dim=embed_dim,
                action_dim=effective_act_dim,
                max_horizon=cfg.wm.horizon,
                **cfg.inverse,
            )
    elif prediction_mode == "byol_wm":
        # BYOL-style temporal prediction: FutureQueryPredictor for online path,
        # EMA copies of encoder + projector for target path. No inverse head.
        future_predictor = FutureQueryPredictor(
            num_context=cfg.wm.history_size,
            horizon=cfg.wm.horizon,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
    else:
        raise ValueError(f"unknown prediction_mode: {prediction_mode}")

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # BYOL target encoder: EMA copy of encoder + projector, no gradient.
    if prediction_mode == "byol_wm":
        target_encoder = copy.deepcopy(encoder)
        target_projector = copy.deepcopy(projector)
        for p in list(target_encoder.parameters()) + list(target_projector.parameters()):
            p.requires_grad_(False)

    # Reconstruction decoder (anti-collapse auxiliary), built only when enabled.
    recon_weight = float(cfg.loss.get("reconstruction", {}).get("weight", 0.0))
    if recon_weight > 0.0:
        recon_out_size = int(cfg.loss.get("reconstruction", {}).get("out_size", 64))
        decoder = ConvDecoder(latent_dim=embed_dim, out_size=recon_out_size)

    # masked_transition plans with the autoregressive forward predictor; only its
    # training objective differs, so it shares the AR rollout path.
    rollout_mode = (
        "direct_horizon"
        if prediction_mode in ("direct_horizon", "masked_horizon", "byol_wm", "masked_multi_step")
        else "autoregressive"
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        future_predictor=future_predictor,
        inverse_predictor=inverse_predictor,
        target_encoder=target_encoder,
        target_projector=target_projector,
        decoder=decoder,
        rollout_mode=rollout_mode,
        preprocess_pixels=use_gpu_image_preprocess,
        image_size=cfg.img_size if use_gpu_image_preprocess else None,
        # AC-CPC trains a cosine-InfoNCE objective; unit-normalizing the whole
        # latent pipeline makes the planner's MSE criterion agree with it.
        normalize_latents=(prediction_mode == "ac_cpc"),
    )

    scheduler_cfg, scheduler_interval, scheduler_frequency, scheduler_monitor = (
        build_scheduler_section(cfg)
    )
    optimizer_entry = {
        "modules": 'model',
        "optimizer": dict(cfg.optimizer),
        "scheduler": scheduler_cfg,
        "interval": scheduler_interval,
        "frequency": scheduler_frequency,
    }
    if scheduler_monitor:
        optimizer_entry["monitor"] = scheduler_monitor

    optimizers = {
        'model_opt': {
            **optimizer_entry,
        },
    }

    forward_fns = {
        "autoregressive": lejepa_forward,
        "direct_horizon": mh_lejepa_forward,
        "masked_transition": masked_transition_forward,
        "masked_endpoint_inverse": masked_endpoint_inverse_forward,
        "masked_horizon": masked_horizon_forward,
        "masked_multi_step": masked_multi_step_forward,
        "ac_cpc": ac_cpc_forward,
        "byol_wm": byol_wm_forward,
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(forward_fns[prediction_mode], cfg=cfg),
        optim=optimizers,
    )
    attach_gradient_norm_logger(
        world_model, int(cfg.logging.get("grad_norm_every_n_steps", 0) or 0)
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    resume_path = resolve_resume_checkpoint(
        run_dir, cfg.output_model_name, cfg.get("resume", {})
    )
    check_resume_config_compatibility(run_dir, cfg, resume_path)
    run_metadata = {
        "git": get_git_metadata(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "MODAL_TASK_ID",
                "MODAL_FUNCTION_CALL_ID",
                "MODAL_CLOUD_PROVIDER",
                "STABLEWM_HOME",
            )
            if os.environ.get(key)
        },
    }

    logger = build_wandb_logger(cfg, run_dir, run_metadata)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), f)
    dump_run_metadata(run_dir, cfg, resume_path, run_metadata)

    log_every_n_steps = int(cfg.logging.get("log_every_n_steps", 50) or 50)
    progress_enabled = bool(cfg.logging.get("progress_bar", True))

    callbacks = [
        JsonlMetricsCallback(
            dirpath=run_dir,
            log_every_n_steps=log_every_n_steps,
            log_to_trainer=True,
        ),
        RNGStateCallback(),
    ]

    if progress_enabled:
        callbacks.append(
            ProgressBarCallback(
                refresh_every_n_steps=int(
                    cfg.logging.get("progress_refresh_steps", log_every_n_steps)
                    or log_every_n_steps
                ),
            )
        )

    if bool(cfg.get("dump_object", True)):
        callbacks.append(
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=int(cfg.checkpoint.get("object_epoch_interval", 1) or 1),
            )
        )

    stop_after_epoch = cfg.runtime.get("stop_after_epoch", None)
    if stop_after_epoch is not None:
        callbacks.append(StopAfterEpochCallback(int(stop_after_epoch)))

    checkpoint_cfg = cfg.get("checkpoint", {})
    if bool(checkpoint_cfg.get("enabled", True)):
        callbacks.append(
            ResumableCheckpointCallback(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                step_interval=int(checkpoint_cfg.get("step_interval", 0) or 0),
                time_interval_seconds=float(
                    checkpoint_cfg.get("time_interval_seconds", 0) or 0
                ),
                monitor=str(checkpoint_cfg.get("monitor", "validate/loss")),
                mode=str(checkpoint_cfg.get("mode", "min")),
                save_best=bool(checkpoint_cfg.get("save_best", True)),
                keep_last_n=int(checkpoint_cfg.get("keep_last_n", 0) or 0),
                save_on_train_epoch_end=bool(
                    checkpoint_cfg.get("save_on_train_epoch_end", True)
                ),
            )
        )

    early_cfg = cfg.get("early_stopping", {})
    if bool(early_cfg.get("enabled", False)):
        callbacks.append(
            MinEpochEarlyStopping(
                monitor=str(early_cfg.get("monitor", "validate/loss")),
                mode=str(early_cfg.get("mode", "min")),
                patience=int(early_cfg.get("patience", 10)),
                min_delta=float(early_cfg.get("min_delta", 0.0)),
                min_epochs=int(early_cfg.get("min_epochs", 0)),
                verbose=True,
                strict=bool(early_cfg.get("strict", False)),
            )
        )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault("num_sanity_val_steps", 1)
    if progress_enabled:
        # Our ProgressBarCallback replaces Lightning's TQDM bar, which renders
        # poorly in captured (non-TTY) logs such as Modal.
        trainer_kwargs.setdefault("enable_progress_bar", False)
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False,
    )

    print(f"Run directory: {run_dir}", flush=True)
    if resume_path is not None:
        print(f"Resuming training from checkpoint: {resume_path}", flush=True)
    trainer.fit(
        world_model,
        datamodule=data_module,
        ckpt_path=str(resume_path) if resume_path is not None else None,
        weights_only=False if resume_path is not None else None,
    )
    return


if __name__ == "__main__":
    run()
