"""Tests for AC-CPC (action-conditioned contrastive predictive coding).

Covers the InfoNCE forward objective (ac_cpc_forward), the unit-sphere
normalization that keeps training (cosine) and planning (MSE) consistent, the
same-trajectory negative masking, and the central anti-collapse claim: a
collapsed encoder is pinned at the chance InfoNCE loss and cannot drive it down.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from omegaconf import OmegaConf
from torch import nn

from jepa import JEPA
from module import ARPredictor
from train import ac_cpc_forward

ROOT = Path(__file__).resolve().parents[1]


class ConstantEncoder(nn.Module):
    """Collapsed encoder: every observation maps to the same vector."""

    def __init__(self, embed_dim, value=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.value = value

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        batch = pixels.size(0)
        cls = torch.full((batch, 1, self.embed_dim), self.value, device=pixels.device)
        return SimpleNamespace(last_hidden_state=cls)


class LearnableEncoder(nn.Module):
    """Tiny encoder whose CLS token depends on the input pixels (has grad)."""

    def __init__(self, embed_dim, in_features):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        cls = self.proj(pixels.flatten(1)).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=cls)


def _make_jepa(encoder, embed_dim, normalize_latents=True):
    predictor = ARPredictor(
        num_frames=4,
        depth=1,
        heads=2,
        mlp_dim=32,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        dim_head=4,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=nn.Identity(),  # action_dim == embed_dim so act_emb == action
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        rollout_mode="autoregressive",
        normalize_latents=normalize_latents,
    )


def _cfg(temperature=0.1):
    return OmegaConf.create(
        {
            "wm": {"history_size": 3, "num_preds": 1},
            "loss": {"cpc": {"temperature": temperature}},
        }
    )


def _batch(batch=4, steps=4, embed_dim=6, channels=3, hw=2):
    return {
        "pixels": torch.randn(batch, steps, channels, hw, hw),
        "action": torch.randn(batch, steps, embed_dim),
    }


# --------------------------------------------------------------------------- #
# normalize_latents: the whole pipeline lives on the unit sphere
# --------------------------------------------------------------------------- #


def test_normalize_latents_puts_encoder_and_predictor_on_unit_sphere():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, 3 * 2 * 2), embed_dim, normalize_latents=True)
    batch = _batch(embed_dim=embed_dim)

    enc = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    emb_norms = enc["emb"].flatten(0, 1).norm(dim=-1)
    assert torch.allclose(emb_norms, torch.ones_like(emb_norms), atol=1e-5)

    pred = model.predict(enc["emb"][:, :3], enc["act_emb"][:, :3])
    pred_norms = pred.flatten(0, 1).norm(dim=-1)
    assert torch.allclose(pred_norms, torch.ones_like(pred_norms), atol=1e-5)


def test_encode_predict_loadable_without_normalize_latents_attr():
    """Backward-compat: checkpoints pickled before normalize_latents existed have
    no such attribute. encode/predict/predict_future must fall back to no-op (via
    getattr default) rather than raise AttributeError, exactly like the
    preprocess_pixels shim."""
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, 3 * 2 * 2), embed_dim, normalize_latents=True)
    # simulate an old pickle: the attribute simply does not exist
    del model.normalize_latents
    assert not hasattr(model, "normalize_latents")

    batch = _batch(embed_dim=embed_dim)
    enc = model.encode({"pixels": batch["pixels"], "action": batch["action"]})  # must not raise
    pred = model.predict(enc["emb"][:, :3], enc["act_emb"][:, :3])              # must not raise
    assert torch.isfinite(enc["emb"]).all() and torch.isfinite(pred).all()
    # with the attribute absent, normalization is skipped -> norms are NOT forced to 1
    norms = enc["emb"].flatten(0, 1).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_normalize_latents_off_by_default_leaves_norms_free():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, 3 * 2 * 2), embed_dim, normalize_latents=False)
    enc = model.encode({"pixels": _batch(embed_dim=embed_dim)["pixels"]})
    norms = enc["emb"].flatten(0, 1).norm(dim=-1)
    # generic linear features are essentially never all exactly unit-norm
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


# --------------------------------------------------------------------------- #
# ac_cpc_forward: shapes, finiteness, backprop, metrics
# --------------------------------------------------------------------------- #


def test_ac_cpc_forward_returns_finite_loss_and_backprops():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, 3 * 2 * 2), embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = ac_cpc_forward(harness, _batch(embed_dim=embed_dim), "fit", _cfg())

    for key in ("cpc_loss", "loss", "cpc_acc", "cpc_temperature", "emb_std"):
        assert key in out and torch.isfinite(out[key]).all()
    assert torch.allclose(out["loss"], out["cpc_loss"])
    assert 0.0 <= float(out["cpc_acc"]) <= 1.0
    assert torch.allclose(out["cpc_temperature"], torch.tensor(0.1))

    out["loss"].backward()
    grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------- #
# same-trajectory negative masking
# --------------------------------------------------------------------------- #


def test_ac_cpc_single_trajectory_has_no_negatives_so_loss_is_zero():
    """With one trajectory, every off-diagonal candidate is a same-trajectory
    frame and is masked out, leaving only the positive. The softmax is then over
    a single surviving logit, so the loss is exactly 0 and accuracy is 1."""
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, 3 * 2 * 2), embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = ac_cpc_forward(harness, _batch(batch=1, embed_dim=embed_dim), "fit", _cfg())
    assert float(out["loss"]) < 1e-5
    assert float(out["cpc_acc"]) == 1.0


# --------------------------------------------------------------------------- #
# The core anti-collapse claim
# --------------------------------------------------------------------------- #


def test_collapsed_encoder_is_pinned_at_chance_infonce():
    """A constant (collapsed) encoder maps every future to the same point, so all
    surviving candidates have identical cosine similarity and the softmax is
    uniform. The InfoNCE loss is therefore pinned at log(#candidates) — chance —
    and cannot be driven toward zero. This is exactly why AC-CPC resists collapse
    without SIGReg: the only way to lower the loss is to make futures
    *distinguishable*, i.e. to encode what varies across trajectories."""
    torch.manual_seed(0)
    embed_dim = 6
    batch, steps = 2, 4  # P = steps - num_preds = 3 predictions per trajectory
    model = _make_jepa(ConstantEncoder(embed_dim), embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = ac_cpc_forward(harness, _batch(batch=batch, steps=steps, embed_dim=embed_dim), "fit", _cfg())

    # each row: 1 positive + P negatives from the single other trajectory
    n_preds = 1
    P = steps - n_preds
    n_candidates = 1 + (batch - 1) * P
    assert math.isclose(float(out["loss"]), math.log(n_candidates), rel_tol=1e-4)


def test_distinguishable_futures_beat_chance():
    """A learnable encoder trained for a few steps drives the InfoNCE loss below
    the collapsed/chance floor — the optimisable direction is to separate
    futures, not to collapse them."""
    torch.manual_seed(0)
    embed_dim = 8
    batch, steps = 6, 4
    model = _make_jepa(LearnableEncoder(embed_dim, 3 * 2 * 2), embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)
    batch_data = _batch(batch=batch, steps=steps, embed_dim=embed_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    first = float(ac_cpc_forward(harness, batch_data, "fit", _cfg())["loss"])
    for _ in range(60):
        opt.zero_grad()
        loss = ac_cpc_forward(harness, batch_data, "fit", _cfg())["loss"]
        loss.backward()
        opt.step()
    last = float(ac_cpc_forward(harness, batch_data, "fit", _cfg())["loss"])

    P = steps - 1
    chance = math.log(1 + (batch - 1) * P)
    assert last < first
    assert last < chance  # learned to distinguish futures -> below the collapse floor


# --------------------------------------------------------------------------- #
# config composition
# --------------------------------------------------------------------------- #


def test_accpc_config_raw_yaml_selects_ac_cpc_mode():
    cfg = yaml.safe_load((ROOT / "config/train/lewm_accpc.yaml").read_text())
    assert cfg["wm"]["type"] == "lewm_accpc"
    assert cfg["wm"]["prediction_mode"] == "ac_cpc"
    assert cfg["loss"]["sigreg"]["weight"] == 0.0
    assert cfg["loss"]["cpc"]["temperature"] == 0.1


def test_accpc_config_composes_and_disables_sigreg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_accpc")

    assert cfg.encoder_scale == "tiny"
    assert cfg.wm.embed_dim == 192
    assert cfg.loss.sigreg.weight == 0.0
    assert cfg.loss.sigreg.kwargs.num_proj == 1024  # survives deep-merge, unused at w=0
    assert cfg.wm.prediction_mode == "ac_cpc"
    assert cfg.loss.cpc.temperature == 0.1
