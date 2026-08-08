from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn

from jepa import JEPA
from module import ARPredictor, HorizonInverseDynamics, InverseDynamics
from train import (
    action_discrimination_loss,
    masked_endpoint_inverse_forward,
    masked_transition_forward,
    state_transition_discrimination_loss,
)


ROOT = Path(__file__).resolve().parents[1]


class ConstantEncoder(nn.Module):
    """Collapsed encoder: maps every observation to the same vector."""

    def __init__(self, embed_dim, value=0.0):
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
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        flat = pixels.flatten(1)
        cls = self.proj(flat).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=cls)


def _make_jepa(encoder, embed_dim, action_dim, inverse=True):
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
    inverse_predictor = (
        InverseDynamics(latent_dim=embed_dim, action_dim=action_dim, hidden_dim=32, depth=2)
        if inverse
        else None
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=nn.Identity(),  # action_dim == embed_dim so act_emb == action
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        inverse_predictor=inverse_predictor,
        rollout_mode="autoregressive",
    )


def _make_horizon_inverse_jepa(encoder, embed_dim, action_dim, max_horizon=3):
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
    inverse_predictor = HorizonInverseDynamics(
        latent_dim=embed_dim,
        action_dim=action_dim,
        max_horizon=max_horizon,
        horizon_embed_dim=4,
        hidden_dim=32,
        depth=2,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        inverse_predictor=inverse_predictor,
        rollout_mode="autoregressive",
    )


def _cfg(
    mask_mode="both",
    inverse_weight=1.0,
    forward_prob=0.5,
    inverse_warmup_epochs=0,
    inverse_loss_type="mse",
    state_nce_weight=0.0,
    state_nce_mode="delta",
    state_nce_hard_negatives=0,
):
    return OmegaConf.create(
        {
            "wm": {"history_size": 3, "num_preds": 1},
            "loss": {
                "masked": {
                    "mask_mode": mask_mode,
                    "inverse_weight": inverse_weight,
                    "inverse_warmup_epochs": inverse_warmup_epochs,
                    "forward_prob": forward_prob,
                    "inverse_loss_type": inverse_loss_type,
                    "action_nce_temperature": 0.1,
                    "state_nce": {
                        "weight": state_nce_weight,
                        "mode": state_nce_mode,
                        "temperature": 0.1,
                        "hard_negatives": state_nce_hard_negatives,
                    },
                }
            },
        }
    )


def _endpoint_inverse_cfg(
    max_horizon=3,
    train_mode="random",
    eval_mode="all",
    inverse_loss_type="mse",
):
    return OmegaConf.create(
        {
            "wm": {"history_size": 3, "num_preds": 1, "horizon": max_horizon},
            "loss": {
                "masked": {
                    "inverse_weight": 1.0,
                    "inverse_warmup_epochs": 0,
                    "inverse_loss_type": inverse_loss_type,
                    "action_nce_temperature": 0.1,
                    "inverse_min_horizon": 1,
                    "inverse_max_horizon": max_horizon,
                    "inverse_train_mode": train_mode,
                    "inverse_eval_mode": eval_mode,
                }
            },
        }
    )


def _batch(batch=2, steps=4, embed_dim=6, channels=3, hw=2):
    return {
        "pixels": torch.randn(batch, steps, channels, hw, hw),
        "action": torch.randn(batch, steps, embed_dim),
    }


# --------------------------------------------------------------------------- #
# InverseDynamics module
# --------------------------------------------------------------------------- #


def test_inverse_dynamics_shape_and_backprop():
    torch.manual_seed(0)
    batch, steps, dim, act = 2, 3, 8, 5
    head = InverseDynamics(latent_dim=dim, action_dim=act, hidden_dim=16, depth=2)

    z_t = torch.randn(batch, steps, dim, requires_grad=True)
    z_next = torch.randn(batch, steps, dim)
    out = head(z_t, z_next)

    assert out.shape == (batch, steps, act)
    out.square().mean().backward()
    assert z_t.grad is not None and torch.isfinite(z_t.grad).all()


def test_inverse_dynamics_rejects_zero_depth():
    with pytest.raises(ValueError, match="depth"):
        InverseDynamics(latent_dim=4, action_dim=2, depth=0)


def test_jepa_predict_action_requires_inverse_predictor():
    model = _make_jepa(ConstantEncoder(4), embed_dim=4, action_dim=4, inverse=False)
    with pytest.raises(RuntimeError, match="inverse predictor"):
        model.predict_action(torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))


def test_action_discrimination_loss_backprops_and_reports_accuracy():
    torch.manual_seed(0)
    pred = torch.randn(3, 2, 4, requires_grad=True)
    target = torch.randn(3, 2, 4)

    loss, acc = action_discrimination_loss(pred, target, temperature=0.1)

    assert torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_state_transition_discrimination_loss_backprops_and_reports_accuracy():
    torch.manual_seed(0)
    anchor = torch.randn(3, 2, 5)
    target_delta = torch.randn(3, 2, 5)
    pred = (anchor + target_delta + 0.01 * torch.randn(3, 2, 5)).requires_grad_()
    target = anchor + target_delta

    loss, acc = state_transition_discrimination_loss(
        pred, target, anchor, mode="delta", temperature=0.1
    )

    assert torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_state_transition_discrimination_loss_supports_hard_negatives():
    torch.manual_seed(0)
    anchor = torch.randn(4, 2, 5)
    target = torch.randn(4, 2, 5)
    pred = (target + 0.01 * torch.randn(4, 2, 5)).requires_grad_()

    hard_loss, hard_acc = state_transition_discrimination_loss(
        pred,
        target,
        anchor,
        mode="absolute",
        temperature=0.1,
        hard_negatives=2,
    )
    full_loss, _ = state_transition_discrimination_loss(
        pred,
        target,
        anchor,
        mode="absolute",
        temperature=0.1,
    )

    assert torch.isfinite(hard_loss)
    assert torch.isfinite(full_loss)
    assert 0.0 <= float(hard_acc) <= 1.0
    assert hard_loss <= full_loss + 1e-6
    hard_loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# --------------------------------------------------------------------------- #
# masked_transition_forward objective
# --------------------------------------------------------------------------- #


def test_masked_forward_returns_both_losses_and_backprops():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    sigreg = lambda emb: emb.pow(2).mean()
    harness = SimpleNamespace(model=model, sigreg=sigreg, log_dict=lambda *a, **k: None)

    out = masked_transition_forward(harness, _batch(embed_dim=embed_dim), "fit", _cfg())

    for key in ("forward_loss", "inverse_loss", "loss", "emb_std"):
        assert key in out and torch.isfinite(out[key]).all()
    # combined loss == forward + inverse at unit weight
    assert torch.allclose(out["loss"], out["forward_loss"] + out["inverse_loss"])

    out["loss"].backward()
    grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # the inverse-dynamics head trains too
    assert all(p.grad is not None for p in model.inverse_predictor.parameters())


def test_masked_forward_inverse_weight_scales_inverse_term():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = masked_transition_forward(
        harness, _batch(embed_dim=embed_dim), "fit", _cfg(inverse_weight=3.0)
    )
    assert torch.allclose(out["loss"], out["forward_loss"] + 3.0 * out["inverse_loss"])
    assert torch.allclose(out["inverse_weight_effective"], torch.tensor(3.0))


def test_masked_forward_action_nce_inverse_is_discriminative_and_backprops():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = masked_transition_forward(
        harness,
        _batch(batch=3, embed_dim=embed_dim),
        "fit",
        _cfg(inverse_loss_type="action_nce"),
    )

    for key in ("forward_loss", "inverse_loss", "inverse_acc", "loss", "emb_std"):
        assert key in out and torch.isfinite(out[key]).all()
    assert 0.0 <= float(out["inverse_acc"]) <= 1.0
    assert torch.allclose(out["loss"], out["forward_loss"] + out["inverse_loss"])

    out["loss"].backward()
    assert all(p.grad is not None for p in model.inverse_predictor.parameters())


def test_masked_forward_state_nce_adds_transition_contrastive_loss():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = masked_transition_forward(
        harness,
        _batch(batch=3, embed_dim=embed_dim),
        "fit",
        _cfg(inverse_loss_type="action_nce", state_nce_weight=0.2),
    )

    for key in ("state_nce_loss", "state_nce_acc", "state_nce_weight"):
        assert key in out and torch.isfinite(out[key]).all()
    expected = out["forward_loss"] + out["inverse_loss"] + 0.2 * out["state_nce_loss"]
    assert torch.allclose(out["loss"], expected)

    out["loss"].backward()
    assert all(p.grad is not None for p in model.encoder.parameters())


def test_masked_forward_state_nce_accepts_absolute_hard_negatives():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = masked_transition_forward(
        harness,
        _batch(batch=4, embed_dim=embed_dim),
        "fit",
        _cfg(
            inverse_loss_type="action_nce",
            state_nce_weight=0.2,
            state_nce_mode="absolute",
            state_nce_hard_negatives=3,
        ),
    )

    for key in ("state_nce_loss", "state_nce_acc", "state_nce_weight"):
        assert key in out and torch.isfinite(out[key]).all()
    expected = out["forward_loss"] + out["inverse_loss"] + 0.2 * out["state_nce_loss"]
    assert torch.allclose(out["loss"], expected)


def test_masked_forward_logs_collapse_diagnostics():
    """effective_rank + inverse_margin are emitted, finite, in range, for both
    inverse loss types, and never alter the training loss."""
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    # MSE inverse: margin is the mean-action MSE floor minus the achieved loss.
    out = masked_transition_forward(harness, _batch(embed_dim=embed_dim), "fit", _cfg())
    for key in ("effective_rank", "inverse_margin", "inverse_baseline"):
        assert key in out and torch.isfinite(out[key]).all()
    assert 1.0 <= float(out["effective_rank"]) <= embed_dim + 1e-3
    # diagnostics are pure: the loss is still exactly forward + inverse
    assert torch.allclose(out["loss"], out["forward_loss"] + out["inverse_loss"])

    # NCE inverse: margin is accuracy over chance (acc - 1/N), baseline = 1/N.
    batch = 3
    out = masked_transition_forward(
        harness, _batch(batch=batch, embed_dim=embed_dim), "fit",
        _cfg(inverse_loss_type="action_nce"),
    )
    n = batch * (4 - 1)  # B * transitions
    assert torch.allclose(out["inverse_baseline"], torch.tensor(1.0 / n))
    assert torch.allclose(out["inverse_margin"], out["inverse_acc"].float() - 1.0 / n)


def test_masked_forward_effective_rank_detects_collapse():
    """A constant encoder reads emb_std ~ 0 and effective_rank ~ 1."""
    embed_dim = 6
    model = _make_jepa(ConstantEncoder(embed_dim, value=0.7), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)
    out = masked_transition_forward(harness, _batch(embed_dim=embed_dim), "fit", _cfg())
    assert float(out["emb_std"]) < 1e-5
    assert abs(float(out["effective_rank"]) - 1.0) < 1e-2


def test_masked_forward_optional_sigreg_weight_adds_regularizer():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    sigreg = lambda emb: emb.pow(2).mean()
    harness = SimpleNamespace(model=model, sigreg=sigreg, log_dict=lambda *a, **k: None)
    cfg = _cfg()
    cfg.loss.sigreg = {"weight": 0.2}

    out = masked_transition_forward(harness, _batch(embed_dim=embed_dim), "fit", cfg)

    assert "sigreg_loss" in out
    expected = out["forward_loss"] + out["inverse_loss"] + 0.2 * out["sigreg_loss"]
    assert torch.allclose(out["loss"], expected)


def test_masked_forward_inverse_weight_warmup_uses_current_epoch():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, current_epoch=2, log_dict=lambda *a, **k: None)

    out = masked_transition_forward(
        harness,
        _batch(embed_dim=embed_dim),
        "fit",
        _cfg(inverse_weight=0.9, inverse_warmup_epochs=4),
    )

    expected_weight = torch.tensor(0.45)
    assert torch.allclose(out["inverse_weight_effective"], expected_weight)
    assert torch.allclose(out["loss"], out["forward_loss"] + expected_weight * out["inverse_loss"])


def test_masked_forward_random_mode_picks_one_task():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(LearnableEncoder(embed_dim, in_features=3 * 2 * 2), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    # forward_prob=1 -> loss is exactly the forward term; both metrics still logged.
    out = masked_transition_forward(
        harness, _batch(embed_dim=embed_dim), "fit", _cfg(mask_mode="random", forward_prob=1.0)
    )
    assert torch.allclose(out["loss"], out["forward_loss"])
    assert torch.isfinite(out["inverse_loss"]).all()

    # forward_prob=0 -> loss is exactly the (weighted) inverse term.
    out = masked_transition_forward(
        harness, _batch(embed_dim=embed_dim), "fit", _cfg(mask_mode="random", forward_prob=0.0)
    )
    assert torch.allclose(out["loss"], out["inverse_loss"])


def test_masked_forward_rejects_unknown_mask_mode():
    model = _make_jepa(LearnableEncoder(6, in_features=3 * 2 * 2), 6, 6)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)
    with pytest.raises(ValueError, match="mask_mode"):
        masked_transition_forward(harness, _batch(embed_dim=6), "fit", _cfg(mask_mode="bogus"))


def test_masked_endpoint_inverse_keeps_ar_forward_and_uses_horizon_head():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_horizon_inverse_jepa(
        LearnableEncoder(embed_dim, in_features=3 * 2 * 2),
        embed_dim,
        embed_dim,
        max_horizon=3,
    )
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = masked_endpoint_inverse_forward(
        harness,
        _batch(steps=6, embed_dim=embed_dim),
        "fit",
        _endpoint_inverse_cfg(max_horizon=3),
    )

    for key in ("forward_loss", "inverse_loss", "loss", "emb_std", "inverse_horizon"):
        assert key in out and torch.isfinite(out[key]).all()
    assert torch.allclose(out["loss"], out["forward_loss"] + out["inverse_loss"])

    out["loss"].backward()
    assert all(p.grad is not None for p in model.inverse_predictor.parameters())


def test_masked_endpoint_inverse_action_nce_is_discriminative():
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_horizon_inverse_jepa(
        LearnableEncoder(embed_dim, in_features=3 * 2 * 2),
        embed_dim,
        embed_dim,
        max_horizon=3,
    )
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    out = masked_endpoint_inverse_forward(
        harness,
        _batch(batch=3, steps=6, embed_dim=embed_dim),
        "fit",
        _endpoint_inverse_cfg(max_horizon=3, inverse_loss_type="action_nce"),
    )

    for key in ("inverse_loss", "inverse_acc", "inverse_baseline", "inverse_margin"):
        assert key in out and torch.isfinite(out[key]).all()
    assert 0.0 <= out["inverse_acc"] <= 1.0
    assert torch.allclose(out["loss"], out["forward_loss"] + out["inverse_loss"])

    out["loss"].backward()
    assert all(p.grad is not None for p in model.inverse_predictor.parameters())


# --------------------------------------------------------------------------- #
# The core claim: a collapsed encoder cannot solve inverse dynamics
# --------------------------------------------------------------------------- #


def test_collapsed_encoder_cannot_distinguish_actions():
    """With a constant (collapsed) encoder every (z_t, z_{t+1}) pair is identical,
    so the inverse head must emit the SAME action prediction for every transition
    regardless of the true action. This is exactly why masking the action prevents
    collapse: the loss is floored at the action variance and cannot reach zero."""
    torch.manual_seed(0)
    embed_dim = 6
    model = _make_jepa(ConstantEncoder(embed_dim, value=0.3), embed_dim, embed_dim)
    harness = SimpleNamespace(model=model, log_dict=lambda *a, **k: None)

    batch = _batch(embed_dim=embed_dim)
    out = masked_transition_forward(harness, batch, "fit", _cfg())

    # collapse monitor reads ~0 spread
    assert out["emb_std"] < 1e-5

    # inverse predictions are identical across every transition (input is constant)
    emb = model.encode(batch)["emb"]
    pred_act = model.predict_action(emb[:, :-1], emb[:, 1:])
    flat = pred_act.flatten(0, 1)
    assert torch.allclose(flat, flat[:1].expand_as(flat), atol=1e-5)

    # therefore the inverse loss cannot beat predicting the constant mean action:
    # it stays at least as large as the action variance, never 0.
    tgt = batch["action"][:, : emb.size(1) - 1]
    best_constant_loss = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
    assert out["inverse_loss"] >= 0.5 * best_constant_loss


# --------------------------------------------------------------------------- #
# Config composition
# --------------------------------------------------------------------------- #


def test_masked_config_raw_yaml_selects_masked_mode():
    cfg = yaml.safe_load((ROOT / "config/train/lewm_masked.yaml").read_text())
    assert cfg["wm"]["type"] == "lewm_masked"
    assert cfg["wm"]["prediction_mode"] == "masked_transition"
    assert cfg["loss"]["sigreg"]["weight"] == 0.0
    assert cfg["loss"]["masked"]["mask_mode"] == "both"
    assert cfg["loss"]["masked"]["inverse_warmup_epochs"] == 0
    assert cfg["inverse"]["depth"] == 2


def test_masked_config_composes_and_disables_sigreg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_masked")

    # base lewm settings still inherited
    assert cfg.encoder_scale == "tiny"
    assert cfg.wm.embed_dim == 192
    # SIGReg disabled but its kwargs survive the deep-merge (defensive, unused at w=0)
    assert cfg.loss.sigreg.weight == 0.0
    assert cfg.loss.sigreg.kwargs.num_proj == 1024
    assert cfg.wm.prediction_mode == "masked_transition"
    assert cfg.inverse.hidden_dim == 512
    assert cfg.loss.masked.inverse_warmup_epochs == 0


def test_masked_action_nce_config_composes():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_masked_action_nce")

    assert cfg.wm.prediction_mode == "masked_transition"
    assert cfg.output_model_name == "lewm_masked_action_nce"
    assert cfg.loss.sigreg.weight == 0.0
    assert cfg.loss.masked.inverse_loss_type == "action_nce"
    assert cfg.loss.masked.action_nce_temperature == 0.1


def test_masked_action_state_nce_config_composes():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_masked_action_state_nce")

    assert cfg.wm.prediction_mode == "masked_transition"
    assert cfg.output_model_name == "lewm_masked_action_state_nce"
    assert cfg.loss.masked.inverse_loss_type == "action_nce"
    assert cfg.loss.masked.inverse_weight == 0.30
    assert cfg.loss.masked.state_nce.weight == 0.03
    assert cfg.loss.masked.state_nce.mode == "delta"


def test_masked_state_nce_config_composes_as_action_nce_replacement():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_masked_state_nce")

    assert cfg.wm.prediction_mode == "masked_transition"
    assert cfg.output_model_name == "lewm_masked_state_nce"
    assert cfg.loss.masked.inverse_loss_type == "action_nce"
    assert cfg.loss.masked.inverse_weight == 0.0
    assert cfg.loss.masked.state_nce.weight == 0.10
    assert cfg.loss.masked.state_nce.mode == "delta"


def test_masked_action_state_hard_nce_config_composes():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_masked_action_state_hard_nce")

    assert cfg.wm.prediction_mode == "masked_transition"
    assert cfg.output_model_name == "lewm_masked_action_state_hard_nce"
    assert cfg.loss.masked.inverse_loss_type == "action_nce"
    assert cfg.loss.masked.inverse_weight == 0.30
    assert cfg.loss.masked.state_nce.weight == 0.03
    assert cfg.loss.masked.state_nce.mode == "absolute"
    assert cfg.loss.masked.state_nce.hard_negatives == 64


def test_masked_endpoint_action_nce_config_composes():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="lewm_masked_endpoint_action_nce")

    assert cfg.wm.prediction_mode == "masked_endpoint_inverse"
    assert cfg.output_model_name == "lewm_masked_endpoint_action_nce"
    assert cfg.loss.masked.inverse_loss_type == "action_nce"
    assert cfg.loss.masked.action_nce_temperature == 0.1
    assert cfg.loss.masked.inverse_weight == 0.30
    assert cfg.loss.masked.inverse_min_horizon == 1
    assert cfg.loss.masked.inverse_max_horizon == cfg.wm.horizon
    assert cfg.data.dataset.num_steps == cfg.wm.history_size + cfg.wm.horizon
