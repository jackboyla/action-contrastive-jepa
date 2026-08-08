from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jepa import JEPA
from module import FutureQueryPredictor


class DummyEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.last_pixels = None

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        self.last_pixels = pixels.detach().clone()
        batch = pixels.size(0)
        cls = torch.zeros(batch, 1, self.embed_dim, device=pixels.device)
        return SimpleNamespace(last_hidden_state=cls)


class RecordingFuturePredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ctx_emb = None
        self.ctx_act = None
        self.fut_act = None

    def forward(self, ctx_emb, ctx_act, fut_act):
        self.ctx_emb = ctx_emb.detach().clone()
        self.ctx_act = ctx_act.detach().clone()
        self.fut_act = fut_act.detach().clone()
        return torch.zeros(
            ctx_emb.size(0),
            fut_act.size(1),
            self.embed_dim,
            device=ctx_emb.device,
        )


def test_future_query_predictor_returns_horizon_sequence_and_backprops():
    torch.manual_seed(0)
    batch, context, horizon, dim = 2, 3, 5, 8
    predictor = FutureQueryPredictor(
        num_context=context,
        horizon=horizon,
        depth=2,
        heads=2,
        mlp_dim=32,
        input_dim=dim,
        hidden_dim=dim,
        dim_head=4,
    )

    ctx_emb = torch.randn(batch, context, dim, requires_grad=True)
    ctx_act = torch.randn(batch, context, dim)
    fut_act = torch.randn(batch, horizon, dim)

    out = predictor(ctx_emb, ctx_act, fut_act)

    assert out.shape == (batch, horizon, dim)
    out.square().mean().backward()
    assert ctx_emb.grad is not None
    assert torch.isfinite(ctx_emb.grad).all()


def test_future_query_predictor_rejects_oversized_context_or_horizon():
    predictor = FutureQueryPredictor(
        num_context=3,
        horizon=5,
        depth=1,
        heads=1,
        mlp_dim=16,
        input_dim=4,
        hidden_dim=4,
        dim_head=4,
    )

    with pytest.raises(ValueError, match="context length"):
        predictor(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), torch.zeros(1, 5, 4))

    with pytest.raises(ValueError, match="horizon"):
        predictor(torch.zeros(1, 3, 4), torch.zeros(1, 3, 4), torch.zeros(1, 6, 4))


def test_jepa_predict_future_projects_predictor_output():
    torch.manual_seed(0)
    batch, context, horizon = 2, 3, 5
    embed_dim, hidden_dim = 8, 12
    future_predictor = FutureQueryPredictor(
        num_context=context,
        horizon=horizon,
        depth=1,
        heads=2,
        mlp_dim=32,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dim_head=6,
    )
    model = JEPA(
        encoder=DummyEncoder(embed_dim),
        action_encoder=nn.Identity(),
        future_predictor=future_predictor,
        pred_proj=nn.Linear(hidden_dim, embed_dim),
        rollout_mode="direct_horizon",
    )

    pred = model.predict_future(
        torch.randn(batch, context, embed_dim),
        torch.randn(batch, context, embed_dim),
        torch.randn(batch, horizon, embed_dim),
    )

    assert pred.shape == (batch, horizon, embed_dim)


def test_jepa_preprocesses_uint8_pixels_on_device():
    encoder = DummyEncoder(embed_dim=4)
    model = JEPA(
        encoder=encoder,
        action_encoder=nn.Identity(),
        preprocess_pixels=True,
        image_size=4,
    )

    pixels = torch.full((1, 2, 3, 2, 2), 255, dtype=torch.uint8)
    model.encode({"pixels": pixels})

    assert encoder.last_pixels.shape == (2, 3, 4, 4)
    assert encoder.last_pixels.dtype == torch.float32
    expected = torch.tensor(
        [
            (1.0 - 0.485) / 0.229,
            (1.0 - 0.456) / 0.224,
            (1.0 - 0.406) / 0.225,
        ]
    ).view(1, 3, 1, 1)
    assert torch.allclose(encoder.last_pixels[:1], expected.expand(1, 3, 4, 4))


def test_rollout_direct_uses_context_and_future_action_window():
    batch, samples, context, total_actions, dim = 2, 3, 3, 7, 4
    future_predictor = RecordingFuturePredictor(embed_dim=dim)
    model = JEPA(
        encoder=DummyEncoder(dim),
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        future_predictor=future_predictor,
        rollout_mode="direct_horizon",
    )

    info = {
        "pixels": torch.zeros(batch, samples, context, 3, 2, 2),
    }
    action_sequence = torch.arange(
        batch * samples * total_actions * dim, dtype=torch.float32
    ).view(batch, samples, total_actions, dim)

    out = model.rollout_direct(info, action_sequence)

    expected_future = total_actions - context + 1
    assert out["predicted_emb"].shape == (
        batch,
        samples,
        context + expected_future,
        dim,
    )

    expected_ctx_act = action_sequence[:, :, :context].reshape(
        batch * samples, context, dim
    )
    expected_fut_act = action_sequence[:, :, context - 1 :].reshape(
        batch * samples, expected_future, dim
    )
    assert torch.equal(future_predictor.ctx_act, expected_ctx_act)
    assert torch.equal(future_predictor.fut_act, expected_fut_act)
