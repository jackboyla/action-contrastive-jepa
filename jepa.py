"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor=None,
        action_encoder=None,
        projector=None,
        pred_proj=None,
        future_predictor=None,
        inverse_predictor=None,
        target_encoder=None,
        target_projector=None,
        decoder=None,
        rollout_mode="autoregressive",
        preprocess_pixels=False,
        image_size=None,
        normalize_latents=False,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.future_predictor = future_predictor
        self.inverse_predictor = inverse_predictor
        # EMA target encoder for BYOL-style temporal prediction. None for all
        # other prediction modes — their behaviour is completely unchanged.
        self.target_encoder = target_encoder
        self.target_projector = target_projector or nn.Identity()
        # Optional reconstruction decoder (anti-collapse auxiliary). None for all
        # other modes, leaving their behaviour unchanged.
        self.decoder = decoder
        self.rollout_mode = rollout_mode
        # AC-CPC: L2-normalize every latent (encoder output + predictor output) so
        # the whole pipeline lives on the unit sphere. Then cosine similarity (the
        # InfoNCE training signal) and the planner's MSE criterion agree: for unit
        # vectors ||a-b||^2 = 2 - 2 a.b, so minimizing planning-MSE == maximizing
        # cosine similarity. Off by default -> baselines/masked are unchanged.
        self.normalize_latents = bool(normalize_latents)
        self.preprocess_pixels = bool(preprocess_pixels)
        self.image_size = image_size
        self.register_buffer(
            "_image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_pixels(self, pixels):
        # getattr default keeps released/older pickled checkpoints loadable: they
        # predate the GPU-preprocess feature and have no `preprocess_pixels`
        # attribute, so they fall through to the no-op path (CPU preprocessing was
        # done in the dataset transform at their train/eval time).
        if not getattr(self, "preprocess_pixels", False):
            return pixels.float()

        if pixels.dtype == torch.uint8:
            pixels = pixels.float().div_(255.0)
            pixels = (pixels - self._image_mean) / self._image_std
        else:
            pixels = pixels.float()

        if self.image_size is not None and tuple(pixels.shape[-2:]) != (
            int(self.image_size),
            int(self.image_size),
        ):
            shape = pixels.shape
            flat = rearrange(pixels, "b t c h w -> (b t) c h w")
            flat = F.interpolate(
                flat,
                size=(int(self.image_size), int(self.image_size)),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            pixels = rearrange(flat, "(b t) c h w -> b t c h w", b=shape[0])

        return pixels

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = self._prepare_pixels(info['pixels'])
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        if getattr(self, "normalize_latents", False):
            emb = F.normalize(emb, dim=-1)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        if self.predictor is None:
            raise RuntimeError("autoregressive predictor is not configured")
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        if getattr(self, "normalize_latents", False):
            preds = F.normalize(preds, dim=-1)
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict_future(self, ctx_emb, ctx_act_emb, fut_act_emb):
        """Predict a direct future embedding trajectory.
        ctx_emb: (B, C, D)
        ctx_act_emb: (B, C, A_emb)
        fut_act_emb: (B, H, A_emb)
        """
        if self.future_predictor is None:
            raise RuntimeError("future predictor is not configured")
        preds = self.future_predictor(ctx_emb, ctx_act_emb, fut_act_emb)
        preds = self.pred_proj(rearrange(preds, "b h d -> (b h) d"))
        if getattr(self, "normalize_latents", False):
            preds = F.normalize(preds, dim=-1)
        preds = rearrange(preds, "(b h) d -> b h d", b=ctx_emb.size(0))
        return preds

    def predict_action(self, z_t, z_next, horizon=None):
        """Inverse dynamics: predict the action that carried z_t -> z_next.

        Used only as a training-time anti-collapse objective in masked
        transition modeling; the planner (`rollout`/`get_cost`) never calls it.
        z_t, z_next: (B, T, D)
        horizon: scalar int or None (for plain InverseDynamics)
        returns (B, T, action_dim)
        """
        if self.inverse_predictor is None:
            raise RuntimeError("inverse predictor is not configured")
        if horizon is None:
            return self.inverse_predictor(z_t, z_next)
        return self.inverse_predictor(z_t, z_next, horizon)

    def encode_target(self, info):
        """EMA target encoder pass — no gradient, no action encoding."""
        if self.target_encoder is None:
            raise RuntimeError("target_encoder not configured for this model")
        with torch.no_grad():
            pixels = self._prepare_pixels(info["pixels"])
            b = pixels.size(0)
            pixels = rearrange(pixels, "b t ... -> (b t) ...")
            out = self.target_encoder(pixels, interpolate_pos_encoding=True)
            emb = self.target_projector(out.last_hidden_state[:, 0])
        info["target_emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        return info

    def reconstruct(self, emb):
        """Decode planning latents back to low-res frames.
        emb: (B, T, D) -> recon (B, T, 3, out, out)
        """
        if self.decoder is None:
            raise RuntimeError("decoder not configured for this model")
        b = emb.size(0)
        flat = rearrange(emb, "b t d -> (b t) d")
        recon = self.decoder(flat)
        return rearrange(recon, "(b t) c h w -> b t c h w", b=b)

    def recon_target(self, info, out_size):
        """Build the reconstruction target: the (prepared, normalized) frames the
        encoder sees, bilinearly downsampled to out_size. (B, T, 3, out, out)."""
        pixels = self._prepare_pixels(info["pixels"])  # (B, T, C, H, W) float
        b = pixels.size(0)
        flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        flat = F.interpolate(
            flat, size=(out_size, out_size), mode="bilinear", align_corners=False, antialias=True
        )
        return rearrange(flat, "(b t) c h w -> b t c h w", b=b)

    def ema_update(self, momentum: float) -> None:
        """EMA update: θ_target ← momentum·θ_target + (1−momentum)·θ_online."""
        if self.target_encoder is None:
            return
        pairs = (
            list(zip(self.encoder.parameters(), self.target_encoder.parameters()))
            + list(zip(self.projector.parameters(), self.target_projector.parameters()))
        )
        for p_online, p_target in pairs:
            p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def rollout_direct(self, info, action_sequence):
        """Directly predict future embeddings for an action sequence."""

        assert "pixels" in info, "pixels not in info_dict"
        H0 = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        if H0 > T:
            raise ValueError("action_sequence must be at least as long as the context")

        act_0 = action_sequence[:, :, :H0]
        info["action"] = act_0

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act_0 = rearrange(act_0, "b s ... -> (b s) ...")
        fut_act = rearrange(action_sequence[:, :, H0 - 1 :], "b s ... -> (b s) ...")

        ctx_act_emb = self.action_encoder(act_0)
        fut_act_emb = self.action_encoder(fut_act)
        pred_emb = self.predict_future(emb, ctx_act_emb, fut_act_emb)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                value = info_dict[k]
                if value.is_floating_point():
                    value = value.float()
                info_dict[k] = value.to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        if getattr(self, "rollout_mode", "autoregressive") == "direct_horizon":
            info_dict = self.rollout_direct(info_dict, action_candidates)
        else:
            info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
