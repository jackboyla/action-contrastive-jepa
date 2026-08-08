import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class VarianceReg(nn.Module):
    """Per-dimension variance lower bound (VICReg variance term).

    Pushes each latent dimension to maintain std >= gamma across the batch via a
    soft hinge: relu(gamma - std(z, dim=batch)).mean(). Zero gradient for dims
    already above gamma, positive gradient for dims below it.

    Prevents lazy dimensions from collapsing when the main task provides easy
    shortcuts that ignore certain state variables (e.g. inverse dynamics ignoring
    block orientation on PushT because position alone mostly determines the action).
    Unlike SIGReg, this does not constrain the distribution shape — it only enforces
    a lower bound on per-dim variance, leaving the representation geometry free.
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) — standard emb convention."""
        std = z.flatten(0, 1).std(dim=0)  # (D,)
        return F.relu(self.gamma - std).mean()


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, causal=True):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), causal=causal
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, causal=True):
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, causal=None):
        causal = self.causal if causal is None else causal

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            if isinstance(block, Block):
                x = block(x, causal=causal)
            else:
                x = block(x, c, causal=causal)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class InverseDynamics(nn.Module):
    """Inverse-dynamics head: predict the action that carried z_t -> z_{t+1}.

    This is the anti-collapse mechanism for masked transition modeling. Where
    LeWM bolts on SIGReg to force the latent marginal into an isotropic Gaussian,
    masked transition modeling lets non-collapse fall out of the prediction task
    itself: a collapsed encoder (f(o) = c) maps every (z_t, z_{t+1}) pair to the
    same point, so the action that produced the transition becomes unrecoverable
    and this loss stays high. Distinct states are forced into existence because
    confusing them makes the preceding action impossible to infer.
    """

    def __init__(
        self,
        *,
        latent_dim,
        action_dim,
        hidden_dim=512,
        depth=2,
        dropout=0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("InverseDynamics depth must be >= 1")

        layers = [nn.Linear(2 * latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, z_next):
        """
        z_t:    (B, T, D) latent at the start of each transition
        z_next: (B, T, D) latent at the end of each transition
        returns (B, T, action_dim) predicted action per transition
        """
        x = torch.cat([z_t, z_next], dim=-1)
        return self.net(x)


class HorizonInverseDynamics(nn.Module):
    """Inverse dynamics conditioned on the temporal gap between endpoints.

    Predicts a_t given (z_t, z_{t+k}, e_k) where e_k is a learned horizon
    embedding. At gap k=5, recovering a_t from a distant endpoint forces the
    encoder to represent slow state variables (e.g. block orientation on PushT)
    that don't matter at k=1 but determine which face the agent pushes in 5 steps.
    """

    def __init__(
        self,
        *,
        latent_dim,
        action_dim,
        max_horizon,
        horizon_embed_dim=32,
        hidden_dim=512,
        depth=2,
        dropout=0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("HorizonInverseDynamics depth must be >= 1")
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")

        self.max_horizon = int(max_horizon)
        self.horizon_embedding = nn.Embedding(self.max_horizon + 1, horizon_embed_dim)

        layers = [
            nn.Linear(2 * latent_dim + horizon_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, z_next, horizon):
        """
        z_t:    (B, T, D) latent at the start of each endpoint pair
        z_next: (B, T, D) latent k steps later
        horizon: scalar int, or (B,) / (B, T) tensor with values in [1, max_horizon]
        returns (B, T, action_dim) predicted first action a_t
        """
        if not torch.is_tensor(horizon):
            horizon = torch.full(
                z_t.shape[:-1], int(horizon), dtype=torch.long, device=z_t.device
            )
        else:
            horizon = horizon.to(device=z_t.device, dtype=torch.long)
            if horizon.ndim == 0:
                horizon = horizon.expand(z_t.shape[:-1])
            else:
                while horizon.ndim < len(z_t.shape[:-1]):
                    horizon = horizon.unsqueeze(-1)
                horizon = horizon.expand(z_t.shape[:-1])

        if torch.any(horizon < 1) or torch.any(horizon > self.max_horizon):
            raise ValueError(
                f"horizon must be in [1, {self.max_horizon}] for HorizonInverseDynamics"
            )

        h_emb = self.horizon_embedding(horizon)
        x = torch.cat([z_t, z_next, h_emb], dim=-1)
        return self.net(x)


class ConvDecoder(nn.Module):
    """Decode a planning latent back to a (low-res) observation.

    Reconstruction auxiliary for anti-collapse: forcing the CLS/projector latent
    to decode the full frame makes information that the dynamics/inverse objective
    would happily discard (e.g. PushT block orientation) un-discardable, because
    you cannot reconstruct the block's pixels without encoding its angle. Unlike
    SIGReg/inverse dynamics, the target (the observation) literally contains
    orientation, so the gradient pushing it into the latent is grounded, not
    incidental.

    latent (N, D) -> image (N, 3, out_size, out_size).
    """

    def __init__(self, latent_dim, out_size=64, base_ch=256):
        super().__init__()
        self.out_size = out_size
        self.base_ch = base_ch
        self.fc = nn.Linear(latent_dim, base_ch * 4 * 4)
        chans = [base_ch, base_ch // 2, base_ch // 4, base_ch // 8]
        layers = []
        in_ch = base_ch
        # 4 -> 8 -> 16 -> 32 -> 64 (four upsampling stages)
        for out_ch in chans[1:] + [chans[-1]]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.up = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

    def forward(self, z):
        """z: (N, D) -> (N, 3, out_size, out_size)."""
        x = self.fc(z).view(-1, self.base_ch, 4, 4)
        x = self.up(x)
        return self.head(x)


class FutureQueryPredictor(nn.Module):
    """Direct multi-horizon predictor for future latent trajectories."""

    def __init__(
        self,
        *,
        num_context,
        horizon,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        causal=False,
    ):
        super().__init__()
        self.num_context = num_context
        self.horizon = horizon
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_context + horizon, input_dim)
        )
        self.future_query = nn.Parameter(torch.randn(1, horizon, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            causal=causal,
        )

    def forward(self, ctx_emb, ctx_act, fut_act):
        """
        ctx_emb: (B, C, D)
        ctx_act: (B, C, A_emb)
        fut_act: (B, H, A_emb)
        """
        B, C, _ = ctx_emb.shape
        H = fut_act.size(1)

        if C > self.num_context:
            raise ValueError(f"context length {C} exceeds configured {self.num_context}")
        if H > self.horizon:
            raise ValueError(f"horizon {H} exceeds configured {self.horizon}")
        if ctx_act.size(1) != C:
            raise ValueError("ctx_act must have the same sequence length as ctx_emb")
        if ctx_act.size(0) != B or fut_act.size(0) != B:
            raise ValueError("all predictor inputs must have the same batch size")

        future_query = self.future_query[:, :H].expand(B, -1, -1)
        x = torch.cat([ctx_emb, future_query], dim=1)
        c = torch.cat([ctx_act, fut_act], dim=1)

        ctx_pos = self.pos_embedding[:, self.num_context - C : self.num_context]
        fut_pos = self.pos_embedding[:, self.num_context : self.num_context + H]
        x = x + torch.cat([ctx_pos, fut_pos], dim=1)

        x = self.dropout(x)
        x = self.transformer(x, c)
        return x[:, C:]
