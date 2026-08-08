"""Author-given auxiliary decoder for visualizing frozen LeWM latent embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CLSDecoder(nn.Module):
    """Author-provided CLS-to-pixels transformer decoder.

    This matches the LeWorldModel appendix description: project one CLS embedding
    into key/value tokens, cross-attend with one learned query per output patch,
    then project each query to a pixel patch and rearrange patches into an image.
    """

    def __init__(
        self,
        cls_dim: int = 384,
        img_size: int = 224,
        patch_size: int = 16,
        dim: int = 256,
        heads: int = 8,
        depth: int = 3,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.cls_dim = cls_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        self.queries = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.queries, std=0.02)

        self.cls_proj = nn.Sequential(
            nn.Linear(cls_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            dim, heads, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(dim),
                        "mlp": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
            )

        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        patch_grid = int(self.num_patches**0.5)

        kv = self.cls_proj(x).unsqueeze(1)
        q = self.queries.expand(bsz, -1, -1)

        for layer in self.layers:
            attn_out = layer["cross_attn"](q, kv, kv)[0]
            q = layer["norm1"](q + attn_out)
            mlp_out = layer["mlp"](q)
            q = layer["norm2"](q + mlp_out)

        patches = self.to_pixels(q)
        patches = patches.reshape(
            bsz,
            patch_grid,
            patch_grid,
            self.patch_size,
            self.patch_size,
            3,
        )
        img = patches.permute(0, 5, 1, 3, 2, 4)
        return img.reshape(
            bsz,
            3,
            patch_grid * self.patch_size,
            patch_grid * self.patch_size,
        )


def build_decoder(
    *,
    architecture: str,
    latent_dim: int,
    img_size: int,
    patch_size: int = 16,
    dim: int = 256,
    heads: int = 8,
    depth: int = 3,
) -> nn.Module:
    if architecture not in {"cls", "cls_transformer", "author_cls"}:
        raise ValueError(
            "only the author-given CLS transformer decoder is supported; "
            f"got architecture={architecture!r}"
        )
    return CLSDecoder(
        cls_dim=latent_dim,
        img_size=img_size,
        patch_size=patch_size,
        dim=dim,
        heads=heads,
        depth=depth,
    )


def build_decoder_from_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("model", {})
    decoder = build_decoder(
        architecture=str(model_cfg.get("architecture", "cls_transformer")),
        latent_dim=int(checkpoint["latent_dim"]),
        img_size=int(checkpoint["img_size"]),
        patch_size=int(model_cfg.get("patch_size", 16)),
        dim=int(model_cfg.get("dim", model_cfg.get("hidden_dim", 256))),
        heads=int(model_cfg.get("heads", 8)),
        depth=int(model_cfg.get("depth", 3)),
    )
    decoder.load_state_dict(checkpoint["decoder"])
    return decoder, checkpoint
