import pytest
import torch

from decoder import CLSDecoder, build_decoder, build_decoder_from_checkpoint


def test_cls_decoder_matches_paper_patch_layout_and_backprops():
    decoder = CLSDecoder(
        cls_dim=8,
        img_size=32,
        patch_size=16,
        dim=16,
        heads=4,
        depth=2,
    )
    z = torch.randn(2, 8, requires_grad=True)

    out = decoder(z)

    assert out.shape == (2, 3, 32, 32)
    assert decoder.num_patches == 4
    out.mean().backward()
    assert z.grad is not None


def test_build_decoder_rejects_legacy_architectures():
    with pytest.raises(ValueError, match="author-given CLS transformer"):
        build_decoder(
            architecture="legacy_decoder",
            latent_dim=8,
            img_size=32,
        )


def test_cls_decoder_checkpoint_roundtrip(tmp_path):
    decoder = CLSDecoder(
        cls_dim=8,
        img_size=32,
        patch_size=16,
        dim=16,
        heads=4,
        depth=2,
    )
    path = tmp_path / "decoder.pt"
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "latent_dim": 8,
            "img_size": 32,
            "target_space": "rgb",
            "model": {
                "architecture": "cls_transformer",
                "patch_size": 16,
                "dim": 16,
                "heads": 4,
                "depth": 2,
            },
        },
        path,
    )

    loaded, checkpoint = build_decoder_from_checkpoint(path)

    assert loaded(torch.randn(1, 8)).shape == (1, 3, 32, 32)
    assert checkpoint["target_space"] == "rgb"


def test_legacy_decoder_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "decoder.pt"
    torch.save(
        {
            "decoder": {},
            "latent_dim": 8,
            "img_size": 32,
            "model": {"architecture": "legacy_decoder"},
        },
        path,
    )

    with pytest.raises(ValueError, match="author-given CLS transformer"):
        build_decoder_from_checkpoint(path)
