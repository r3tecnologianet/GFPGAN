import pytest
import torch
from basicsr.losses.gan_loss import r1_penalty

from gfpgan.archs.discriminator_arch import FacialComponentDiscriminatorClean, StyleGAN2DiscriminatorClean

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_stylegan2discriminatorclean():
    """Test arch: StyleGAN2DiscriminatorClean."""
    net = StyleGAN2DiscriminatorClean(out_size=64, channel_multiplier=1).to(DEVICE)
    img = torch.rand((4, 3, 64, 64), device=DEVICE)
    out = net(img)
    assert out.shape == (4, 1)

    # -------------------- return block features ----------------------- #
    out, feats = net(img, return_feats=True)
    assert out.shape == (4, 1)
    assert [f.shape for f in feats] == [(4, 512, 32, 32), (4, 512, 16, 16), (4, 512, 8, 8), (4, 512, 4, 4)]

    # -------------------- batch size not divisible by the stddev group ----------------------- #
    assert net(torch.rand((3, 3, 64, 64), device=DEVICE)).shape == (3, 1)

    # -------------------- invalid size ----------------------- #
    with pytest.raises(ValueError):
        StyleGAN2DiscriminatorClean(out_size=100)


def test_stylegan2discriminatorclean_512_channels():
    net = StyleGAN2DiscriminatorClean(out_size=512, channel_multiplier=2)
    assert net.from_rgb.out_channels == 64
    assert len(net.blocks) == 7


def test_stylegan2discriminatorclean_r1_penalty():
    """R1 needs double backward through the discriminator."""
    net = StyleGAN2DiscriminatorClean(out_size=32).to(DEVICE)
    real = torch.rand((4, 3, 32, 32), device=DEVICE, requires_grad=True)
    real_pred = net(real)
    penalty = r1_penalty(real_pred, real)
    penalty.backward()
    assert torch.isfinite(penalty)
    # weights receive second-order gradients (the last bias does not affect the input gradient)
    assert all(p.grad is not None for name, p in net.named_parameters() if name.endswith('weight'))


def test_facialcomponentdiscriminatorclean():
    """Test arch: FacialComponentDiscriminatorClean."""
    net = FacialComponentDiscriminatorClean().to(DEVICE)
    for size in (80, 120):
        img = torch.rand((2, 3, size, size), device=DEVICE)
        out, feats = net(img)
        assert out.shape == (2, 1, size // 4, size // 4)
        assert feats is None

        out, feats = net(img, return_feats=True)
        assert out.shape == (2, 1, size // 4, size // 4)
        assert len(feats) == 2
        assert feats[0].shape == (2, 128, size // 2, size // 2)
        assert feats[1].shape == (2, 256, size // 4, size // 4)
