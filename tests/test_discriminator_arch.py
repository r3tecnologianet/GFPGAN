import pytest
import torch
from basicsr.losses.gan_loss import r1_penalty

from gfpgan.archs.discriminator_arch import (FacialComponentDiscriminatorClean, StyleGAN2DiscriminatorClean,
                                             UNetDiscriminatorClean)

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

    # -------------------- batch smaller than the stddev group ----------------------- #
    assert net(torch.rand((3, 3, 64, 64), device=DEVICE)).shape == (3, 1)

    # -------------------- batch not divisible by the stddev group ----------------------- #
    # 6 is the case that forces the group to 1: min(6, 4) is 4 and 6 % 4 is 2. A batch of 3 does not reach it,
    # since min(3, 4) is 3 and 3 % 3 is 0.
    assert net(torch.rand((6, 3, 64, 64), device=DEVICE)).shape == (6, 1)

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


def test_unetdiscriminatorclean_returns_a_per_pixel_logit_map():
    """The background model judges each pixel, so the output keeps the input's spatial size."""
    net = UNetDiscriminatorClean(num_feat=8).to(DEVICE)
    out = net(torch.rand((2, 3, 32, 32), device=DEVICE))
    assert out.shape == (2, 1, 32, 32)


def test_unetdiscriminatorclean_feature_shapes():
    """These three features are what replaces the VGG perceptual loss, so their scales are part of the contract."""
    net = UNetDiscriminatorClean(num_feat=8).to(DEVICE)
    out, feats = net(torch.rand((2, 3, 32, 32), device=DEVICE), return_feats=True)
    assert out.shape == (2, 1, 32, 32)
    assert [f.shape for f in feats] == [(2, 16, 16, 16), (2, 32, 8, 8), (2, 64, 4, 4)]


def test_unetdiscriminatorclean_returns_a_bare_tensor_without_return_feats():
    """It follows StyleGAN2DiscriminatorClean rather than the component discriminator, which always pairs."""
    net = UNetDiscriminatorClean(num_feat=8).to(DEVICE)
    assert torch.is_tensor(net(torch.rand((1, 3, 32, 32), device=DEVICE)))


def test_unetdiscriminatorclean_without_skip_connections():
    net = UNetDiscriminatorClean(num_feat=8, skip_connection=False).to(DEVICE)
    assert net(torch.rand((2, 3, 32, 32), device=DEVICE)).shape == (2, 1, 32, 32)


def test_unetdiscriminatorclean_is_spectrally_normalised():
    """Real-ESRGAN's discriminator needs spectral norm to stay stable; without it the training diverges."""
    net = UNetDiscriminatorClean(num_feat=8)
    normalised = [n for n, _ in net.named_parameters() if n.endswith('weight_orig')]
    assert len(normalised) == 8, 'every convolution but the first and the last'


def test_unetdiscriminatorclean_needs_a_size_divisible_by_eight():
    """Three strided halvings and three upsamples only line up again on multiples of 8."""
    net = UNetDiscriminatorClean(num_feat=8).to(DEVICE)
    with pytest.raises(RuntimeError):
        net(torch.rand((1, 3, 30, 30), device=DEVICE))
