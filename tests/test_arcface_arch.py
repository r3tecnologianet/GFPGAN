import torch

from gfpgan.archs.arcface_arch import BasicBlock, Bottleneck, ResNetArcFace

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_resnetarcface():
    """Test arch: ResNetArcFace."""

    net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=True).to(DEVICE).eval()
    img = torch.rand((1, 1, 128, 128), dtype=torch.float32).to(DEVICE)
    output = net(img)
    assert output.shape == (1, 512)

    # -------------------- without SE block ----------------------- #
    net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=False).to(DEVICE).eval()
    output = net(img)
    assert output.shape == (1, 512)


def test_basicblock():
    """Test the BasicBlock in arcface_arch"""
    block = BasicBlock(1, 3, stride=1, downsample=None).to(DEVICE)
    img = torch.rand((1, 1, 12, 12), dtype=torch.float32).to(DEVICE)
    output = block(img)
    assert output.shape == (1, 3, 12, 12)

    # ----------------- use the downsmaple module--------------- #
    downsample = torch.nn.UpsamplingNearest2d(scale_factor=0.5).to(DEVICE)
    block = BasicBlock(1, 3, stride=2, downsample=downsample).to(DEVICE)
    img = torch.rand((1, 1, 12, 12), dtype=torch.float32).to(DEVICE)
    output = block(img)
    assert output.shape == (1, 3, 6, 6)


def test_bottleneck():
    """Test the Bottleneck in arcface_arch"""
    block = Bottleneck(1, 1, stride=1, downsample=None).to(DEVICE)
    img = torch.rand((1, 1, 12, 12), dtype=torch.float32).to(DEVICE)
    output = block(img)
    assert output.shape == (1, 4, 12, 12)

    # ----------------- use the downsmaple module--------------- #
    downsample = torch.nn.UpsamplingNearest2d(scale_factor=0.5).to(DEVICE)
    block = Bottleneck(1, 1, stride=2, downsample=downsample).to(DEVICE)
    img = torch.rand((1, 1, 12, 12), dtype=torch.float32).to(DEVICE)
    output = block(img)
    assert output.shape == (1, 4, 6, 6)
