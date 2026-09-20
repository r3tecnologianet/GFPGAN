import numpy as np
import pytest
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact

from gfpgan.bg_upsampler import SRBackgroundUpsampler, build_from_state


def _checkpoint(tmp_path, scale=4, num_conv=2, num_feat=8, key='params_ema'):
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=scale)
    path = tmp_path / f'net_x{scale}.pth'
    torch.save({key: net.state_dict()}, path)
    return str(path)


@pytest.mark.parametrize('scale', [2, 4])
def test_scale_and_depth_are_read_from_the_weights(tmp_path, scale):
    """The scale must come from the checkpoint, so it cannot disagree with the weights it was trained at."""
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=3, upscale=scale)
    built, read_scale = build_from_state(net.state_dict())
    assert read_scale == scale
    assert len(built.body) == len(net.body)


def test_enhance_upscales_by_the_checkpoint_scale(tmp_path):
    up = SRBackgroundUpsampler(_checkpoint(tmp_path, scale=4), device=torch.device('cpu'))
    out, _ = up.enhance(np.zeros((16, 24, 3), np.uint8))
    assert up.scale == 4
    assert out.shape == (64, 96, 3)
    assert out.dtype == np.uint8


def test_outscale_overrides_the_output_size(tmp_path):
    """GFPGANer asks for its own upscale, which need not equal the checkpoint's."""
    up = SRBackgroundUpsampler(_checkpoint(tmp_path, scale=4), device=torch.device('cpu'))
    out, _ = up.enhance(np.zeros((16, 24, 3), np.uint8), outscale=2)
    assert out.shape == (32, 48, 3)


def test_params_is_used_when_there_is_no_ema(tmp_path):
    up = SRBackgroundUpsampler(_checkpoint(tmp_path, scale=2, key='params'), device=torch.device('cpu'))
    assert up.scale == 2


def test_a_checkpoint_of_another_architecture_is_refused(tmp_path):
    """Loading an unrelated checkpoint must fail loudly rather than produce a silently wrong network."""
    path = tmp_path / 'other.pth'
    torch.save({'params': {'conv_first.weight': torch.zeros(4, 3, 3, 3)}}, path)
    with pytest.raises(ValueError, match='SRVGGNetCompact'):
        SRBackgroundUpsampler(str(path), device=torch.device('cpu'))
