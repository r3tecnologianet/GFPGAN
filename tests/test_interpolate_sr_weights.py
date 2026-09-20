import pytest
import sys
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact

sys.path.insert(0, 'scripts')
from interpolate_sr_weights import interpolate, load_params  # noqa: E402


def _params(seed, num_feat=8):
    torch.manual_seed(seed)
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=2, upscale=4).state_dict()


def test_the_ends_of_the_range_return_the_originals():
    a, b = _params(0), _params(1)
    at_zero, at_one = interpolate(a, b, 0.0), interpolate(a, b, 1.0)
    for key in a:
        assert torch.allclose(at_zero[key], a[key])
        assert torch.allclose(at_one[key], b[key])


def test_the_midpoint_is_the_mean():
    a, b = _params(0), _params(1)
    mid = interpolate(a, b, 0.5)
    for key in a:
        assert torch.allclose(mid[key], (a[key] + b[key]) / 2, atol=1e-6)


def test_mismatched_shapes_are_refused():
    """Blending checkpoints of different widths would silently produce a broken network."""
    with pytest.raises(ValueError, match='shape'):
        interpolate(_params(0, num_feat=8), _params(1, num_feat=16), 0.5)


def test_mismatched_parameter_sets_are_refused():
    a, b = _params(0), _params(1)
    b.pop(next(iter(b)))
    with pytest.raises(ValueError, match='different parameter sets'):
        interpolate(a, b, 0.5)


def test_ema_weights_are_preferred_over_params(tmp_path):
    """Inference uses params_ema when it exists, so the blend must start from the same weights."""
    ema, plain = _params(0), _params(1)
    path = tmp_path / 'both.pth'
    torch.save({'params': plain, 'params_ema': ema}, path)
    loaded = load_params(str(path))
    key = next(iter(ema))
    assert torch.allclose(loaded[key], ema[key])
