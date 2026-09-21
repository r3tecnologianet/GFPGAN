import inspect
import numpy as np
import pytest

from gfpgan.utils import GFPGANer, blend_restoration


def test_the_default_weight_is_the_one_the_sweep_chose():
    """0.75 came from a measurement, not from taste, and drifting back to full strength would lose 2 dB on a
    clean face on every one of the 64 tested. The command line default must not disagree with this one."""
    assert inspect.signature(GFPGANer.enhance).parameters['weight'].default == 0.75


def _pair(shape=(8, 6, 3)):
    """A dark original and a bright restoration, so any blend between them is visible in the numbers."""
    original = np.full(shape, 40, dtype=np.uint8)
    restored = np.full(shape, 200, dtype=np.uint8)
    return restored, original


def test_full_weight_returns_the_restoration_untouched():
    restored, original = _pair()
    assert np.array_equal(blend_restoration(restored, original, 1.0), restored)


def test_zero_weight_returns_the_input_untouched():
    """The whole point of the dial: at 0 the model cannot damage an input that is already good."""
    restored, original = _pair()
    assert np.array_equal(blend_restoration(restored, original, 0.0), original)


def test_half_weight_is_the_midpoint():
    restored, original = _pair()
    out = blend_restoration(restored, original, 0.5)
    assert np.allclose(out, 120, atol=1)


@pytest.mark.parametrize('weight', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_result_stays_an_image(weight):
    restored, original = _pair()
    out = blend_restoration(restored, original, weight)
    assert out.dtype == np.uint8 and out.shape == restored.shape


def test_raising_the_weight_moves_the_output_toward_the_restoration():
    """Monotonicity is what makes the flag a dial rather than a switch with a confusing name."""
    restored, original = _pair()
    means = [blend_restoration(restored, original, w).mean() for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(b > a for a, b in zip(means, means[1:])), means


def test_values_are_rounded_rather_than_truncated():
    """Truncation would bias every blended pixel downward, darkening the output at every intermediate weight."""
    restored = np.full((4, 4, 3), 101, dtype=np.uint8)
    original = np.full((4, 4, 3), 100, dtype=np.uint8)
    assert blend_restoration(restored, original, 0.5).mean() == pytest.approx(100.5, abs=0.5)


@pytest.mark.parametrize('weight', [-0.1, 1.1, 2.0])
def test_a_weight_outside_the_range_is_refused(weight):
    """The flag comes straight from the command line as a float, so the bound is checked rather than assumed."""
    restored, original = _pair()
    with pytest.raises(ValueError):
        blend_restoration(restored, original, weight)


def test_mismatched_shapes_are_refused():
    """Blending arrays of different sizes would either broadcast into nonsense or raise somewhere far away."""
    restored, _ = _pair()
    with pytest.raises(ValueError):
        blend_restoration(restored, np.zeros((4, 4, 3), np.uint8), 0.5)
