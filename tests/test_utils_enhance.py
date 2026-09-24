import cv2
import numpy as np
import pytest
from conftest import RecordingUpsampler, face, pixel

from gfpgan.utils import DEFAULT_BLEND_WEIGHT, GFPGANer, blend_restoration


def test_unsupported_arch_is_refused_by_name():
    """The message has to name the options, since --arch is a user-facing string."""
    with pytest.raises(ValueError, match='Unsupported arch'):
        GFPGANer(model_path='never read', arch='stylegan3')


def test_params_ema_is_preferred_over_params(make_restorer):
    """A trained checkpoint carries both, and inference must take the averaged weights."""
    restorer = make_restorer(params=-1.0, params_ema=1.0)
    _, restored, _ = restorer.enhance(face(), has_aligned=True, weight=1.0)
    assert restored[0].min() == restored[0].max() == pixel(1.0)


def test_params_is_used_when_there_is_no_ema(make_restorer):
    restorer = make_restorer(params=-1.0)
    _, restored, _ = restorer.enhance(face(), has_aligned=True, weight=1.0)
    assert restored[0].min() == restored[0].max() == pixel(-1.0)


def test_aligned_input_is_resized_to_the_face_size(make_restorer):
    cropped, restored, pasted = make_restorer().enhance(face(300), has_aligned=True)
    assert len(cropped) == len(restored) == 1
    assert cropped[0].shape == restored[0].shape == (512, 512, 3)
    assert pasted is None, 'an aligned input has no background to paste onto'


def test_weight_zero_returns_the_aligned_input_untouched(make_restorer):
    """Bit-exact, because this is the claim -w 0 makes to a user passing --aligned."""
    img = face(300, seed=0)
    _, restored, _ = make_restorer().enhance(img, has_aligned=True, weight=0.0)
    np.testing.assert_array_equal(restored[0], cv2.resize(img, (512, 512)))


def test_weight_one_returns_the_network_output(make_restorer):
    restorer = make_restorer(params_ema=0.25)
    _, restored, _ = restorer.enhance(face(), has_aligned=True, weight=1.0)
    assert restored[0].min() == restored[0].max() == pixel(0.25)


def test_the_default_weight_is_the_shared_constant(make_restorer):
    """Calling enhance without a weight must equal blending at DEFAULT_BLEND_WEIGHT, not at full strength."""
    img = face(512, seed=1)
    restorer = make_restorer(params_ema=0.25)
    _, restored, _ = restorer.enhance(img, has_aligned=True)
    network_output = np.full((512, 512, 3), pixel(0.25), np.uint8)
    expected = blend_restoration(network_output, cv2.resize(img, (512, 512)), DEFAULT_BLEND_WEIGHT)
    np.testing.assert_array_equal(restored[0], expected)


def test_whole_image_path_detects_aligns_and_pastes_back(make_restorer):
    restorer = make_restorer(detector='template', upscale=2)
    cropped, restored, pasted = restorer.enhance(face(400, seed=2))
    assert len(cropped) == len(restored) == 1
    assert cropped[0].shape == (512, 512, 3)
    assert pasted.shape == (800, 800, 3), 'the pasted image follows upscale, not the crop size'
    assert pasted.dtype == np.uint8


def test_no_face_found_still_returns_an_upscaled_image(make_restorer):
    """A photograph with no face must come back upscaled rather than raise."""
    cropped, restored, pasted = make_restorer(upscale=2).enhance(np.zeros((120, 80, 3), np.uint8))
    assert cropped == [] and restored == []
    assert pasted.shape == (240, 160, 3)


def test_paste_back_false_skips_the_background(make_restorer):
    restorer = make_restorer(detector='template')
    _, restored, pasted = restorer.enhance(np.zeros((400, 300, 3), np.uint8), paste_back=False)
    assert len(restored) == 1
    assert pasted is None


def test_background_upsampler_is_asked_for_the_restorer_upscale(make_restorer):
    """GFPGANer's own upscale wins: the background must match the size the faces are pasted at."""
    up = RecordingUpsampler()
    restorer = make_restorer(detector='template', upscale=3, bg_upsampler=up)
    _, _, pasted = restorer.enhance(np.zeros((120, 90, 3), np.uint8))
    assert up.calls == [3]
    assert pasted.shape == (360, 270, 3)


def test_a_failing_network_falls_back_to_the_input_face(make_restorer):
    """Inference must not abandon a whole folder because one face ran out of memory."""
    restorer = make_restorer(net='failing')
    cropped, restored, _ = restorer.enhance(face(512, seed=3), has_aligned=True)
    np.testing.assert_array_equal(restored[0], cropped[0])


def test_state_does_not_leak_between_calls(make_restorer):
    """The CLI calls enhance once per image, so each call has to clear the previous face."""
    restorer = make_restorer(detector='template')
    img = face(400, seed=4)
    for _ in range(3):
        cropped, restored, _ = restorer.enhance(img)
        assert len(cropped) == 1 and len(restored) == 1


def test_a_float_image_is_accepted(make_restorer):
    """The blend needs uint8; enhance coerces, so floats from a caller must not fail inside OpenCV."""
    _, restored, _ = make_restorer().enhance(face(512).astype(np.float32), has_aligned=True, weight=0.5)
    assert restored[0].dtype == np.uint8
