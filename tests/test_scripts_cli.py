"""Tests for the two documented command-line scripts, both of which a user runs by hand."""
import cv2
import numpy as np
import pytest
import sys
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact

sys.path.insert(0, 'scripts')
import generate_component_boxes as boxes_script  # noqa: E402
import interpolate_sr_weights as interp_script  # noqa: E402


class _FixedLandmarker():
    """Stands in for MediaPipe Face Mesh: every point at the image centre, or nothing at all.

    The landmarker itself is covered by test_component_boxes; what matters here is the script's own loop, which
    has to skip what it cannot read and keep what it can.
    """

    closed = False

    def __init__(self, model_rootpath=None, found=True):
        self.found = found

    def detect(self, img):
        if not self.found:
            return None
        return np.full((478, 2), img.shape[0] / 2, np.float32)

    def close(self):
        type(self).closed = True


def _landmarker(found=True):
    return lambda model_rootpath=None: _FixedLandmarker(model_rootpath, found=found)


def _run_boxes(monkeypatch, tmp_path, *args, found=True):
    monkeypatch.setattr(boxes_script, 'FaceMeshLandmarker', _landmarker(found))
    out = tmp_path / 'sub' / 'component_boxes.pth'
    monkeypatch.setattr(
        sys, 'argv',
        ['generate_component_boxes.py', '-i',
         str(tmp_path / 'faces'), '-o', str(out)] + [str(a) for a in args])
    boxes_script.main()
    return out


def _write_face(folder, name, size=512, width=None):
    folder.mkdir(parents=True, exist_ok=True)
    img = np.random.RandomState(0).randint(0, 256, (size, width or size, 3)).astype(np.uint8)
    assert cv2.imwrite(str(folder / name), img)


def test_boxes_are_written_for_every_square_face(monkeypatch, tmp_path):
    for name in ('a.png', 'b.jpg'):
        _write_face(tmp_path / 'faces', name)
    out = _run_boxes(monkeypatch, tmp_path)
    saved = torch.load(out)
    assert sorted(saved) == ['a', 'b'], 'keys are names without the extension, as the dataset looks them up'
    assert sorted(saved['a']) == ['left_eye', 'mouth', 'right_eye']


def test_the_output_directory_is_created(monkeypatch, tmp_path):
    _write_face(tmp_path / 'faces', 'a.png')
    out = _run_boxes(monkeypatch, tmp_path)
    assert out.is_file()


def test_non_square_images_are_skipped(monkeypatch, tmp_path):
    """The boxes are computed in a square 512 frame, so a non-square image cannot get one."""
    _write_face(tmp_path / 'faces', 'square.png')
    _write_face(tmp_path / 'faces', 'wide.png', size=256, width=512)
    saved = torch.load(_run_boxes(monkeypatch, tmp_path))
    assert list(saved) == ['square']


def test_files_that_are_not_images_are_ignored(monkeypatch, tmp_path):
    _write_face(tmp_path / 'faces', 'a.png')
    (tmp_path / 'faces' / 'notes.txt').write_text('not an image')
    saved = torch.load(_run_boxes(monkeypatch, tmp_path))
    assert list(saved) == ['a']


def test_a_face_the_landmarker_misses_is_skipped(monkeypatch, tmp_path):
    """40 of 1,524 crops behaved this way when the licensed corpus was built, so it must not crash."""
    _write_face(tmp_path / 'faces', 'a.png')
    saved = torch.load(_run_boxes(monkeypatch, tmp_path, found=False))
    assert saved == {}


def test_the_landmarker_is_closed(monkeypatch, tmp_path):
    _write_face(tmp_path / 'faces', 'a.png')
    _FixedLandmarker.closed = False
    _run_boxes(monkeypatch, tmp_path)
    assert _FixedLandmarker.closed, 'MediaPipe holds a native handle that has to be released'


def test_min_size_widens_a_degenerate_box(monkeypatch, tmp_path):
    _write_face(tmp_path / 'faces', 'a.png')
    saved = torch.load(_run_boxes(monkeypatch, tmp_path, '--min_size', 40))
    x1, y1, x2, y2 = saved['a']['left_eye']
    assert (x2 - x1, y2 - y1) == (40.0, 40.0)


def _sr_checkpoint(path, seed, num_feat=8):
    torch.manual_seed(seed)
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=2, upscale=4)
    torch.save({'params_ema': net.state_dict()}, path)
    return str(path)


def _run_interp(monkeypatch, pixel, gan, alpha, out):
    monkeypatch.setattr(
        sys, 'argv',
        ['interpolate_sr_weights.py', '--pixel', pixel, '--gan', gan, '--alpha',
         str(alpha), '-o',
         str(out)])
    interp_script.main()


@pytest.mark.parametrize('alpha, expected', [(0.0, 'pixel'), (1.0, 'gan')])
def test_the_ends_of_the_range_write_the_original_weights(monkeypatch, tmp_path, alpha, expected):
    pixel = _sr_checkpoint(tmp_path / 'pixel.pth', seed=0)
    gan = _sr_checkpoint(tmp_path / 'gan.pth', seed=1)
    out = tmp_path / 'blended.pth'
    _run_interp(monkeypatch, pixel, gan, alpha, out)
    written = torch.load(out)['params_ema']
    source = torch.load(pixel if expected == 'pixel' else gan)['params_ema']
    for key in source:
        assert torch.allclose(written[key], source[key])


def test_the_blend_is_written_under_the_ema_key(monkeypatch, tmp_path):
    """Inference prefers params_ema, so the blended file has to use that key to be loadable."""
    out = tmp_path / 'blended.pth'
    _run_interp(monkeypatch, _sr_checkpoint(tmp_path / 'a.pth', 0), _sr_checkpoint(tmp_path / 'b.pth', 1), 0.5, out)
    assert list(torch.load(out)) == ['params_ema']


def test_the_midpoint_is_the_mean_of_the_two(monkeypatch, tmp_path):
    pixel = _sr_checkpoint(tmp_path / 'pixel.pth', seed=0)
    gan = _sr_checkpoint(tmp_path / 'gan.pth', seed=1)
    out = tmp_path / 'blended.pth'
    _run_interp(monkeypatch, pixel, gan, 0.5, out)
    written = torch.load(out)['params_ema']
    a, b = torch.load(pixel)['params_ema'], torch.load(gan)['params_ema']
    for key in a:
        assert torch.allclose(written[key], (a[key] + b[key]) / 2, atol=1e-6)


def test_an_alpha_outside_the_range_is_refused(monkeypatch, tmp_path):
    out = tmp_path / 'blended.pth'
    with pytest.raises(SystemExit, match=r'alpha must be within \[0, 1\]'):
        _run_interp(monkeypatch, _sr_checkpoint(tmp_path / 'a.pth', 0), _sr_checkpoint(tmp_path / 'b.pth', 1), 1.5, out)
    assert not out.exists(), 'nothing should be written when the arguments are refused'


def test_checkpoints_of_different_shapes_are_refused(monkeypatch, tmp_path):
    """Blending only means anything between two fine-tunes of one network."""
    pixel = _sr_checkpoint(tmp_path / 'pixel.pth', seed=0, num_feat=8)
    gan = _sr_checkpoint(tmp_path / 'gan.pth', seed=1, num_feat=16)
    with pytest.raises(ValueError, match='shape'):
        _run_interp(monkeypatch, pixel, gan, 0.5, tmp_path / 'blended.pth')
