"""End-to-end tests for the command line, which decides what a user actually gets on disk."""
import cv2
import numpy as np
import os
import pytest
import sys
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from conftest import face

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_gfpgan import main  # noqa: E402


@pytest.fixture
def run_cli(patch_network, checkpoint, tmp_path, monkeypatch):
    """Run main() with a doubled network, returning the output folder."""

    def _run(*args, detector='template', **kwargs):
        patch_network(detector=detector)
        out = tmp_path / 'results'
        argv = ['inference_gfpgan.py', '--model_path', checkpoint(), '-o', str(out)] + [str(a) for a in args]
        monkeypatch.setattr(sys, 'argv', argv)
        main()
        return out

    return _run


def _write(path, size=400, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), face(size, seed=seed))
    return path


def test_a_single_file_writes_every_output_kind(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.png')
    out = run_cli('-i', img)
    assert (out / 'cropped_faces' / 'portrait_00.png').is_file()
    assert (out / 'restored_faces' / 'portrait_00.png').is_file()
    assert (out / 'cmp' / 'portrait_00.png').is_file()
    assert (out / 'restored_imgs' / 'portrait.png').is_file()


def test_the_comparison_image_is_the_two_faces_side_by_side(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.png')
    out = run_cli('-i', img)
    cmp_img = cv2.imread(str(out / 'cmp' / 'portrait_00.png'))
    assert cmp_img.shape == (512, 1024, 3)


def test_a_folder_is_processed_in_order(run_cli, tmp_path):
    folder = tmp_path / 'inputs'
    for name in ('b.png', 'a.png'):
        _write(folder / name)
    out = run_cli('-i', folder)
    assert sorted(p.name for p in (out / 'restored_imgs').iterdir()) == ['a.png', 'b.png']


def test_a_trailing_slash_on_the_input_folder_is_accepted(run_cli, tmp_path):
    folder = tmp_path / 'inputs'
    _write(folder / 'a.png')
    out = run_cli('-i', f'{folder}{os.sep}')
    assert (out / 'restored_imgs' / 'a.png').is_file()


def test_suffix_renames_the_restored_outputs_only(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.png')
    out = run_cli('-i', img, '--suffix', 'v1')
    assert (out / 'restored_faces' / 'portrait_00_v1.png').is_file()
    assert (out / 'restored_imgs' / 'portrait_v1.png').is_file()
    assert (out / 'cropped_faces' / 'portrait_00.png').is_file(), 'the crop is the input, so it keeps its name'


def test_ext_auto_keeps_the_input_extension(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.jpg')
    out = run_cli('-i', img)
    assert (out / 'restored_imgs' / 'portrait.jpg').is_file()


def test_ext_overrides_the_output_format(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.jpg')
    out = run_cli('-i', img, '--ext', 'png')
    assert (out / 'restored_imgs' / 'portrait.png').is_file()
    assert not (out / 'restored_imgs' / 'portrait.jpg').exists()


def test_aligned_writes_faces_but_no_whole_image(run_cli, tmp_path):
    """With --aligned there is no background, so restored_imgs must not appear."""
    img = _write(tmp_path / 'inputs' / 'crop.png', size=512)
    out = run_cli('-i', img, '--aligned', detector='none')
    assert (out / 'restored_faces' / 'crop_00.png').is_file()
    assert not (out / 'restored_imgs').exists()


def test_upscale_sets_the_size_of_the_whole_image(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.png', size=200)
    out = run_cli('-i', img, '-s', 3)
    assert cv2.imread(str(out / 'restored_imgs' / 'portrait.png')).shape == (600, 600, 3)


def test_weight_zero_leaves_the_saved_face_equal_to_the_crop(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'crop.png', size=512)
    out = run_cli('-i', img, '--aligned', '-w', 0, detector='none')
    crop = cv2.imread(str(out / 'cropped_faces' / 'crop_00.png'))
    restored = cv2.imread(str(out / 'restored_faces' / 'crop_00.png'))
    np.testing.assert_array_equal(restored, crop)


def test_the_output_folder_is_created(run_cli, tmp_path):
    img = _write(tmp_path / 'inputs' / 'portrait.png')
    out = run_cli('-i', img)
    assert out.is_dir()


def test_bg_model_upscales_the_background_from_its_own_checkpoint(run_cli, tmp_path):
    """--bg_model reads its scale from the weights, and the face is pasted onto that background."""
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    bg_path = tmp_path / 'bg_x2.pth'
    torch.save({'params_ema': net.state_dict()}, bg_path)
    img = _write(tmp_path / 'inputs' / 'portrait.png', size=120)
    out = run_cli('-i', img, '--bg_model', bg_path, '-s', 2)
    assert cv2.imread(str(out / 'restored_imgs' / 'portrait.png')).shape == (240, 240, 3)
