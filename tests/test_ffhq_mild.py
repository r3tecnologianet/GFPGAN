import cv2
import numpy as np
import pytest
import random
import torch
from basicsr.data import build_dataset

import gfpgan  # noqa: F401  (registers the dataset)


def _face_like(size=128):
    """A smooth image with edges in it.

    White noise must not be used as ground truth here. Resizing, JPEG and blur all destroy noise equally, so on
    noise the mild and heavy branches converge and the test would pass or fail for reasons unrelated to what it
    checks. The same mistake was made once already in tests/test_realesrgan_mild.py.
    """
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
    image = np.stack([0.5 + 0.4 * np.sin(6 * x), 0.5 + 0.4 * np.cos(5 * y), 0.5 + 0.3 * np.sin(4 * (x + y))], axis=2)
    image[size // 4:3 * size // 4, size // 4:3 * size // 4] *= 0.6
    return np.clip(image * 255, 0, 255).astype(np.uint8)


@pytest.fixture
def gt_folder(tmp_path):
    for i in range(4):
        cv2.imwrite(str(tmp_path / f'{i:04d}.png'), _face_like())
    return str(tmp_path)


def _dataset(gt_folder, mild_prob):
    return build_dataset(
        dict(
            name='faces',
            type='FFHQDegradationDataset',
            dataroot_gt=gt_folder,
            io_backend=dict(type='disk'),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            out_size=128,
            use_hflip=False,
            phase='train',
            scale=1,
            mild_prob=mild_prob,
            blur_kernel_size=41,
            kernel_list=['iso', 'aniso'],
            kernel_prob=[0.5, 0.5],
            blur_sigma=[0.1, 10],
            downsample_range=[0.8, 8],
            noise_range=[0, 20],
            jpeg_range=[60, 100],
            color_jitter_prob=None,
            color_jitter_shift=20,
            color_jitter_pt_prob=None,
            gray_prob=0))


def _psnr_lq_against_gt(item):
    """How close the degraded input is to the ground truth, both in [-1, 1]."""
    mse = torch.mean((item['lq'] - item['gt'])**2).item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))  # peak-to-peak of [-1, 1] is 2, so the numerator is 2^2


def _seed(value=0):
    """All three generators: basicsr draws the kernel type from the standard library's random."""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def test_mild_branch_is_really_mild(gt_folder):
    _seed()
    mild = np.mean([_psnr_lq_against_gt(_dataset(gt_folder, 1.0)[i % 4]) for i in range(8)])
    heavy = np.mean([_psnr_lq_against_gt(_dataset(gt_folder, 0.0)[i % 4]) for i in range(8)])
    assert mild > 30, f'the mild branch is not near-identity: {mild:.2f} dB'
    assert mild > heavy + 10, f'mild {mild:.2f} dB is not clearly milder than heavy {heavy:.2f} dB'


def _dataset_without_the_option(gt_folder):
    """A config that never mentions mild_prob, as every existing config does."""
    opt = _dataset(gt_folder, 0).opt.copy()
    del opt['mild_prob']
    return build_dataset(opt)


def test_a_config_without_the_option_is_unchanged(gt_folder):
    """Existing configs do not set mild_prob, and must degrade exactly as they did before."""
    absent = _dataset_without_the_option(gt_folder)
    present = _dataset(gt_folder, 0)
    assert absent.mild_prob == 0

    # random.seed matters as much as the other two: basicsr draws the blur kernel type with random.choices, from
    # the standard library generator. Seeding only numpy and torch leaves that draw to whatever global state the
    # process happens to be in, so the two datasets pick different kernels and disagree -- which is why this test
    # passed in the full suite and failed on its own.
    _seed()
    without_key = _psnr_lq_against_gt(absent[0])
    _seed()
    with_zero = _psnr_lq_against_gt(present[0])
    assert without_key == pytest.approx(with_zero, abs=1e-6)


def test_mild_prob_does_not_mutate_instance_state(gt_folder):
    """The dataset runs in worker processes, so a mild sample must not change the configured knobs."""
    ds = _dataset(gt_folder, 1.0)
    before = (list(ds.blur_sigma), list(ds.downsample_range), list(ds.noise_range), list(ds.jpeg_range),
              list(ds.kernel_list))
    ds[0]
    after = (list(ds.blur_sigma), list(ds.downsample_range), list(ds.noise_range), list(ds.jpeg_range),
             list(ds.kernel_list))
    assert before == after
