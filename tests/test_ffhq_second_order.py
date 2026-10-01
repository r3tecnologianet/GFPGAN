import cv2
import numpy as np
import pytest
import random
import torch
from basicsr.data import build_dataset

import gfpgan  # noqa: F401  (registers the dataset)


def _face_like(size=128):
    """A smooth image with edges; white noise would make every degradation look alike (see test_ffhq_mild.py)."""
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
    image = np.stack([0.5 + 0.4 * np.sin(6 * x), 0.5 + 0.4 * np.cos(5 * y), 0.5 + 0.3 * np.sin(4 * (x + y))], axis=2)
    image[size // 4:3 * size // 4, size // 4:3 * size // 4] *= 0.6
    return np.clip(image * 255, 0, 255).astype(np.uint8)


@pytest.fixture
def gt_folder(tmp_path):
    for i in range(4):
        cv2.imwrite(str(tmp_path / f'{i:04d}.png'), _face_like())
    return str(tmp_path)


def _opt(gt_folder, **extra):
    opt = dict(
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
        gray_prob=0)
    opt.update(extra)
    return opt


def _psnr(item):
    mse = torch.mean((item['lq'] - item['gt'])**2).item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


def _seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _mean_psnr(ds, n=16):
    _seed()
    return np.mean([_psnr(ds[i % len(ds)]) for i in range(n)])


def test_a_config_without_the_option_is_unchanged(gt_folder):
    absent = build_dataset(_opt(gt_folder))
    zero = build_dataset(_opt(gt_folder, second_order_prob=0))
    assert absent.second_order_prob == 0
    _seed()
    a = absent[0]['lq']
    _seed()
    b = zero[0]['lq']
    assert torch.equal(a, b)


def test_the_second_pass_degrades_further(gt_folder):
    first = _mean_psnr(build_dataset(_opt(gt_folder, second_order_prob=0)))
    second = _mean_psnr(build_dataset(_opt(gt_folder, second_order_prob=1)))
    assert second < first - 0.5, f'second order {second:.2f} dB is not below first order {first:.2f} dB'


def test_a_mild_sample_skips_the_second_pass(gt_folder):
    both = _mean_psnr(build_dataset(_opt(gt_folder, mild_prob=1.0, second_order_prob=1)))
    # Measured means: 41.1 dB with the `not mild` guard, 29.8 dB without it; 35 is midway.
    assert both > 35, f'a mild sample was degraded twice: {both:.2f} dB'


def test_the_total_downsampling_stays_within_range(gt_folder):
    ds = build_dataset(_opt(gt_folder, second_order_prob=1))
    _seed()
    for _ in range(50):
        out = ds._second_pass(np.full((16, 16, 3), 0.5, np.float32), 128, 128)  # already at the 8x limit
        assert out.shape[0] >= 16 and out.shape[1] >= 16


def test_the_second_pass_does_not_mutate_instance_state(gt_folder):
    ds = build_dataset(_opt(gt_folder, second_order_prob=1))
    before = (list(ds.blur_sigma), list(ds.downsample_range), list(ds.noise_range), list(ds.jpeg_range))
    ds[0]
    assert before == (list(ds.blur_sigma), list(ds.downsample_range), list(ds.noise_range), list(ds.jpeg_range))
