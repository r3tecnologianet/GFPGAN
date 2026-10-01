import cv2
import numpy as np
import os
import random
import torch
from basicsr.data import build_dataset

import gfpgan  # noqa: F401  (registers the dataset)
from gfpgan.data.ffhq_degradation_dataset import transform_box

BOX = [40.0, 50.0, 70.0, 80.0]


def _opt(folder, boxes_path=None, **extra):
    opt = dict(
        name='faces',
        type='FFHQDegradationDataset',
        dataroot_gt=folder,
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
    if boxes_path:
        opt.update(crop_components=True, component_path=boxes_path)
    opt.update(extra)
    return opt


def _seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _folder_with_square(tmp_path):
    """One 128x128 grey image with a white square filling BOX, and component boxes that all point at it."""
    img = np.full((128, 128, 3), 64, np.uint8)
    x1, y1, x2, y2 = [int(v) for v in BOX]
    img[y1:y2, x1:x2] = 255
    cv2.imwrite(str(tmp_path / '0000.png'), img)
    boxes_path = str(tmp_path / 'components' / 'boxes.pth')
    os.makedirs(os.path.dirname(boxes_path), exist_ok=True)
    torch.save({'0000': {'left_eye': BOX, 'right_eye': BOX, 'mouth': BOX}}, boxes_path)
    return str(tmp_path), boxes_path


def test_transform_box_identity_and_shift():
    identity = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    assert transform_box(BOX, identity, 128, 128) == BOX
    shift = np.array([[1, 0, 5], [0, 1, -3]], np.float64)
    assert transform_box(BOX, shift, 128, 128) == [45.0, 47.0, 75.0, 77.0]


def test_transform_box_is_clipped_to_the_image():
    shift = np.array([[1, 0, 100], [0, 1, 0]], np.float64)
    assert transform_box(BOX, shift, 128, 128)[2] == 128.0


def test_zero_jitter_leaves_the_ground_truth_unchanged(tmp_path):
    folder, _ = _folder_with_square(tmp_path)
    plain = build_dataset(_opt(folder))
    zero = build_dataset(_opt(folder, align_jitter=dict(rotation=0, scale=0, shift=0)))
    assert torch.allclose(plain[0]['gt'], zero[0]['gt'], atol=1e-6)


def test_jitter_moves_the_ground_truth(tmp_path):
    folder, _ = _folder_with_square(tmp_path)
    plain = build_dataset(_opt(folder))
    _seed()
    moved = build_dataset(_opt(folder, align_jitter=dict(rotation=2, scale=0.05, shift=3)))[0]['gt']
    assert not torch.allclose(plain[0]['gt'], moved, atol=1e-3)


def test_component_boxes_follow_the_jittered_image(tmp_path):
    folder, boxes_path = _folder_with_square(tmp_path)
    ds = build_dataset(_opt(folder, boxes_path, align_jitter=dict(rotation=2, scale=0.05, shift=3)))
    for seed in range(10):
        _seed(seed)
        item = ds[0]
        white = (item['gt'][0] > 0.9).nonzero().float()  # rows, cols of the square after the warp
        cy, cx = white[:, 0].mean().item(), white[:, 1].mean().item()
        x1, y1, x2, y2 = item['loc_left_eye'].tolist()
        assert x1 <= cx <= x2 and y1 <= cy <= y2
        assert abs((x1 + x2) / 2 - cx) < 1.5 and abs((y1 + y2) / 2 - cy) < 1.5
