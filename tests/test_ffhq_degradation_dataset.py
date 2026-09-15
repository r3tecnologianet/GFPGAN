import cv2
import numpy as np
import os
import pytest
import torch
import yaml

from gfpgan.data.ffhq_degradation_dataset import FFHQDegradationDataset


def test_ffhq_degradation_dataset(tmp_path):

    with open('tests/data/test_ffhq_degradation_dataset.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    # synthetic gt image, so the test does not depend on face datasets
    gt_path = os.path.join(str(tmp_path), '00000000.png')
    cv2.imwrite(gt_path, np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
    opt['dataroot_gt'] = str(tmp_path)

    dataset = FFHQDegradationDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 1  # whether to read correct meta info
    assert dataset.kernel_list == ['iso', 'aniso']  # correct initialization the degradation configurations
    assert dataset.color_jitter_prob == 1

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'lq', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 512, 512)
    assert result['lq'].shape == (3, 512, 512)
    assert result['gt_path'] == gt_path

    # ------------------ test with probability = 0 -------------------- #
    opt['color_jitter_prob'] = 0
    opt['color_jitter_pt_prob'] = 0
    opt['gray_prob'] = 0
    opt['io_backend'] = dict(type='disk')
    dataset = FFHQDegradationDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 1  # whether to read correct meta info
    assert dataset.kernel_list == ['iso', 'aniso']  # correct initialization the degradation configurations
    assert dataset.color_jitter_prob == 0

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'lq', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 512, 512)
    assert result['lq'].shape == (3, 512, 512)
    assert result['gt_path'] == gt_path

    # ------------------ test with crop_components -------------------- #
    boxes = {
        'left_eye': [164.0, 209.0, 236.0, 281.0],
        'right_eye': [284.0, 209.0, 356.0, 281.0],
        'mouth': [200.0, 310.0, 320.0, 430.0]
    }
    component_path = os.path.join(str(tmp_path), 'components', 'boxes.pth')
    os.makedirs(os.path.dirname(component_path))
    torch.save({'00000000': boxes}, component_path)
    opt['crop_components'] = True
    opt['component_path'] = component_path

    for use_hflip, expected_left_eye in ((False, boxes['left_eye']), (True, [156.0, 209.0, 228.0, 281.0])):
        opt['use_hflip'] = use_hflip
        opt['io_backend'] = dict(type='disk')  # the file client pops 'type', so use a new dict for each dataset
        dataset = FFHQDegradationDataset(opt)
        if use_hflip:  # force the flip
            dataset.opt['use_hflip'] = True
            np.random.seed(0)
            result = None
            for _ in range(20):
                result = dataset.__getitem__(0)
                if result['loc_left_eye'].tolist() != boxes['left_eye']:
                    break
        else:
            result = dataset.__getitem__(0)
        expected_keys = ['gt', 'lq', 'gt_path', 'loc_left_eye', 'loc_right_eye', 'loc_mouth']
        assert set(expected_keys).issubset(set(result.keys()))
        assert result['loc_left_eye'].shape == (4, )
        assert result['loc_right_eye'].shape == (4, )
        assert result['loc_mouth'].shape == (4, )
        assert result['loc_left_eye'].tolist() == expected_left_eye
    opt['crop_components'] = False

    # ------------------ lmdb backend should have paths ends with lmdb -------------------- #
    with pytest.raises(ValueError):
        opt['dataroot_gt'] = str(tmp_path)
        opt['io_backend'] = dict(type='lmdb')
        dataset = FFHQDegradationDataset(opt)
