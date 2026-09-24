"""Shared test doubles for the inference path.

``GFPGANer`` hardcodes a 512x512 generator, so building the real one for every test would cost more than the
code under test is worth. These doubles stand in for the network and the detector, which are covered on their
own in test_gfpgan_arch and test_face_helper, so that what gets exercised here is GFPGANer's own wiring: the
checkpoint key selection, the blend weight, the aligned and whole-image paths, and the paste-back.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from gfpgan import utils
from gfpgan.face_helper import FACE_TEMPLATE_512, FaceHelper


class NoDetector():
    """Finds nothing."""

    def detect(self, img):
        return np.zeros((0, 4), np.float32), np.zeros((0, ), np.float32), np.zeros((0, 6, 2), np.float32)


class TemplateDetector():
    """Reports one face whose keypoints are the alignment template, scaled and shifted.

    The first four keypoints are the ones ``MEDIAPIPE_KEYPOINT_INDICES`` selects, so this drives the real
    alignment and paste-back without depending on MediaPipe finding a face in a synthetic image.
    """

    def __init__(self, scale=0.5, offset=(20, 40)):
        self.landmark = FACE_TEMPLATE_512 * scale + np.array(offset, np.float32)

    def detect(self, img):
        kps = np.zeros((1, 6, 2), np.float32)
        kps[0, :4] = self.landmark
        x1, y1 = self.landmark.min(axis=0)
        x2, y2 = self.landmark.max(axis=0)
        return np.array([[x1, y1, x2, y2]], np.float32), np.array([0.9], np.float32), kps


class ConstantNet(nn.Module):
    """Returns a constant image in [-1, 1], which makes the blend and the loaded weights observable as pixels."""

    def __init__(self, value=0.5, **kwargs):
        super(ConstantNet, self).__init__()
        self.bias = nn.Parameter(torch.tensor([float(value)]))

    def forward(self, x, return_rgb=True, **kwargs):
        return self.bias.clamp(-1, 1).expand_as(x).contiguous(), None


class FailingNet(ConstantNet):
    """Raises the error that ``enhance`` is written to survive."""

    def forward(self, x, return_rgb=True, **kwargs):
        raise RuntimeError('out of memory')


class RecordingUpsampler():
    """Background upsampler that records the outscale GFPGANer asked for."""

    def __init__(self, scale=2):
        self.scale = scale
        self.calls = []

    def enhance(self, img, outscale=None):
        self.calls.append(outscale)
        h, w = img.shape[:2]
        return np.full((h * outscale, w * outscale, 3), 7, np.uint8), None


NETS = {'constant': ConstantNet, 'failing': FailingNet}
DETECTORS = {'none': NoDetector, 'template': TemplateDetector}


def pixel(value):
    """The uint8 a constant network output becomes, since enhance denormalises from [-1, 1]."""
    return round((value + 1) / 2 * 255)


def face(size=200, seed=None):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (size, size, 3)).astype(np.uint8)


@pytest.fixture
def checkpoint(tmp_path):
    """Write a checkpoint holding one state dict per key, each with a distinct constant."""

    def _write(name='net.pth', **values):
        state = {key: ConstantNet(value=value).state_dict() for key, value in (values or {'params_ema': 0.5}).items()}
        path = tmp_path / name
        torch.save(state, path)
        return str(path)

    return _write


@pytest.fixture
def patch_network(monkeypatch):
    """Replace the generator and the detector that GFPGANer builds for itself."""

    def _patch(net='constant', detector='none'):
        monkeypatch.setattr(utils, 'GFPGANv1Clean', NETS[net])
        face_det = DETECTORS[detector]()
        monkeypatch.setattr(
            utils,
            'FaceHelper',
            lambda upscale_factor, face_size=512, det_model=None, model_rootpath=None: FaceHelper(
                upscale_factor, face_size=face_size, face_det=face_det))
        return face_det

    return _patch


@pytest.fixture
def make_restorer(checkpoint, patch_network):
    """Build a GFPGANer whose network and detector are doubles, on the CPU."""

    def _make(net='constant', detector='none', upscale=2, bg_upsampler=None, **values):
        patch_network(net=net, detector=detector)
        return utils.GFPGANer(
            model_path=checkpoint(**values), upscale=upscale, bg_upsampler=bg_upsampler, device=torch.device('cpu'))

    return _make
