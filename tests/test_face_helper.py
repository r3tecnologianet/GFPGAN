import numpy as np
import pytest

from gfpgan.face_helper import FACE_TEMPLATE_512, FaceHelper, _nms, _window_positions


class _NoDetector():

    def detect(self, img):
        return np.zeros((0, 4), np.float32), np.zeros((0, ), np.float32), np.zeros((0, 6, 2), np.float32)


def test_window_positions():
    assert _window_positions(100, 200, 100) == [0]
    assert _window_positions(500, 200, 100) == [0, 100, 200, 300]
    assert _window_positions(450, 200, 100) == [0, 100, 200, 250]


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert _nms(boxes, scores, 0.4) == [0, 2]


def test_face_helper_align_and_paste():
    helper = FaceHelper(upscale_factor=2, face_det=_NoDetector())
    img = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
    helper.read_image(img)
    assert helper.get_face_landmarks() == 0

    # keypoints of a face that is the template scaled by 0.5 and shifted
    landmark = FACE_TEMPLATE_512 * 0.5 + np.array([20, 40], np.float32)
    helper.all_landmarks = [landmark]
    helper.align_warp_face()
    assert len(helper.cropped_faces) == 1
    assert helper.cropped_faces[0].shape == (512, 512, 3)
    # the affine matrix maps the keypoints onto the template
    affine = helper.affine_matrices[0]
    mapped = landmark @ affine[:, :2].T + affine[:, 2]
    np.testing.assert_allclose(mapped, FACE_TEMPLATE_512, atol=1e-2)

    helper.get_inverse_affine()
    helper.add_restored_face(helper.cropped_faces[0])
    out = helper.paste_faces_to_input_image()
    assert out.shape == (800, 600, 3)
    assert out.dtype == np.uint8


def test_mediapipe_detector_blank_image(tmp_path):
    pytest.importorskip('mediapipe')
    from gfpgan.face_helper import MediaPipeFaceDetector

    detector = MediaPipeFaceDetector(model_rootpath=str(tmp_path))
    boxes, scores, keypoints = detector.detect(np.zeros((480, 640, 3), np.uint8))
    detector.close()
    assert boxes.shape == (0, 4)
    assert scores.shape == (0, )
    assert keypoints.shape == (0, 6, 2)
