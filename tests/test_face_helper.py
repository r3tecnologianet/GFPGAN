import numpy as np
import pytest

from gfpgan.face_helper import (FACE_TEMPLATE_512, LANDMARKER_LEFT_EYE, LANDMARKER_LIPS, LANDMARKER_RIGHT_EYE,
                                LANDMARKER_TEMPLATE_512, FaceHelper, _nms, _window_positions, landmarker_keypoints,
                                similarity_lstsq)


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
    helper = FaceHelper(upscale_factor=2, face_det=_NoDetector(), landmarker=None)
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


def test_mediapipe_detector_blank_image():
    pytest.importorskip('mediapipe')
    from gfpgan.face_helper import MediaPipeFaceDetector

    detector = MediaPipeFaceDetector(model_rootpath='gfpgan/weights')  # reuse the downloaded model
    boxes, scores, keypoints = detector.detect(np.zeros((480, 640, 3), np.uint8))
    detector.close()
    assert boxes.shape == (0, 4)
    assert scores.shape == (0, )
    assert keypoints.shape == (0, 6, 2)


class _OneBoxDetector():
    """One face at a fixed box; its six BlazeFace keypoints are distinct and recognisable."""

    box = np.array([200, 220, 400, 460], np.float32)

    def detect(self, img):
        kps = np.arange(12, dtype=np.float32).reshape(1, 6, 2) + 300
        return self.box[None], np.array([0.9], np.float32), kps


class _FakeLandmarker():
    """Puts the eye and lip centres at fixed crop coordinates and records the crop it was given."""

    centres = np.array([[50, 60], [150, 60], [100, 200]], np.float32)

    def __init__(self, found=True):
        self.found = found
        self.crops = []

    def detect(self, img):
        self.crops.append(img.shape)
        if not self.found:
            return None
        points = np.zeros((478, 2), np.float32)
        for centre, idx in zip(self.centres, (LANDMARKER_LEFT_EYE, LANDMARKER_RIGHT_EYE, LANDMARKER_LIPS)):
            points[list(idx)] = centre
        return points


def test_landmarker_keypoints_are_the_contour_centres():
    points = np.random.RandomState(0).rand(478, 2).astype(np.float32) * 500
    expected = [points[list(idx)].mean(0) for idx in (LANDMARKER_LEFT_EYE, LANDMARKER_RIGHT_EYE, LANDMARKER_LIPS)]
    np.testing.assert_allclose(landmarker_keypoints(points), expected, rtol=1e-6)


def test_similarity_lstsq_recovers_a_known_similarity():
    angle, scale, shift = np.radians(7), 1.3, np.array([12.0, -5.0])
    rot = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = np.array([[10, 20], [200, 25], [90, 180], [40, 300]], np.float64)
    dst = src @ rot.T + shift
    matrix = similarity_lstsq(src, dst)
    np.testing.assert_allclose(matrix[:, :2], rot, atol=1e-5)
    np.testing.assert_allclose(matrix[:, 2], shift, atol=1e-3)


def test_refined_keypoints_come_from_a_padded_crop_around_the_box():
    landmarker = _FakeLandmarker()
    helper = FaceHelper(upscale_factor=1, face_det=_OneBoxDetector(), landmarker=landmarker)
    helper.read_image(np.zeros((800, 800, 3), np.uint8))
    assert helper.get_face_landmarks() == 1
    # pad is half the longer box side (240 / 2 = 120): the crop spans x 80..520 and y 100..580
    assert landmarker.crops == [(480, 440, 3)]
    np.testing.assert_allclose(helper.all_landmarks[0], _FakeLandmarker.centres + [80, 100])


def test_without_landmarks_the_blazeface_keypoints_are_used():
    helper = FaceHelper(upscale_factor=1, face_det=_OneBoxDetector(), landmarker=_FakeLandmarker(found=False))
    helper.read_image(np.zeros((800, 800, 3), np.uint8))
    assert helper.get_face_landmarks() == 1
    expected = (np.arange(12, dtype=np.float32).reshape(6, 2) + 300)[:4]
    np.testing.assert_allclose(helper.all_landmarks[0], expected)


def test_three_keypoints_align_to_the_landmarker_template():
    helper = FaceHelper(upscale_factor=1, face_det=_NoDetector(), landmarker=None)
    helper.read_image(np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8))
    landmark = LANDMARKER_TEMPLATE_512 * 0.5 + np.array([20, 40], np.float32)
    helper.all_landmarks = [landmark]
    helper.align_warp_face()
    affine = helper.affine_matrices[0]
    np.testing.assert_allclose(landmark @ affine[:, :2].T + affine[:, 2], LANDMARKER_TEMPLATE_512, atol=1e-2)
