import numpy as np
import pytest

from gfpgan.component_boxes import component_boxes_from_landmarks, flip_component_boxes, square_box


def test_square_box():
    points = np.array([[100, 200], [140, 210]], np.float32)  # 40 x 10 bounding box, center (120, 205)
    assert square_box(points, 1.5, 10, 100, 512) == [90.0, 175.0, 150.0, 235.0]
    # clamped to min_size and max_size
    assert square_box(points, 0.1, 32, 100, 512) == [104.0, 189.0, 136.0, 221.0]
    assert square_box(points, 10, 32, 100, 512) == [70.0, 155.0, 170.0, 255.0]
    # clipped to the image
    assert square_box(np.array([[0, 0], [20, 20]], np.float32), 2, 10, 100, 512) == [0.0, 0.0, 30.0, 30.0]


def test_component_boxes_and_flip():
    landmarks = np.array(
        [[180, 240], [220, 250], [300, 240], [340, 250], [220, 360], [300, 380]],  # left eye, right eye, mouth
        np.float32)
    indices = {'left_eye': [0, 1], 'right_eye': [2, 3], 'mouth': [4, 5]}
    boxes = component_boxes_from_landmarks(landmarks, indices, image_size=512, eye_margin=1.8, mouth_margin=1.5)
    assert boxes['left_eye'] == [164.0, 209.0, 236.0, 281.0]
    assert boxes['right_eye'] == [284.0, 209.0, 356.0, 281.0]
    assert boxes['mouth'] == [200.0, 310.0, 320.0, 430.0]

    flipped = flip_component_boxes(boxes, 512)
    assert flipped['left_eye'] == [512 - 356.0, 209.0, 512 - 284.0, 281.0]
    assert flipped['right_eye'] == [512 - 236.0, 209.0, 512 - 164.0, 281.0]
    assert flipped['mouth'] == [192.0, 310.0, 312.0, 430.0]
    assert flip_component_boxes(flipped, 512) == boxes


def test_face_mesh_landmarker_blank_image(tmp_path):
    pytest.importorskip('mediapipe')
    from gfpgan.component_boxes import FaceMeshLandmarker, contour_indices

    indices = contour_indices()
    assert set(indices) == {'left_eye', 'right_eye', 'mouth'}
    assert all(len(v) > 0 for v in indices.values())

    landmarker = FaceMeshLandmarker(model_rootpath=str(tmp_path))
    assert landmarker.detect(np.zeros((512, 512, 3), np.uint8)) is None
    landmarker.close()
