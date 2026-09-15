import cv2
import numpy as np
from basicsr.utils.download_util import load_file_from_url

# MediaPipe Face Landmarker (Face Mesh V2, Apache License 2.0 model card)
FACE_LANDMARKER_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/'
                       'face_landmarker.task')

COMPONENTS = ('left_eye', 'right_eye', 'mouth')


def contour_indices():
    """Face Mesh landmark indices of each facial component contour, in the observer's view.

    MediaPipe names the eyes from the subject's view, so the observer's left eye is ``FACEMESH_RIGHT_EYE``.
    """
    import mediapipe as mp

    connections = mp.solutions.face_mesh_connections

    def _indices(edges):
        return sorted({i for edge in edges for i in edge})

    return {
        'left_eye': _indices(connections.FACEMESH_RIGHT_EYE),
        'right_eye': _indices(connections.FACEMESH_LEFT_EYE),
        'mouth': _indices(connections.FACEMESH_LIPS)
    }


def square_box(points, margin, min_size, max_size, image_size):
    """Square box around points.

    The center is the midpoint of the axis-aligned bounding box of the points. The side is the longer bounding box
    side times ``margin``, clamped to [min_size, max_size]. The box is clipped to the image.

    Returns:
        list[float]: x1, y1, x2, y2.
    """
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half = float(np.clip(max(x2 - x1, y2 - y1) * margin, min_size, max_size)) / 2
    return [
        float(np.clip(center_x - half, 0, image_size)),
        float(np.clip(center_y - half, 0, image_size)),
        float(np.clip(center_x + half, 0, image_size)),
        float(np.clip(center_y + half, 0, image_size))
    ]


def component_boxes_from_landmarks(landmarks,
                                   indices,
                                   image_size=512,
                                   eye_margin=1.8,
                                   mouth_margin=1.1,
                                   min_size=32,
                                   max_size=192):
    """Facial component boxes (observer's view) from face landmarks of an aligned face.

    Args:
        landmarks (ndarray): Landmarks in pixels, (N, 2).
        indices (dict): Landmark indices of each component, as returned by ``contour_indices``.
        image_size (int): Size of the square aligned face. Default: 512.
        eye_margin (float): Box side multiplier for the eyes. Default: 1.8.
        mouth_margin (float): Box side multiplier for the mouth. Default: 1.1.
        min_size (float): Minimum box side in pixels. Default: 32.
        max_size (float): Maximum box side in pixels. Default: 192.

    Returns:
        dict: ``left_eye``, ``right_eye`` and ``mouth`` boxes as [x1, y1, x2, y2].
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    margins = {'left_eye': eye_margin, 'right_eye': eye_margin, 'mouth': mouth_margin}
    return {
        name: square_box(landmarks[indices[name]], margins[name], min_size, max_size, image_size)
        for name in COMPONENTS
    }


def flip_component_boxes(boxes, width):
    """Component boxes after a horizontal flip of an image with the given width."""

    def _flip(box):
        return [width - box[2], box[1], width - box[0], box[3]]

    return {
        'left_eye': _flip(boxes['right_eye']),
        'right_eye': _flip(boxes['left_eye']),
        'mouth': _flip(boxes['mouth'])
    }


class FaceMeshLandmarker():
    """MediaPipe Face Landmarker for a single face.

    Args:
        model_rootpath (str): Folder where the model file is downloaded.
    """

    def __init__(self, model_rootpath=None):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        model_path = load_file_from_url(
            url=FACE_LANDMARKER_URL, model_dir=model_rootpath, progress=True, file_name=None)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path), running_mode=RunningMode.IMAGE, num_faces=1)
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, img):
        """Detect landmarks.

        Args:
            img (ndarray): BGR uint8 image, (h, w, 3).

        Returns:
            ndarray | None: Landmarks in pixels, (478, 2), or None if no face is found.
        """
        import mediapipe as mp

        h, w = img.shape[0:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self.landmarker.close()
