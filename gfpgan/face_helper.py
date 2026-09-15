import cv2
import numpy as np
from basicsr.utils.download_util import load_file_from_url

# MediaPipe face detector models (Apache License 2.0 model cards)
MEDIAPIPE_FACE_MODELS = {
    'blaze_face_short_range': ('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/'
                               'float16/latest/blaze_face_short_range.tflite'),
    'blaze_face_full_range': ('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/'
                              'float16/latest/blaze_face_full_range.tflite'),
}

# Alignment template for a 512x512 face, in the observer's view: left eye, right eye, nose tip, mouth center.
# Derived from the 5-point template of facexlib (MIT License, https://github.com/xinntao/facexlib);
# the mouth center is the mean of its two mouth corners.
FACE_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [257.17511, 371.28081]], dtype=np.float32)

# indices of (observer's left eye, observer's right eye, nose tip, mouth center) in MediaPipe keypoints
MEDIAPIPE_KEYPOINT_INDICES = (0, 1, 2, 3)


def _window_positions(length, side, stride):
    if length <= side:
        return [0]
    positions = list(range(0, length - side + 1, stride))
    if positions[-1] != length - side:
        positions.append(length - side)
    return positions


def _nms(boxes, scores, iou_threshold):
    """Non-maximum suppression. boxes: (N, 4) as x1, y1, x2, y2."""
    order = np.argsort(scores)[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[1:][iou <= iou_threshold]
    return keep


class MediaPipeFaceDetector():
    """Face detector based on MediaPipe BlazeFace, with optional sliding-window passes for small faces.

    Args:
        model_name (str): Key of ``MEDIAPIPE_FACE_MODELS``. Default: blaze_face_short_range.
        model_rootpath (str): Folder where the model file is downloaded.
        min_score (float): Minimum detection confidence. Default: 0.5.
        tile_scales (tuple[float]): Window side as a fraction of the shorter image side for the extra passes.
            Empty tuple disables them. Default: (0.5, ).
        min_tile_size (int): Windows smaller than this are skipped. Default: 128.
        iou_threshold (float): IoU threshold to merge detections across passes. Default: 0.4.
    """

    def __init__(self,
                 model_name='blaze_face_short_range',
                 model_rootpath=None,
                 min_score=0.5,
                 tile_scales=(0.5, ),
                 min_tile_size=128,
                 iou_threshold=0.4):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode

        model_path = load_file_from_url(
            url=MEDIAPIPE_FACE_MODELS[model_name], model_dir=model_rootpath, progress=True, file_name=None)
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_score)
        self.detector = FaceDetector.create_from_options(options)
        self.tile_scales = tile_scales
        self.min_tile_size = min_tile_size
        self.iou_threshold = iou_threshold

    def _detect_window(self, rgb, x0, y0):
        import mediapipe as mp

        h, w = rgb.shape[0:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        results = []
        for detection in self.detector.detect(image).detections:
            bbox = detection.bounding_box
            box = [
                x0 + bbox.origin_x, y0 + bbox.origin_y, x0 + bbox.origin_x + bbox.width,
                y0 + bbox.origin_y + bbox.height
            ]
            keypoints = [[x0 + kp.x * w, y0 + kp.y * h] for kp in detection.keypoints]
            results.append((box, detection.categories[0].score, keypoints))
        return results

    def detect(self, img):
        """Detect faces.

        Args:
            img (ndarray): BGR uint8 image, (h, w, 3).

        Returns:
            tuple: boxes (N, 4) as x1, y1, x2, y2; scores (N, ); keypoints (N, K, 2) in pixels.
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[0:2]
        results = self._detect_window(rgb, 0, 0)
        for scale in self.tile_scales:
            side = int(min(h, w) * scale)
            if side < self.min_tile_size:
                continue
            stride = side // 2
            for y in _window_positions(h, side, stride):
                for x in _window_positions(w, side, stride):
                    results += self._detect_window(rgb[y:y + side, x:x + side], x, y)

        if len(results) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0, ), np.float32), np.zeros((0, 6, 2), np.float32)
        boxes = np.array([r[0] for r in results], dtype=np.float32)
        scores = np.array([r[1] for r in results], dtype=np.float32)
        keypoints = np.array([r[2] for r in results], dtype=np.float32)
        keep = _nms(boxes, scores, self.iou_threshold)
        return boxes[keep], scores[keep], keypoints[keep]

    def close(self):
        self.detector.close()


class FaceHelper():
    """Detect, align, and paste back faces for restoration.

    Detection uses MediaPipe BlazeFace. Alignment is a similarity transform from four keypoints (eyes, nose tip,
    mouth center) to ``FACE_TEMPLATE_512``. Paste-back uses a feathered square mask.

    Args:
        upscale_factor (int): Upscale factor of the final output.
        face_size (int): Size of the aligned faces. Default: 512.
        det_model (str): Key of ``MEDIAPIPE_FACE_MODELS``. Default: blaze_face_short_range.
        model_rootpath (str): Folder where the detector model is downloaded.
        tile_scales (tuple[float]): See ``MediaPipeFaceDetector``. Default: (0.5, 0.25).
        min_score (float): Minimum detection confidence. Default: 0.6.
        face_det (object): Detector with a ``detect(img)`` method returning (boxes, scores, keypoints).
            If None, a ``MediaPipeFaceDetector`` is created. Default: None.
    """

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 det_model='blaze_face_short_range',
                 model_rootpath=None,
                 tile_scales=(0.5, 0.25),
                 min_score=0.6,
                 face_det=None):
        self.upscale_factor = upscale_factor
        self.face_size = (face_size, face_size)
        self.face_template = FACE_TEMPLATE_512 * (face_size / 512.0)
        if face_det is None:
            face_det = MediaPipeFaceDetector(
                model_name=det_model, model_rootpath=model_rootpath, min_score=min_score, tile_scales=tile_scales)
        self.face_det = face_det
        self.clean_all()

    def clean_all(self):
        self.all_landmarks = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []

    def read_image(self, img):
        """img: BGR image loaded by cv2 (uint8 or uint16; gray, BGR or BGRA)."""
        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # drop alpha channel
            img = img[:, :, 0:3]
        self.input_img = img

    def get_face_landmarks(self, only_keep_largest=False, only_center_face=False, eye_dist_threshold=None):
        """Detect faces and keep their alignment keypoints. Returns the number of faces kept."""
        boxes, scores, keypoints = self.face_det.detect(np.ascontiguousarray(self.input_img, dtype=np.uint8))
        for box, score, kps in zip(boxes, scores, keypoints):
            landmark = kps[list(MEDIAPIPE_KEYPOINT_INDICES)]
            # skip side faces and faces that are too small
            if eye_dist_threshold is not None and np.linalg.norm(landmark[0] - landmark[1]) < eye_dist_threshold:
                continue
            self.all_landmarks.append(landmark)
            self.det_faces.append(np.append(box, score))
        if len(self.det_faces) == 0:
            return 0

        h, w = self.input_img.shape[0:2]
        if only_keep_largest:
            areas = [(f[2] - f[0]) * (f[3] - f[1]) for f in self.det_faces]
            idx = int(np.argmax(areas))
        elif only_center_face:
            centers = [np.array([(f[0] + f[2]) / 2, (f[1] + f[3]) / 2]) for f in self.det_faces]
            idx = int(np.argmin([np.linalg.norm(c - np.array([w / 2, h / 2])) for c in centers]))
        else:
            return len(self.all_landmarks)
        self.det_faces = [self.det_faces[idx]]
        self.all_landmarks = [self.all_landmarks[idx]]
        return 1

    def align_warp_face(self):
        """Align and warp faces to the face template."""
        for landmark in self.all_landmarks:
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            if affine_matrix is None:
                continue
            self.affine_matrices.append(affine_matrix)
            cropped_face = cv2.warpAffine(
                self.input_img,
                affine_matrix,
                self.face_size,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(135, 133, 132))
            self.cropped_faces.append(cropped_face)

    def get_inverse_affine(self):
        for affine_matrix in self.affine_matrices:
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, upsample_img=None):
        h, w = self.input_img.shape[0:2]
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        if upsample_img is None:
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # half-pixel offset for more precise back alignment when upscaling
            if self.upscale_factor > 1:
                inverse_affine[:, 2] += 0.5 * self.upscale_factor
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

            # feathered square mask
            mask = np.ones(self.face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            border = max(int(2 * self.upscale_factor), 1)
            inv_mask_erosion = cv2.erode(inv_mask, np.ones((border, border), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            edge = max(int(np.sum(inv_mask_erosion)**0.5) // 20, 1)
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((edge * 2, edge * 2), np.uint8))
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (edge * 2 + 1, edge * 2 + 1), 0)[:, :, None]

            if len(upsample_img.shape) == 2:
                upsample_img = upsample_img[:, :, None]
            if upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            return upsample_img.astype(np.uint16)
        return upsample_img.astype(np.uint8)
