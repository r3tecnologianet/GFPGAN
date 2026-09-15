import argparse
import cv2
import glob
import os
import torch

from gfpgan.component_boxes import FaceMeshLandmarker, component_boxes_from_landmarks, contour_indices

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def main():
    """Generate facial component boxes (left eye, right eye, mouth) for aligned faces.

    The output is a dict {image name without extension: {'left_eye': [x1, y1, x2, y2], ...}} saved with torch.save,
    used by FFHQDegradationDataset with crop_components=True and component_path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Folder with aligned square face images.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output .pth file.')
    parser.add_argument('--model_rootpath', type=str, default='gfpgan/weights', help='Folder for the landmarker model.')
    parser.add_argument('--eye_margin', type=float, default=1.8)
    parser.add_argument('--mouth_margin', type=float, default=1.1)
    parser.add_argument('--min_size', type=float, default=32)
    parser.add_argument('--max_size', type=float, default=192)
    args = parser.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.input, '*')) if p.lower().endswith(IMG_EXTENSIONS))
    landmarker = FaceMeshLandmarker(model_rootpath=args.model_rootpath)
    indices = contour_indices()

    boxes, failed = {}, []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != img.shape[1]:
            failed.append(name)
            continue
        landmarks = landmarker.detect(img)
        if landmarks is None:
            failed.append(name)
            continue
        boxes[name] = component_boxes_from_landmarks(
            landmarks,
            indices,
            image_size=img.shape[0],
            eye_margin=args.eye_margin,
            mouth_margin=args.mouth_margin,
            min_size=args.min_size,
            max_size=args.max_size)
    landmarker.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(boxes, args.output)
    print(f'Saved boxes for {len(boxes)} of {len(paths)} images to {args.output}.')
    if failed:
        print(f'No boxes (unreadable, not square or no face found): {", ".join(failed)}')


if __name__ == '__main__':
    main()
