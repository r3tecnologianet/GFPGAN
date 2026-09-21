import argparse
import cv2
import glob
import numpy as np
import os
from basicsr.utils import imwrite

from gfpgan import GFPGANer
from gfpgan.bg_upsampler import SRBackgroundUpsampler
from gfpgan.utils import DEFAULT_BLEND_WEIGHT


def _blend_weight(value):
    """Reject a weight outside [0, 1] while parsing rather than inside the per-face loop.

    Left to the loop, the failure arrives after the model has been loaded and after earlier images have already
    been written, so the run aborts on the first photograph containing a face and leaves a half-filled output
    folder behind.
    """
    weight = float(value)
    if not 0.0 <= weight <= 1.0:
        raise argparse.ArgumentTypeError(f'must be between 0 and 1, got {weight}')
    return weight


def build_parser():
    """Built here rather than inside main so that a test can read what the flags default to."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image or folder.')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to licensed model weights.')
    parser.add_argument(
        '--arch', type=str, default='clean', help='Model architecture. Option: clean | RestoreFormer. Default: clean')
    parser.add_argument(
        '--channel_multiplier', type=int, default=2, help='Channel multiplier of the clean arch. Default: 2')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument(
        '-w',
        '--weight',
        type=_blend_weight,
        default=DEFAULT_BLEND_WEIGHT,
        help='How much of the restoration to keep, from 0 to 1. At 1 the output is the restored face; at 0 the '
        'aligned 512 crop passes through unchanged, which is the input itself only with --aligned, since '
        'otherwise the face is still warped to 512 and pasted back. Lower it when the input is already good, '
        'since the model damages a clean face. The default was measured on aligned crops rather than chosen: '
        'see docs/training_stability.md. Default: 0.75')
    parser.add_argument(
        '--bg_model',
        type=str,
        default=None,
        help='Path to licensed super-resolution weights for the background. Default: none, which upscales the '
        'background with Lanczos.')
    return parser


def main():
    """Inference demo for GFPGAN (for users).

    Pretrained weights are not downloaded automatically: the official release weights are trained on
    non-commercial data. Pass weights you are licensed to use with --model_path.
    """
    args = build_parser().parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up GFPGAN restorer ------------------------
    # The background is upscaled with Lanczos unless --bg_model supplies super-resolution weights. Nothing is
    # downloaded: the Real-ESRGAN release weights are trained on non-commercial data.
    bg_upsampler = None
    if args.bg_model is not None:
        bg_upsampler = SRBackgroundUpsampler(args.bg_model)
        print(f'Background super-resolution from {args.bg_model} (x{bg_upsampler.scale})')

    restorer = GFPGANer(
        model_path=args.model_path,
        upscale=args.upscale,
        arch=args.arch,
        channel_multiplier=args.channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
