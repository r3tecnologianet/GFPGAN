"""Blend a pixel-loss super-resolution checkpoint with its adversarial continuation.

Network interpolation (Wang et al., "Deep Network Interpolation for Continuous Imagery Effect Transition",
arXiv:1811.10515, and ESRGAN, arXiv:1809.00219) sets

    theta(alpha) = (1 - alpha) * theta_pixel + alpha * theta_gan

and picks alpha on validation. Because the two models are fine-tuned from one another their parameters stay
aligned, so the blend behaves smoothly between them: fidelity at alpha 0, texture at alpha 1. It costs no
training, which is why searching for the right number of adversarial iterations is the wrong experiment --
train both ends once, then turn the dial.

Measured on this fork's background models over 64 validation pairs with a near-identity degradation, alpha 0.5
gained 0.26 dB of PSNR over the pixel model (bootstrap 95% CI +0.185 to +0.331) at no cost on the heavily
degraded set, where the difference was inside noise. SSIM falls monotonically with alpha, so the blend is a
trade rather than a free gain; see docs/background_super_resolution.md.

Usage:
  python scripts/interpolate_sr_weights.py --pixel A.pth --gan B.pth --alpha 0.5 -o blended.pth
"""
import argparse
import torch


def load_params(path):
    """Return the parameter dict, preferring the EMA weights that inference uses."""
    state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError(f'{path} does not contain a state dict')
    for key in ('params_ema', 'params'):
        if key in state:
            return state[key]
    return state


def interpolate(pixel, gan, alpha):
    """theta(alpha) = (1 - alpha) * pixel + alpha * gan, over identically shaped parameter sets."""
    if pixel.keys() != gan.keys():
        missing = set(pixel) ^ set(gan)
        raise ValueError(f'checkpoints have different parameter sets, e.g. {sorted(missing)[:3]}')
    blended = {}
    for key in pixel:
        a, b = pixel[key], gan[key]
        if a.shape != b.shape:
            raise ValueError(f'{key} has shape {tuple(a.shape)} against {tuple(b.shape)}')
        blended[key] = ((1 - alpha) * a.float() + alpha * b.float()).to(a.dtype)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixel', required=True, help='checkpoint trained with pixel loss only')
    parser.add_argument('--gan', required=True, help='its adversarial continuation')
    parser.add_argument('--alpha', type=float, required=True, help='0 keeps the pixel model, 1 the adversarial one')
    parser.add_argument('-o', '--output', required=True, help='where to write the blended checkpoint')
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit(f'alpha must be within [0, 1], got {args.alpha}')

    blended = interpolate(load_params(args.pixel), load_params(args.gan), args.alpha)
    torch.save({'params_ema': blended}, args.output)
    print(f'alpha {args.alpha}: {args.pixel} + {args.gan} -> {args.output} ({len(blended)} tensors)')


if __name__ == '__main__':
    main()
