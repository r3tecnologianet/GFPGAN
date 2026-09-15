# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

GFPGAN is a blind face restoration method (CVPR 2021) that uses a pretrained StyleGAN2 as a prior. It is built on top of **BasicSR** (training framework, registries, losses, base model) and **facexlib** (face detection, alignment, paste-back). Much of the behavior lives in those packages, not in this repo.

## License cleanup branch

Branch `license-cleanup` removes or disables components with commercial-use restrictions. `LICENSE_CLEANUP.md` lists what was removed, what was kept with caveats and the remaining blockers. When changing code on this branch:
- Do not import `basicsr.ops.*` or `basicsr.archs.stylegan2_arch` (CUDA kernels under the Nvidia Source Code License-NC).
- Do not add code derived from DFDNet, ParseNet/PSFRGAN or other non-commercial sources.
- Do not add download URLs for the official GFPGAN, RestoreFormer, Real-ESRGAN or facexlib release weights, or face images/datasets without documented licenses.

## Setup

```bash
pip install basicsr facexlib
pip install -r requirements.txt
python setup.py develop          # also generates gfpgan/version.py from VERSION + git sha
```

basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, which was removed in torchvision 0.17, so use torchvision ≤ 0.16 (Python ≤ 3.11).

## Commands

Inference (no automatic weight download; pass weights you are licensed to use):
```bash
python inference_gfpgan.py -i <image_or_folder> -o results --model_path <weights.pth> --arch clean --channel_multiplier 2 -s 2
```
`--arch` accepts `clean | RestoreFormer`. `--aligned` skips detection for pre-cropped 512x512 faces.

Training: there are no training configs on this branch. `gfpgan/train.py` still hands off to BasicSR's `train_pipeline` with `-opt <config.yml>`, but a config needs a discriminator without NVIDIA kernels, which does not exist yet (see `LICENSE_CLEANUP.md`).

Tests (`setup.cfg` sets `addopts=tests/`, so plain `pytest` runs everything):
```bash
pytest
pytest tests/test_ffhq_degradation_dataset.py::test_ffhq_degradation_dataset   # single test
```
Arch tests only run their bodies when CUDA is available. The dataset test generates a synthetic image, so it needs no data files.

Lint (matches CI in `.github/workflows/pylint.yml`; line length 120, single quotes enforced by pre-commit):
```bash
codespell
flake8 .
isort --check-only --diff gfpgan/ inference_gfpgan.py setup.py
yapf -r -d gfpgan/ inference_gfpgan.py setup.py
```

## Architecture

**Registry auto-discovery.** `gfpgan/archs/__init__.py`, `gfpgan/models/__init__.py` and `gfpgan/data/__init__.py` import every file ending in `_arch.py`, `_model.py` and `_dataset.py` respectively, so their classes register into BasicSR's `ARCH_REGISTRY` / `MODEL_REGISTRY` / `DATASET_REGISTRY`. New components must follow that filename suffix and use the `@..._REGISTRY.register()` decorator; YAML configs refer to them by class name (`model_type: GFPGANModel`, `type: FFHQDegradationDataset`, etc.).

**Architectures** (selected via `arch` in `GFPGANer`):
- `gfpganv1_clean_arch.py` + `stylegan2_clean_arch.py` (`clean`): U-Net encoder whose features modulate a pure-PyTorch StyleGAN2 decoder via channel-split SFT layers (`sft_half`); `forward(x, return_rgb, weight)` returns `(image, rgb_pyramid)`.
- `restoreformer_arch.py` (`RestoreFormer`): a separate transformer-based restorer plugged into the same pipeline.
- `arcface_arch.py`: identity network, used only for the identity loss during training.

**Inference pipeline** (`gfpgan/utils.py:GFPGANer`, used by `inference_gfpgan.py`): facexlib `FaceRestoreHelper` detects faces (RetinaFace), aligns and warps each one to 512x512, the network restores it in [-1, 1] normalized RGB, then the helper inverse-affine pastes faces onto the background with `use_parse=False` (no ParseNet mask) and no background upsampler. Checkpoints load `params_ema` if present, else `params`. facexlib's detection weights are downloaded into `gfpgan/weights` relative to the working directory, and those weights are a remaining license blocker.

**Training** (`gfpgan/models/gfpgan_model.py:GFPGANModel`, a BasicSR `BaseModel`): `FFHQDegradationDataset` synthesizes low-quality inputs on the fly (blur, downsample, noise, JPEG, color jitter, grayscale); facial component boxes were removed from the dataset. The model still supports pixel, multi-scale pyramid, perceptual, GAN, facial-component discriminator (Gram-matrix style loss, expects `loc_left_eye`/`loc_right_eye`/`loc_mouth` in the batch) and identity losses, each toggled by the corresponding entries in the YAML.

## Release

Bump `VERSION`, then push a git tag: `release.yml` (GitHub release) and `publish-pip.yml` (PyPI) both run only on tag pushes. Don't hand-edit `gfpgan/version.py`, since it is generated.
