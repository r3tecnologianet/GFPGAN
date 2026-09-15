# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

GFPGAN is a blind face restoration method (CVPR 2021) that uses a StyleGAN2-style generator as a prior. It is built on top of **BasicSR** (training framework, registries, losses, base model). Face detection, alignment and facial component landmarks use **MediaPipe**.

## License cleanup branch

Branch `license-cleanup` removes or replaces components with commercial-use restrictions. `LICENSE_CLEANUP.md` lists the status of each blocker, what was removed, added and kept with caveats. When changing code on this branch:
- Do not import `basicsr.ops.*` or `basicsr.archs.stylegan2_arch` (CUDA kernels under the Nvidia Source Code License-NC).
- Do not add code derived from DFDNet, ParseNet/PSFRGAN or other non-commercial sources.
- Do not add download URLs for the official GFPGAN, RestoreFormer, Real-ESRGAN or facexlib release weights, or face images/datasets without documented licenses.
- `tests/test_train_config.py` checks that `options/train_gfpgan_clean.yml` uses only clean components.

## Setup

```bash
pip install torch==2.1.2 torchvision==0.16.2 "numpy<2"
pip install --no-build-isolation basicsr==1.4.2
pip install "mediapipe==0.10.14" -r requirements.txt
python setup.py develop          # also generates gfpgan/version.py from VERSION + git sha
```

Version constraints: basicsr 1.4.2 imports `torchvision.transforms.functional_tensor` (removed in torchvision 0.17), so use torchvision ≤ 0.16 (Python ≤ 3.11) and numpy < 2. mediapipe 1.x needs numpy 2; mediapipe 0.10.x works but cannot run the BlazeFace full-range model, so the default detector is short range. MediaPipe models download into `gfpgan/weights/` (git-ignored) and run on CPU.

## Commands

Inference (no automatic weight download; pass weights you are licensed to use):
```bash
python inference_gfpgan.py -i <image_or_folder> -o results --model_path <weights.pth> --arch clean --channel_multiplier 2 -s 2
```
`--arch` accepts `clean | RestoreFormer`. `--aligned` skips detection for pre-cropped 512x512 faces. The input folder must contain only images.

Facial component boxes for a folder of aligned 512x512 training faces:
```bash
python scripts/generate_component_boxes.py -i <aligned_faces_dir> -o <component_boxes.pth>
```

Training (via BasicSR's `train_pipeline`; set the dataset paths in the config first):
```bash
python gfpgan/train.py -opt options/train_gfpgan_clean.yml
python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_clean.yml --launcher pytorch
```
A config `name` containing `debug` makes BasicSR log every iteration and validate/save every 8 iterations. Experiments are written to `experiments/<name>/` (git-ignored).

Tests (`setup.cfg` sets `addopts=tests/`, so plain `pytest` runs everything; tests use CUDA when available):
```bash
pytest
pytest tests/test_face_helper.py::test_face_helper_align_and_paste   # single test
```
The MediaPipe tests download their models on first run.

Lint (matches CI in `.github/workflows/pylint.yml`; line length 120, single quotes enforced by pre-commit):
```bash
codespell
flake8 .
isort --check-only --diff gfpgan/ scripts/ inference_gfpgan.py setup.py
yapf -r -d gfpgan/ scripts/ inference_gfpgan.py setup.py
```

## Architecture

**Registry auto-discovery.** `gfpgan/archs/__init__.py`, `gfpgan/models/__init__.py` and `gfpgan/data/__init__.py` import every file ending in `_arch.py`, `_model.py` and `_dataset.py` respectively, so their classes register into BasicSR's `ARCH_REGISTRY` / `MODEL_REGISTRY` / `DATASET_REGISTRY`. New components must follow that filename suffix and use the `@..._REGISTRY.register()` decorator; YAML configs refer to them by class name.

**Architectures:**
- `gfpganv1_clean_arch.py` + `stylegan2_clean_arch.py` (`clean`): U-Net encoder whose features modulate a pure-PyTorch StyleGAN2 decoder via channel-split SFT layers (`sft_half`); `forward(x, return_rgb, weight)` returns `(image, rgb_pyramid)`.
- `discriminator_arch.py`: `StyleGAN2DiscriminatorClean` (global; `forward(x, return_feats)` returns logits, or logits and block features) and `FacialComponentDiscriminatorClean` (always returns `(patch_logits, feats_or_None)`, two feature maps for the Gram style loss).
- `restoreformer_arch.py` (`RestoreFormer`): a separate transformer-based restorer plugged into the same inference pipeline.
- `arcface_arch.py`: identity network; not used by the clean config.

**Inference pipeline** (`gfpgan/utils.py:GFPGANer`, used by `inference_gfpgan.py`): `gfpgan/face_helper.py:FaceHelper` detects faces with MediaPipe BlazeFace on the full image plus overlapping windows (merged by NMS), aligns each face to 512x512 with a similarity transform from eyes, nose tip and mouth center, the network restores it in [-1, 1] normalized RGB, then faces are pasted back with a feathered square mask onto a Lanczos-upscaled background. Checkpoints load `params_ema` if present, else `params`.

**Training** (`gfpgan/models/gfpgan_model.py:GFPGANModel`, a BasicSR `BaseModel`): `FFHQDegradationDataset` synthesizes low-quality inputs on the fly (blur, downsample, noise, JPEG, color jitter, grayscale). With `crop_components`, it reads `{image name: {'left_eye', 'right_eye', 'mouth': [x1, y1, x2, y2]}}` (observer's view) from `component_path`, as built by `gfpgan/component_boxes.py`, and flips boxes with the image. The model combines pixel, multi-scale pyramid, discriminator feature matching (`feature_matching_weight`), GAN with lazy R1, facial component GAN and Gram style losses; `perceptual_opt` and `network_identity` are still accepted but not used by the clean config. The clean config uses learning rate 2e-4 because the from-scratch generator and discriminators diverged at 2e-3.

## Release

Bump `VERSION`, then push a git tag: `release.yml` (GitHub release) and `publish-pip.yml` (PyPI) both run only on tag pushes. Don't hand-edit `gfpgan/version.py`, since it is generated.
