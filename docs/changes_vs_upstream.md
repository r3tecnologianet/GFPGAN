# Changes against upstream GFPGAN

Executive summary of the `license-cleanup` branch against its base, commit `7552a77` of TencentARC/GFPGAN.
Per-component status and evidence are in `LICENSE_CLEANUP.md`; the training work is in `docs/training_stability.md`.

**Goal:** remove every component that carries commercial-use restrictions, and keep the method working.

**Size of the change:** 48 files changed, 1660 insertions and 2752 deletions — 23 files removed, 11 added,
13 modified and 1 renamed. The project is smaller than the original.

## 1. Face detection and alignment — replaced

- Removed the `facexlib` dependency and its RetinaFace detector, whose weights are trained on WIDER FACE.
- Added `gfpgan/face_helper.py`: MediaPipe BlazeFace (Apache 2.0 model card) over the full image plus overlapping
  windows merged by NMS, similarity alignment from eyes, nose tip and mouth centre, and feathered paste-back.
- Removed the ParseNet paste-back mask, replaced by a feathered square mask.

## 2. Architectures — the ones needing NVIDIA CUDA kernels are gone

- Removed `gfpganv1_arch.py`, `stylegan2_bilinear_arch.py`, `gfpgan_bilinear_arch.py` and
  `scripts/convert_gfpganv_to_clean.py`, all of which depend on `basicsr.ops` (Nvidia Source Code License-NC).
- Kept only the `clean` variant, which is pure PyTorch.
- Added `gfpgan/archs/discriminator_arch.py`: the global StyleGAN2 discriminator and the facial component
  discriminators reimplemented without those kernels.

## 3. Perceptual loss — replaced

The VGG19/ImageNet perceptual loss is gone. Discriminator feature matching (`feature_matching_weight`) takes its
place and needs no external weights.

## 4. Facial component boxes — recomputed

- Removed `scripts/parse_landmark.py`, derived from DFDNet (CC BY-NC-SA 4.0) and reading FFHQ landmarks.
- Added `gfpgan/component_boxes.py` and `scripts/generate_component_boxes.py`, which derive eye and mouth boxes
  from MediaPipe Face Mesh contours.

## 5. Identity loss — disabled

ArcFace is out of the training path. `GFPGANModel` still accepts the options, and the clean config uses neither.
Identity preservation is therefore not measured by anything in this repository.

## 6. Training configuration — rewritten and validated

`options/train_gfpgan_v1.yml` became `options/train_gfpgan_clean.yml`, carrying the settings that survived twelve
screening rounds and ten runs (`docs/training_stability.md`): generator lr 2.5e-5, global discriminator lr 2e-5,
component discriminators lr 2.5e-5, GAN weight 0.05, R1 every 4 iterations, generator gradient clipping 10, and
the component Gram style loss off. Upstream used lr 2e-3, which diverges to NaN within 4 iterations when there is
no pretrained prior to start from.

## 7. Inference — no automatic downloads

`inference_gfpgan.py` no longer downloads weights: `--model_path` is required. The Real-ESRGAN background
upsampler is gone; the background is upscaled with Lanczos instead.

## 8. Example data and tests

- Removed `inputs/` (photographs of identifiable people with no documented licence) and the FFHQ-named test data.
- Added `tests/test_face_helper.py`, `tests/test_component_boxes.py`, `tests/test_discriminator_arch.py` and
  `tests/test_train_config.py`, all on synthetic images. The suite went from 2 to 20 tests.

## 9. New documentation

`LICENSE_CLEANUP.md` (status of each blocker), `docs/training_stability.md` (screening rounds, runs and metrics,
including the comparison against the official GFPGAN v1.4) and `CLAUDE.md`.

## Still open

- **No pretrained generative prior.** Upstream starts from a StyleGAN2 trained on FFHQ; here the decoder trains
  from scratch, which is the main reason its restorations are weaker on real photographs.
- **Licensed corpus.** Training currently uses FFHQ as a stand-in, so the weights are for evaluation only.
- **Release weights.** The official checkpoints stay out of this repository, and the build produced here is not a
  release candidate.
