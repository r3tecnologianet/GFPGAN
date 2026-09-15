# License cleanup

Branch `license-cleanup`, based on `7552a77` (TencentARC/GFPGAN). Goal: remove or replace components with commercial-use restrictions. Technical analysis, not legal advice. Evidence and sources: `gfpgan-spec/specs/license-inventory.md`.

## Blocker status

| Blocker | Status | Commit | Notes |
|---|---|---|---|
| Face detector (facexlib RetinaFace, WIDER FACE weights) | Resolved | `bee16bd` | MediaPipe BlazeFace (Apache 2.0 model card). Default is short range with sliding-window passes: the full-range model fails in mediapipe 0.10.x, and mediapipe 1.x needs numpy 2, which torch 2.1 / basicsr 1.4.2 do not support. MediaPipe runs on CPU |
| ParseNet paste-back mask | Resolved | `0d61da2`, `bee16bd` | Feathered square mask |
| Discriminators with NVIDIA kernels | Resolved | `66b7c3c` | `StyleGAN2DiscriminatorClean`, `FacialComponentDiscriminatorClean` (standard PyTorch ops) |
| Perceptual loss with VGG19/ImageNet | Resolved | `eb87dfa` | Discriminator feature matching (`feature_matching_weight`) |
| Facial component boxes derived from DFDNet / FFHQ landmarks | Resolved | `c68aed2` | Box rule over MediaPipe Face Mesh V2 contours; `scripts/generate_component_boxes.py` |
| Training configs | Resolved | `0c12b14` | `options/train_gfpgan_clean.yml`; smoke-tested only (see Verification) |
| Identity loss (ArcFace) | Open | — | Disabled; needs a face recognition model trained on licensed data |
| No pretrained generative prior | Open | — | The config trains GFPGANv1Clean from scratch (`decoder_load_path: ~`) at learning rate 2e-4; a StyleGAN2 prior trained on licensed data does not exist |
| Weights | Open | — | All release weights are trained on FFHQ. Retraining needs a licensed face dataset, which does not exist; not attempted |

## Removed

| Item | Reason |
|---|---|
| `gfpgan/archs/gfpganv1_arch.py` | Uses `basicsr.ops.fused_act` and `basicsr.archs.stylegan2_arch` (CUDA kernels under the Nvidia Source Code License-NC) |
| `gfpgan/archs/stylegan2_bilinear_arch.py`, `gfpgan/archs/gfpgan_bilinear_arch.py` | Use `basicsr.ops.fused_act` |
| `scripts/parse_landmark.py` | Facial component box computation derived from DFDNet (CC BY-NC-SA 4.0); reads FFHQ landmarks |
| `scripts/convert_gfpganv_to_clean.py` | Depends on `gfpganv1_arch` and converts restricted weights |
| `options/train_gfpgan_v1.yml`, `options/train_gfpgan_v1_simple.yml` | Use removed architectures, BasicSR `StyleGAN2Discriminator` (NVIDIA kernels), FFHQ, VGG19/ImageNet and ArcFace |
| `cog_predict.py`, `cog.yaml` | Download restricted weights and use Real-ESRGAN |
| `PaperModel.md` | Instructions for the model that needs NVIDIA kernels |
| `inputs/` | Photos of identifiable people; source and license not documented |
| `tests/data/gt/`, `tests/data/ffhq_gt.lmdb/`, `tests/data/test_eye_mouth_landmarks.pth` | Image and landmarks following FFHQ naming |
| `tests/test_utils.py`, old `tests/test_gfpgan_model.py`, `tests/data/test_gfpgan_model.yml` | Depend on removed architectures and restricted weights |
| `facexlib` dependency | Replaced by `mediapipe` |

## Added

| File | Content |
|---|---|
| `gfpgan/face_helper.py` | MediaPipe BlazeFace detection with sliding windows and NMS, similarity alignment from eyes, nose tip and mouth center, feathered paste-back |
| `gfpgan/archs/discriminator_arch.py` | Global and facial component discriminators in standard PyTorch |
| `gfpgan/component_boxes.py` | Component box rule, horizontal flip, MediaPipe Face Landmarker wrapper |
| `scripts/generate_component_boxes.py` | Builds the component boxes file for a folder of aligned faces |
| `options/train_gfpgan_clean.yml` | Training config with clean components only |
| `tests/test_face_helper.py`, `tests/test_discriminator_arch.py`, `tests/test_gfpgan_model.py`, `tests/test_component_boxes.py`, `tests/test_train_config.py` | Tests for the added components |

## Modified

| File | Change |
|---|---|
| `gfpgan/utils.py` | Uses `gfpgan.face_helper.FaceHelper`; only `clean` and `RestoreFormer` architectures; `det_model` option |
| `gfpgan/models/gfpgan_model.py` | Optional discriminator feature matching loss (`feature_matching_weight`) |
| `gfpgan/data/ffhq_degradation_dataset.py` | `crop_components` reads boxes from `component_path` and flips them with the image. JPEG quality is sampled as an integer: basicsr 1.4.2 `random_add_jpg_compression` passes a float that current OpenCV rejects (the upstream test at `7552a77` fails the same way) |
| `inference_gfpgan.py` | No automatic weight download; `--model_path` is required; no Real-ESRGAN |
| `requirements.txt`, `setup.cfg` | `facexlib` replaced by `mediapipe` |
| `tests/test_gfpgan_arch.py`, `tests/test_ffhq_degradation_dataset.py`, `tests/data/test_ffhq_degradation_dataset.yml` | Clean architecture only; synthetic images and component boxes |
| `MANIFEST.in`, `.github/workflows/pylint.yml`, `.gitignore` | Removed `inputs/`; downloaded MediaPipe models ignored |
| `README.md` | Notice at the top pointing to this file |

## Kept with caveats

| Item | Caveat |
|---|---|
| `gfpgan/archs/stylegan2_clean_arch.py`, `gfpganv1_clean_arch.py` | Pure PyTorch; `LICENSE` says StyleGAN2 code was modified from stylegan2-pytorch (MIT). Confirm there is no code from the official NVIDIA implementation |
| `gfpgan/archs/restoreformer_arch.py` | Modified from RestoreFormer (Apache 2.0). Release weights not verified |
| `gfpgan/archs/arcface_arch.py` | Source not stated; not used by the clean config |
| `gfpgan/models/gfpgan_model.py` | Still accepts `perceptual_opt` and `network_identity` from configs; the clean config uses neither |
| `basicsr` dependency | The package ships the NVIDIA kernels; no code on this branch imports them |
| `gfpgan/face_helper.py` alignment template | Derived from the facexlib 5-point template (MIT License), attributed in the file |
| `assets/gfpgan_logo.png`, `README*.md`, `Comparisons.md`, `FAQ.md` | Upstream branding and documentation, with links to restricted weights and facexlib |
| `LICENSE` | Unchanged |

## Verification

Environment: Python 3.11.16, torch 2.1.2+cu121, torchvision 0.16.2+cu121, numpy 1.26.4, opencv-python 4.11.0.86, basicsr 1.4.2, mediapipe 0.10.14, NVIDIA GeForce RTX 3060 Ti (8 GB).

| Check | Result |
|---|---|
| `pytest` (GPU) | 20 passed |
| `flake8 .`, `isort --check-only`, `yapf -r -d` | Pass |
| `codespell` | 1 finding: the isort `default_section` value in `setup.cfg` (unchanged since `7552a77`) |
| Inference, random-init `clean` weights, 4 public-domain NASA portraits (not committed) | Faces detected: 4/4 in a group portrait, 1/1 in three single portraits; aligned crops checked visually |
| Component boxes on the 7 aligned faces above | 7/7 generated; boxes checked visually |
| Global discriminator at 512x512, batch 1, forward + backward + R1 | 528 ms/iter, 1.3 GiB peak |
| Smoke training of `options/train_gfpgan_clean.yml` on 8 synthetic images (60 iterations, batch 2, R1 every 4) | No NaN/inf losses; all losses logged (pixel, pyramid, feature matching, GAN, component GAN, component style, R1); checkpoints and validation run; validation PSNR 7.25 → 8.85 dB; a loss spike around iteration 30 partially recovered. At learning rate 2e-3 the same run diverged to NaN within 4 iterations. A smoke run validates the pipeline, not convergence |
