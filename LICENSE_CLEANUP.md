# License cleanup

Branch `main`, based on `7552a77` (TencentARC/GFPGAN). Goal: remove or replace components with commercial-use restrictions. Technical analysis, not legal advice. Evidence and sources: `gfpgan-spec/specs/license-inventory.md`.

## Blocker status

| Blocker | Status | Commit | Notes |
|---|---|---|---|
| Face detector (facexlib RetinaFace, WIDER FACE weights) | Resolved | `bee16bd` | MediaPipe BlazeFace (Apache 2.0 model card). Default is short range with sliding-window passes: the full-range model fails in mediapipe 0.10.x, and mediapipe 1.x needs numpy 2, which torch 2.1 / basicsr 1.4.2 do not support. MediaPipe runs on CPU |
| ParseNet paste-back mask | Resolved | `0d61da2`, `bee16bd` | Feathered square mask |
| Discriminators with NVIDIA kernels | Resolved | `66b7c3c` | `StyleGAN2DiscriminatorClean`, `FacialComponentDiscriminatorClean` (standard PyTorch ops) |
| Perceptual loss with VGG19/ImageNet | Resolved | `eb87dfa` | Discriminator feature matching (`feature_matching_weight`) |
| Facial component boxes derived from DFDNet / FFHQ landmarks | Resolved | `c68aed2` | Box rule over MediaPipe Face Mesh V2 contours; `scripts/generate_component_boxes.py` |
| Training configs | Resolved | `0c12b14` | `options/train_gfpgan_clean.yml`; training settings shown stable over 10,000 iterations on 9927 aligned faces, with the facial component Gram style loss off (`docs/training_stability.md`) |
| Identity loss (ArcFace) | Open | — | Disabled; needs a face recognition model trained on licensed data |
| No pretrained generative prior | Open | — | The config trains GFPGANv1Clean from scratch (`decoder_load_path: ~`) at generator learning rate 2.5e-5; a StyleGAN2 prior trained on licensed data does not exist |
| Weights (face) | Open, and not solvable by downloading anything | — | All release weights are trained on FFHQ. Retraining needs a face corpus that is both licensed for commercial use and collected with the subjects' consent. A survey of the field found exactly one, and it is three orders of magnitude too small. See "Why there is no face corpus" below |
| Weights (background) | Resolved | — | `--bg_model` takes super-resolution weights trained here on CC0 photographs. Scenery and objects carry no biometric data and no likeness right, so unlike the face model this one is distributable. `docs/background_super_resolution.md` |

## Why there is no face corpus

The branch removed every restricted component and left one blocker: FFHQ is CC BY-NC-SA 4.0, so no weight trained
on it can be shipped. The obvious repair is to train on something else. Two surveys of the field, one of consented
and commercially licensed corpora and one of synthetic faces, found that nothing downloadable qualifies. Technical
analysis, not legal advice; what follows is what the sources say.

The largest pool that clears the copyright gate has since been collected to exhaustion rather than sampled, twice.
Every Wikimedia Commons file tagged as a photograph of a human, not imported from Flickr and at least 512 px gave 876
face crops under a CC0, public domain and CC BY filter. The project owner then admitted CC BY-SA on 2026-09-21,
accepting an unsettled share-alike question in exchange for breadth, which added 2,662 crops. Hand review kept 1,524
of the 3,538, and the aligned corpus is 1,478 faces, 1,414 of them training.

That is 2% of FFHQ's 70,000, and it is the ceiling of this source rather than a sample of it. The measurement, the
sources ruled out on the way -- PD12M on quality, every Flickr pool because the list that would remove FFHQ's own
photographs is itself CC BY-NC-SA -- and what it leaves open are in
[docs/commons_face_corpus.md](docs/commons_face_corpus.md).

**Three independent gates, and a copyright licence only opens the first.** Creative Commons says so in its own
text: CC BY 4.0 section 2(b) does not license "publicity, privacy, and/or other similar personality rights", and
CC0 section 4(c) has the affirmer disclaim "responsibility for clearing rights of other persons". A CC0 photograph
is a settled question about the photographer and an open one about everyone visible in it. Biometric regimes
attach independently of who owns the image: GDPR Article 4(14) names facial images and Article 9 requires explicit
consent for processing that uniquely identifies, and EU AI Act Article 5(1)(e), applicable since 2 February 2025,
prohibits creating or expanding facial recognition databases by untargeted scraping of facial images.

**Exactly one corpus passes both gates.** The Face Research Lab London Set is CC BY 4.0 with signed consent
covering altered forms -- and holds about 1,000 images of 102 adults, roughly 0.2 per cent of FFHQ. It is usable
for evaluation, not for training a prior. Everything else fails one gate or the other, and the two are
anti-correlated: the properly consented sets (Casual Conversations, Chicago Face Database, FERET) are contractually
research-only, and the commercially licensed image pools (Unsplash Lite, Open Images, CC BY Flickr) carry no
subject consent at all.

**Synthetic faces move the problem rather than solve it.** Every GAN or diffusion face corpus traces back to FFHQ
or CASIA-WebFace through its generator. The rendered corpora have the best provenance in the field -- Microsoft's
DigiFace-1M and Face Synthetics derive from 511 consented 3D scans plus artist-authored assets -- and are closed by
the most explicit clause found anywhere: their Research Use of Data Agreement makes "artificial intelligence models
trained on Data" into "Results" and bars using the Data or any Results in a commercial offering. It is the one
licence that names the trained weights, so there is nothing to interpret.

**The enforcement reaches the model.** In the FTC's Everalbum/Paravision order the remedy was to delete the models
developed from biometric data collected without consent, not to pay a fine. EDPB Opinion 28/2024 contemplates
erasure of the model itself and puts a duty on the deployer to verify the model was not developed unlawfully. A
consent defect in a corpus therefore propagates into the checkpoint and into whoever ships it, which is why this
cannot be treated as a licensing detail to be resolved later.

**Buying it is not a shortcut either.** No face-data vendor surveyed publishes a participant release form. The two
that publish licence text at all disclaim every warranty -- Defined.ai's reads "THE DATA IS PROVIDED 'AS IS' AND
LICENSOR HEREBY DISCLAIMS ALL WARRANTIES" -- and their consent language sits in supplier codes of conduct, which
bind the vendor's suppliers rather than promising anything to the buyer. Every "fully consented" claim found was
product-page marketing rather than a term of sale, and none publishes a price. The protection actually needed, a
representation that each subject signed a release covering machine learning training and sublicensable commercial
use, with indemnity and a copy of the form, appears in none of the published terms and would have to be negotiated
from nothing. Part of the reason is structural: BIPA section 15(c) bars a private entity from selling, leasing,
trading or otherwise profiting from biometric information, which is a problem for the vendor's business model and
not only for the buyer's use.

**What this leaves.** Three routes, none of them a download and none of them quick: commission a collection with
documented consent, negotiate a vendor contract that carries the consent representations no published terms
currently offer, or ship the code and the training pipeline and let users bring weights they are entitled to. `inference_gfpgan.py` already implements the third -- it requires
`--model_path` and downloads nothing -- so the branch is closer to a defensible position than the open blocker
suggests, provided "commercially usable" is understood to mean the code and the pipeline rather than the face
weights. Every public face restorer shares this failure: CodeFormer is non-commercial, GPEN ships no licence at
all, and GFPGAN's own release weights state no terms, all of them resting on FFHQ.

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
| `gfpgan/models/gfpgan_model.py` | Optional discriminator feature matching loss (`feature_matching_weight`); optional generator gradient norm clipping (`generator_grad_clip`, logs `g_grad_norm`) |
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
