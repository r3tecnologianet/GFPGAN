# Training stability of `options/train_gfpgan_clean.yml`

Goal: find settings under which the clean training config trains stably on real, licensed faces. Each run changes one factor from the previous stable-or-best run.

## Stability criterion

A run is stable when, over 5000 consecutive iterations:
1. no logged loss is NaN or inf;
2. the generator loss total (pixel, feature matching, GAN, component GAN, component style) never exceeds 10x its rolling median over the previous 200 iterations;
3. `real_score` and `fake_score` of the global discriminator stay within |value| < 100;
4. validation PSNR (`net_g_ema`) never drops by more than 1 dB between two validations.

The check is applied to the training log by an analysis script (kept outside the repository).

## Data

| Item | Value |
|---|---|
| Source | 716 hand-reviewed face crops from PD12M, collected by the OpenGFPGAN project (`datatest/faces-716-crops`) |
| Licences (per-image records) | 351 CC0 1.0, 298 Public Domain Mark 1.0 (of the 649 images used) |
| Sources | Wikimedia Commons 594, museums and archives 55 |
| Crop resolution | shorter side 600–5394 px, median 1103 px |
| Alignment | `gfpgan/face_helper.py` (BlazeFace short range, largest face, eye distance ≥ 20 px) to 512x512: 693 of 716 aligned |
| Component boxes | `scripts/generate_component_boxes.py`: 649 of 693 |
| Split | 617 training, 32 validation (random, seed 0); validation inputs degraded once with the training degradation ranges (seed 0) |
| Location | `/mnt/dados/gfpgan-clean/faces-716` (not in the repository) |

617 faces are far too few to train a good restoration model. The runs test training stability, not output quality.

**Dataset change after run 07.** The 617 faces come from PD12M, whose detected faces are mostly not usable as clean ground truth: engravings, half-tone prints, damaged scans and blurred images. Hand labels made by the OpenGFPGAN project rate 104 of 395 crops as usable (26%). Automatic quality signals do not find them: measured against those labels, NIQE has AUC 0.535 and Laplacian sharpness 0.459, both at chance, matching that project's own measurement of sharpness (0.446) and halftone (0.532). With no working automatic filter, further stability work uses an already curated aligned face dataset (FFHQ at 512x512), and the licensed corpus is revisited afterwards.

| Item | Value | Full set |
|---|---|---|
| Source | `Ryan-sjtu/ffhq512-caption` on Hugging Face, 8 of 54 parquet shards (3.8 GiB) | all 54 shards (26 GiB) |
| Faces | 10,000 aligned 512x512 images, none discarded | 70,000, none discarded |
| Component boxes | 9991 of 10,000 | 69,930 of 70,000 |
| Split | 9927 training, 64 validation | 69,674 training, 256 validation |
| Location | `/mnt/dados/gfpgan-clean/ffhq512` | `/mnt/dados/gfpgan-clean/ffhq512full` |

Both splits are random with seed 0, images without a detected face are dropped, and the validation inputs are degraded once with the training degradation ranges (seed 0). Neither is in the repository.

FFHQ is used here because it is aligned and curated, which is what the stability runs need. It carries its own licence terms and is not a licensed corpus for release; the clean-room data question is separate and stays open.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 08 | base 11 on the FFHQ dataset | base 11 (seed 0), 9927 training faces | **Stable.** 10,000 iterations: no NaN, one generator spike at 1882 (G total 1.93 against a median of 0.10), no \|score\| ≥ 100, no PSNR drop > 1 dB; clean streak 8118 (1883–10,000). Validation PSNR 10.91 → 21.50 dB at 3500, then 20.97–21.18 to the end |

With 9927 training faces an epoch is 4963 iterations instead of 308, so each sample is seen 16 times less often. The episodes that recurred at fixed iterations with the 617-face set are gone: a single small spike in 10,000 iterations. Validation PSNR is 3.5 dB below the 617-face runs and flattens instead of falling, which is what a harder, more varied validation set gives.

Run 08 evaluated on the 64 FFHQ validation pairs (EMA weights):

| Iteration | PSNR (dB) | Component PSNR (dB) | LMD (px) | LMD failures | NIQE |
|---|---|---|---|---|---|
| degraded input | 20.64 | 20.10 | 5.65 | 7 | 12.86 |
| 1000 | 18.40 | 18.89 | 8.87 | 3 | 13.11 |
| 2000 | 20.88 | 20.78 | 6.73 | 3 | 11.60 |
| 4000 | 21.47 | 20.94 | 5.96 | 3 | 10.43 |
| 6000 | 21.04 | 20.36 | 5.74 | 2 | 8.72 |
| 8000 | 21.05 | 20.29 | 6.02 | 2 | 7.47 |
| 10,000 | 21.15 | 20.40 | 5.38 | 2 | **7.31** |
| ground truth | — | — | 0 | 0 | 3.75 |

On this dataset the generator beats its input on every metric from iteration 2000 on, and keeps improving NIQE (12.86 → 7.31) while PSNR stays near 21 dB. LMD ends below the degraded input (5.38 against 5.65) and MediaPipe fails to find a face in 2 of 64 outputs against 7 of 64 inputs. The PSNR peak at 4000 followed by a plateau is the same blur-for-texture trade as in the 617-face runs, without the later decline.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 09 | base 12 on the FFHQ dataset | base 11 plus the three facial component discriminators at component lr 2.5e-5, 9927 training faces | **Stable.** 10,000 iterations with no violation at all; clean streak 10,000, from the first iteration. Validation PSNR 10.81 → 21.35 dB at 4750, then 21.18 at 10,000 |

The facial component discriminators cost nothing on this dataset: run 09 is clean from the first iteration (run 08 had one spike) and ends at the same PSNR (21.18 against 21.15). The verdict against them in round 12 came from the 617-face set, where they lowered every quality metric; it does not hold here.

Runs 08 and 09 on the same 64 validation pairs (EMA weights):

| Iteration | PSNR 08 / 09 (dB) | Component PSNR 08 / 09 (dB) | LMD 08 / 09 (px) | NIQE 08 / 09 |
|---|---|---|---|---|
| degraded input | 20.64 | 20.10 | 5.65 | 12.86 |
| 2000 | 20.88 / 19.99 | 20.78 / 20.16 | 6.73 / 7.47 | 11.60 / 12.30 |
| 4000 | 21.47 / 21.29 | 20.94 / 20.19 | 5.96 / 6.73 | 10.43 / 10.45 |
| 6000 | 21.04 / 21.03 | 20.36 / 20.15 | 5.74 / 5.61 | 8.72 / 7.96 |
| 8000 | 21.05 / 21.05 | 20.29 / 20.26 | 6.02 / 5.16 | 7.47 / 5.44 |
| 10,000 | 21.15 / 21.18 | 20.40 / 20.39 | 5.38 / **5.05** | 7.31 / **5.29** |
| ground truth | — | — | 0 | 3.75 |

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 10 | base 12 on the full FFHQ dataset, 100,000 iterations | base 12 (seed 0), 69,674 training faces, 256 validation pairs, learning rate halved at 75,000 and 90,000 | **Stable.** 100,000 iterations with no violation at all: no NaN, no generator spike, no \|score\| ≥ 100, no PSNR drop > 1 dB; clean streak 100,000. Validation PSNR 21.48 → 23.14 dB at 87,500, 23.05 at the end; about 16 hours on one RTX 3060 Ti at 0.60 s/iteration |

Seven times more data removes the last instability and raises quality: run 10 never spikes in 100,000 iterations, and its validation PSNR (23.05 dB on 256 pairs) is well above run 09's 21.18 dB on 64 pairs. The PSNR rises steadily to iteration 87,500 instead of peaking early and falling, so the overfitting seen with 617 and 9927 faces does not appear here.

Run 10 evaluated on the 256 FFHQ validation pairs (EMA weights):

| Iteration | PSNR (dB) | Component PSNR (dB) | LMD (px) | LMD failures | NIQE |
|---|---|---|---|---|---|
| degraded input | 21.86 | 21.31 | 6.23 | 12 | 12.74 |
| 10,000 | 21.82 | 21.17 | 5.01 | 2 | 6.00 |
| 25,000 | 22.15 | 21.72 | 4.07 | 0 | 4.36 |
| 50,000 | 22.64 | 22.33 | 3.71 | 0 | 4.39 |
| 75,000 | 22.99 | 22.69 | 3.37 | 0 | 4.48 |
| 100,000 | **23.05** | **22.60** | **3.21** | 1 | **3.75** |
| ground truth | — | — | 0 | 0 | 3.77 |

The generator beats its input on every metric: +1.19 dB PSNR, +1.29 dB component PSNR, LMD 3.21 px against 6.23, and NIQE 3.75, equal to the ground truth's 3.77 (the 617-face runs reached 6.38 against a ground truth of 5.68). Realism saturates around iteration 25,000 (NIQE 4.36) and the rest of the run buys fidelity: PSNR +0.90 dB and LMD −0.86 px from 25,000 to 100,000. The learning rate halving at 75,000 and 90,000 coincides with the last NIQE gain, from 4.48 to 3.75.

This closes the stability work: `options/train_gfpgan_clean.yml` trains without a single violation for 100,000 iterations and produces restorations that a no-reference quality metric cannot separate from real faces. What remains open is the licensed corpus (FFHQ is a stand-in), and identity preservation, which no metric here measures.

## Evaluation build

The run 10 generator was exported for an independent team to judge output quality: the EMA weights at iteration
100,000 alone, as `params_ema`, 332.5 MiB, sha256
`4d29d02293fb4f34d3f29d9d030ba07dce41f0269227ea809d227cecceb54f69`. It is packaged outside the repository, at
`/mnt/dados/gfpgan-clean/export/`, with the training config, a model card and checksums, and both inference paths
(`--aligned` and full detection) were run against the exported file.

These weights are trained on FFHQ and are for evaluation only, not for release: the licensed corpus is still the
open task, so the export is deliberately not committed here.

### Comparison with the official GFPGAN v1.4

The official release weights use the same `GFPGANv1Clean` architecture, so they load into this code with
`strict=True` (285 tensors, no missing or extra key). They were downloaded as a local reference only: they are
trained on FFHQ with non-commercial terms, are not committed, redistributed, or used for anything but this
comparison.

Same 256 FFHQ validation pairs, same evaluation code:

| | Degraded input | Official v1.4 | This build | Ground truth |
|---|---|---|---|---|
| PSNR | 21.86 dB | 21.69 dB | **23.05 dB** | — |
| PSNR over eye and mouth boxes | 21.31 dB | 21.42 dB | **22.60 dB** | — |
| Landmark distance | 6.23 px | **2.47 px** | 3.21 px | 0 |
| NIQE | 12.74 | 4.36 | **3.77** | 3.77 |

**These numbers flatter this build and should not be read as beating the official model.** The validation set is
FFHQ degraded by the same synthetic pipeline this build trained on, so it is in-domain here and out-of-domain for
v1.4. Landmark distance, the metric least sensitive to that bias, favours v1.4 by a wide margin (2.47 against
3.21 px).

Visual inspection (grids in `/mnt/dados/gfpgan-clean/compare/`) agrees with the landmark result, not with PSNR:

- On the synthetic validation faces both models produce plausible restorations. This build keeps more skin and
  fabric texture; v1.4 is smoother and cleaner.
- On one validation face with dark skin this build fails badly, with heavy artefacts across the face, while v1.4
  restores it cleanly but lightens the skin tone noticeably — two different failures, both worth recording.
- On real unaligned photographs, which neither model saw in this form, **v1.4 is clearly better**: this build
  produces melted mouth and beard detail and a deformation around the nose, where v1.4 stays clean and faithful.
- In one photo the detector reported a face in a neon sign, and both models "restored" it: a detector false
  positive, not a generator problem.

Conclusion: the in-domain metrics overstate this build. On real photographs it is still behind the official model,
which is expected from a decoder trained from scratch for 2.9 epochs without a generative prior. The gap is a
data and schedule problem, not a stability problem.

Run 09 starts slower, matches run 08 by iteration 4000 and then pulls ahead: at 10,000 its NIQE is 5.29 against 7.31 (ground truth 3.75) and its LMD 5.05 against 5.38, with the same PSNR and component PSNR. The facial component discriminators earn their place once the data is curated.

Conclusion: `options/train_gfpgan_clean.yml` takes base 12 — base 11 plus the three facial component discriminators at component discriminator lr 2.5e-5, with the component Gram style loss still off.

## Environment

Python 3.11.16, torch 2.1.2+cu121, basicsr 1.4.2, mediapipe 0.10.14, NVIDIA GeForce RTX 3060 Ti (8 GB).

## Runs

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 01 | baseline | `options/train_gfpgan_clean.yml` as committed: lr 2e-4 (G, D, components), batch 2, R1 weight 10 every 16 iterations, feature matching 1.0, component style 200, pyramid removed at 50,000 | Stopped at 500 iterations (0.64 s/iter, 7.8 GiB). No NaN, no generator spike; 12 iterations with \|score\| ≥ 100, first at 23 (fake_score −283); max fake_score 3.8e3, max component style loss 5.7e3; PSNR 10.30 dB at 250 |

### Screening at 500 iterations, one factor changed from run 01

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Verdict |
|---|---|---|---|---|---|
| R1 every 4 iterations | 145, from 346 | 18, from 339 | 29, from 19 | 9.11 → 5.93 | Worse |
| Discriminator lr 2e-5 | 366, from 135 | 0 | 14, from 23 | 5.93 constant | Worse |
| Equalized learning rate in the global discriminator | 448, from 40 | 0 | 25, from 31 | 5.93 constant | Worse |

Diagnosis from the iterations before the first non-finite value, identical in the three variants: the global discriminator becomes confident (fake_score −1e3 to −1e17), the generator adversarial loss grows, and the final generator output explodes (pixel loss up to 1e15) while the U-Net intermediate outputs stay bounded (pyramid losses 0.2–90). The component Gram style loss, which grows at least quadratically with feature magnitude, reaches inf first. Changes on the discriminator side do not help; the generator, trained from scratch without a prior, is pushed by adversarial terms from the first iteration.

Next factors, on the generator side: gradient norm clipping (`generator_grad_clip`) and a linear warm-up of the adversarial terms (`adversarial_warmup_iters`). The criterion is evaluated as the longest run of 5000 consecutive iterations without a violation.

### Screening round 2 at 500 iterations, one factor changed from run 01

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|
| Run 01 baseline (first 500) | 0 | 0 | 12, iterations 23–71 | 10.30 | 429 (72–500) | Reference |
| `generator_grad_clip: 10` | 0 | 7, 277–298 | 9, 105–287 | 11.89 → 13.30 | 202 (299–500) | Best; generator gradient norm median 11.9, max 7.9e8 before clipping |
| `generator_grad_clip: 1` | 0 | 10, 249–373 | 11, 9–359 | 11.92 → 13.36 | 127 (374–500) | Same PSNR as 10, more violations |
| `adversarial_warmup_iters: 250` | 0 | 0 | 458 | 8.48 → 6.19 | 14 | Worse: with less adversarial pressure on the generator, the discriminator keeps large logits |

Decision: run 02 is run 01 with `generator_grad_clip: 10` for 5000 iterations. Only `generator_grad_clip` is kept in the code; the equalized learning rate option and the adversarial warm-up option were removed after their screening results.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 02 | `generator_grad_clip: 10` | run 01 settings plus generator gradient norm clipping at 10 | Exploded at iteration 444–456 and stopped: generator gradient norm inf, pixel loss 1.4e8, component style loss 1.7e20. PSNR 11.76 dB at 250 |

Run 02 diagnosis, from the per-iteration losses: iterations 380–439 are calm (generator gradient norm 1–6). At 440–443 the eye discriminators start to win (right-eye D loss 1.39 → 0.98), the eye GAN loss and the component Gram style loss rise (style 0.003 → 0.078) and the gradient norm reaches 51. At 444 the component path explodes first (left-eye GAN 15.2, style 20.7, gradient norm 1560) while the global fake_score is still −0.83; the global discriminator follows from 446 (fake_score −318, then −3.8e3). The trigger is the facial component path: the local discriminators have no R1, and once they separate, the style loss (weight 200) sends very large gradients to the generator. Gradient clipping does not contain it, because Adam normalizes the gradient scale and a non-finite norm makes clipping produce NaN.

### Screening round 3 at 1500 iterations, one factor changed from run 02

Longer screens, because run 02 exploded after the 500 iterations used before.

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|
| `comp_style_weight: 20` | 6, 484–503, recovered | 88, 237–1422 | 71, 98–1419 (min fake_score −3.4e6) | 12.13 → 19.94, rising at every validation | 297 (796–1092) | Best: survives the explosions and keeps learning; still unstable |
| No facial component discriminators | 0 | 145, from 302 | 791, from 303 | 13.06 → 16.28 | 301 (1–301) | The global path alone destabilizes after 300 iterations; generator gradient norm median 99.6 |
| Component discriminator lr 2e-5 | 0 (stopped at 268) | 68, from 201 (up to 5.7e10) | 140, from 194 | 12.48 | 193 (1–193) | Worst; stopped early |

Reading: there are two sources of instability. The facial component path (local discriminators without R1, Gram style loss) produces the non-finite values; lowering the style weight from 200 to 20 removes most of them. The global discriminator path produces large scores and generator spikes on its own. The discriminator-side screens of rounds 1 and 2 ran with style weight 200, so their results were confounded by the component explosion and are repeated on the new base.

New base (base 03): run 02 with `comp_style_weight: 20`.

### Screening round 4 at 1500 iterations, one factor changed from base 03

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|
| Base 03 (`comp_style_weight: 20`, from round 3) | 6 | 88 | 71 | 12.13 → 19.94, rising | 297 | Reference |
| `comp_style_weight: 2` | 0 | 102, 448–1359 | 222, 31–1463 | 11.80 → 19.14, then 16.91 (drop of 2.2 dB) | 344 (500–843) | Worse: no NaN, but more score violations and a PSNR drop |
| Global discriminator lr 2e-5 | 6, 365–840 | 81, 263–1348 | 32, 104–853 (min fake_score −1.1e5) | 10.98 → 19.52, rising | 430 (391–820) | Best so far: half the score violations and the longest clean streak |
| `r1_reg_weight: 50` | 980, from 520 | 280, from 205 | 1254, from 16 | 12.02 → 14.26, then 5.93 | 302 | Worst: collapses after 500 iterations |

New base (base 04): base 03 with global discriminator lr 2e-5. Remaining issues: 6 sporadic non-finite values from the component path, 81 generator spikes, 32 score violations.

### Screening round 5 at 1500 iterations, one factor changed from base 04

Factors: generator lr 1e-4; facial component GAN loss weight 0.1; R1 on the facial component discriminators (`comp_r1_reg_weight`, new option, same interval as the global R1).

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|
| Base 04 (global discriminator lr 2e-5, from round 4) | 6 | 81 | 32 | 10.98 → 19.52, rising | 430 | Reference |
| Generator lr 1e-4 | 0 | 30, 787–1372 | 4, 789–1356 | 11.63 → 21.66, rising | 786 (1–786) | Best so far on every measure; generator gradient norm median 4.5, max 3.9e6 (was ~1e17) |
| Component GAN loss weight 0.1 | 0 | 103, 499–1327 | 57, 34–1318 | 12.30 → 15.75 → 14.17 (drop of 1.58 dB) → 21.59 | 447 (52–498) | Worse than generator lr 1e-4 |
| `comp_r1_reg_weight: 10` | 0 | 45, 203–1478 | 57, 191–1478 (min fake_score −1.5e5) | 11.21 → 21.43, rising | 471 (994–1464) | Mixed against base 04 (no NaN, fewer spikes, more score violations); worse than generator lr 1e-4 on every measure. Option not kept |

New base (base 05): base 04 with generator lr 1e-4 (generator 1e-4, global discriminator 2e-5, component discriminators 2e-4, gradient clipping 10, component style weight 20, R1 weight 10 every 16 iterations, batch 2).

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 03 | base 05 for 5000 iterations | base 05 | Stopped at 1091 iterations: no NaN, 33 generator spikes in two episodes (394–413, 922–1038), 4 score violations, PSNR 11.70 → 18.58 dB rising, longest clean streak 456 (469–924). Episodes recur about every 500 iterations, so 5000 consecutive clean iterations were not reachable with this base |

Run 03 diagnosis: both episodes start in the facial component path. At 390–395 the right-eye discriminator starts to win (D loss 1.39 → 1.01), the generator right-eye GAN loss jumps 0.87 → 16.1 and the gradient norm reaches 721; the global fake_score follows one iteration later (−1.0e4). At 922–929 the same happens through the left eye (D loss 0.84, generator left-eye GAN loss 294).

The component discriminator update, inherited from upstream GFPGAN, adds the real term with the component GAN loss (vanilla, weight 1) and the fake term with the global GAN loss (`cri_gan`: wgan_softplus, weight 0.1). The local discriminators therefore receive a 10x weaker penalty of a different kind on restored patches than on real patches, which is consistent with them suddenly winning.

### Screening round 6 at 1500 iterations, one factor changed from base 05

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 05 (generator lr 1e-4 row of round 5) | 0 | 30 | 4 | 21.66 | 3.9e6 | 786 | Reference |
| `component_d_fake_loss: component` (fake term with the component GAN loss) | 0 | 37, 585–1349 | 7, 63–1343 | 11.69 → 21.60, rising | 1.4e10 | 712 (613–1324) | No improvement; option not kept |
| `comp_style_weight: 0` | 0 | 17, 206–1405, magnitude 25–129 | 3, all at 46–70 | 11.68 → 22.04, rising | 5.0e5 | 614 (740–1353) | Better: half the spikes and much smaller, no score violation after iteration 70, highest PSNR. The component Gram style loss is the main source of the exploding gradients |
| Component GAN loss weight 0.1 | 0 | 46, 770–1441 (one of 5.1e6 at 770) | 10, 770–1420 (min fake_score −6.1e4) | 12.45 → 23.19, rising | 1.3e8 | 769 (1–769) | Highest PSNR, but one explosion at 770; the lower generator loss median (0.33) also makes the 10x spike threshold stricter |

New base (base 06): base 05 with `comp_style_weight: 0` (no component Gram style loss); it had the smallest and fewest spikes and no score violation after iteration 70.

Runs with the same settings vary: run 03 (base 05) already had 33 spikes at iteration 1091, against 30 in 1500 iterations for the base 05 screen. Single 1500-iteration screens cannot separate small differences.

Spike composition in the `comp_style_weight: 0` screen: the generator loss total over the run has medians pixel 0.019, feature matching 0.060, global GAN 0.071 and component GAN about 0.69 per component. In 16 of the 17 spikes the component GAN terms dominate (10 to 237 per component, generator gradient norm 380 to 8000), while pixel, feature matching and global GAN stay small; only the 1354–1368 episode also moves the global fake_score (−15 to −39). The remaining instability comes from the facial component discriminators.

### Screening round 7 at 1500 iterations, one factor changed from base 06

Factors: component GAN loss weight 0.1; component discriminator lr 5e-5; no facial component discriminators (tests whether the global path alone is stable with the current learning rates).

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 06 (`comp_style_weight: 0` row of round 6) | 0 | 17, magnitude 25–129 | 3, all at 46–70 | 22.04 | 5.0e5 | 614 | Reference |
| Component GAN loss weight 0.1 | 0 | 46, 454–1026, magnitude 3.8–17 | 9, 457–1001 (min fake_score −3.5e3) | 12.25 → 23.39, rising | 3.4e4 | 501 (496–996); clean from 1027 to the end (474) | Mixed: highest PSNR and smallest gradients, but more violations. The lower generator loss median (0.33 against 2.2) makes the 10x spike threshold much stricter |
| Component discriminator lr 5e-5 | 0 | 25, 712–1497, magnitude 81–161 | 14, 155–1496 | 12.24 → 22.38, rising | 1.3e6 | 556 (156–711) | Worse: more score violations until the end |
| No facial component discriminators | 0 | 22, 708–1500 | 4, all at 710–720 (min fake_score −527) | 12.37 → 24.02, rising | 1.8e3 | 707 (1–707) | Best: one episode at 708–720, highest PSNR, smallest gradients. Without component losses the generator loss median is 0.11–0.12, so the 10x spike threshold is very strict |

New base (base 07): base 06 without the facial component discriminators. This removes GFP-GAN's facial component losses from training; they are to be revisited once the global path is stable.

Spike composition on base 07 (no component discriminators): medians pixel 0.014, feature matching 0.039, global GAN 0.070. Every spike is a global-path event: the global discriminator suddenly becomes confident on restored images (fake_score −527 at 710, −398 at 719) while real_score drifts up (to 6.5), and the generator feature matching (up to 63), global GAN (up to 53) and pixel (up to 12) losses jump together, with gradient norms up to 1.8e3. After 720 only small spikes remain (fake_score −5 to −34).

### Screening round 8 at 1500 iterations, one factor changed from base 07

Factors on the global path: feature matching weight 0.1; R1 every 4 iterations (repeated without the component explosion that confounded round 1); global GAN loss weight 0.05.

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 07 (no component discriminators, from round 7) | 0 | 22 | 4, 710–720 | 24.02 | 1.8e3 | 707 | Reference |
| `feature_matching_weight: 0.1` | 0 | 15, 680–1499 | 5, 682–687 (min fake_score −910) | 12.25 → 23.88, rising | 1.9e3 | 799 (700–1498) | Slightly better; the generator loss median drops to 0.085, so the 10x spike threshold is 0.85 |
| R1 every 4 iterations | 0 | 12, 595–1429 | 5, 597–604 (min fake_score −225) | 12.52 → 24.37, rising | 663 | 823 (606–1428) | Best so far: fewest spikes, shortest episode, smallest gradients, highest PSNR. Stronger R1 on the global discriminator helps once the component explosion is gone (round 1 was confounded) |
| Global GAN loss weight 0.05 | 0 | 22, all at 586–621 | 8, 586–610 (min fake_score −756) | 12.51 → 24.49, rising | 8.3e4 | 879 (622–1500, to the end) | Also better: one sharper episode, then clean to the end |

New base (base 08): base 07 with R1 every 4 iterations (fewest total violations, 17 against 30, and 100x smaller gradients than global GAN weight 0.05, which is tested on top of it next).

Every variant of rounds 7 and 8 has its episode in the same range, iterations 580–720. With 617 training faces and batch 2, one epoch is about 308 iterations, and all runs use seed 0, hence the same batch order. Hypothesis: a specific batch around iteration 600 triggers the episode. Round 9 changes only the seed to test it.

### Screening round 9 at 1500 iterations, one factor changed from base 08

Factors: global GAN loss weight 0.05; feature matching weight 0.1; `manual_seed: 1` (different data order, to test the batch hypothesis).

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 08 (R1 every 4 iterations, from round 8) | 0 | 12, 595–1429 | 5, 597–604 | 24.37 | 663 | 823 | Reference |
| Global GAN loss weight 0.05 | 0 | 6, all at 699–712 (largest 15) | 1, at 706 (fake_score −143) | 12.48 → 24.64, rising | 367 | 788 (713–1500, to the end) | Best of all runs: one weak 13-iteration episode, then clean to the end |
| Feature matching weight 0.1 | — | — | — | — | — | — | Not run: CUDA unavailable (see below); run on base 09 after the reboot |
| `manual_seed: 1` | — | — | — | — | — | — | Not run: CUDA unavailable (see below); run on base 09 after the reboot |

New base (base 09): base 08 with global GAN loss weight 0.05. Its episode again falls in the 580–720 range, consistent with the batch hypothesis, which the seed variant was going to test.

**Interruption.** At 2026-09-15 07:02:45, `unattended-upgrade` upgraded the NVIDIA user-space packages from 580.159.03 to 580.173.02 while the loaded kernel module stayed at 580.159.03. From then on every new process fails CUDA initialization (`cudaGetDeviceCount()` error 804, `nvidia-smi`: "Driver/library version mismatch"), so the last two round 9 variants failed before their first iteration. Training resumes after a reboot (or a reload of the NVIDIA kernel module).

After the reboot the kernel module and user space are both 580.173.02, the NVIDIA and kernel packages are held, and the environment was rebuilt with the same versions.

### Screening round 9 (continued) at 1500 iterations, one factor changed from base 09

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 09 (global GAN loss weight 0.05, seed 0) | 0 | 6, all at 699–712 | 1, at 706 | 24.64 | 367 | 788 (to the end) | Reference |
| `feature_matching_weight: 0.1` | 0 | 20, 588–1498 | 4, at 41 and 1441–1450 (min fake_score −1236) | 12.12 → 23.39, then 23.18 | 1.7e3 | 838 (603–1440) | Worse: a second episode at 1441–1450 and PSNR stops rising |
| `manual_seed: 1` | 0 | 37, 474–1069 | 5, at 480–488, 825–831 and 1069 (min fake_score −2866) | 12.89 → 23.38, rising | 4.1e3 | 473 (1–473); clean from 1070 to the end (431) | Worse: three episodes at different iterations |

The seed changes both the data order and the initialization. With seed 1 the episodes move to 474, 825 and 1069, so they are not tied to one batch around iteration 600. The global discriminator confidence episodes are intrinsic to the setup, and their number varies strongly between seeds, so base 09's single episode with seed 0 is partly luck.

Base 09 stays the best configuration. Run 04 trains it for 5000 iterations to see whether the episodes stop once the generator has learned the coarse structure (in base 09 and seed 1 the last 431–788 iterations were clean).

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 04 | base 09 for 5000 iterations | base 09 (seed 0) | Not stable: no NaN, 60 generator spikes, 10 score violations (last at 4079), no PSNR drop > 1 dB; PSNR 12.57 → 25.45 dB at 4000, then 25.27 at 5000; generator gradient norm median 2.85, max 2.5e5; longest clean streak 1182 (2890–4071) |

Episodes do not stop with training; they recur every 500–1500 iterations. The first one came at 816, against 699 in the base 09 screen with the same config and seed, so training on the GPU is not deterministic.

Two kinds of events show up in the per-iteration losses of run 04 and the round 9 screens:
- Discriminator-led episodes: fake_score drifts to −20 to −60 for a few iterations before the generator losses jump (816–819 in run 04, 480 and 825–831 with seed 1). Most events are of this kind.
- Single-iteration generator output explosions: at 2880 in run 04 the pixel loss goes from 0.02 to 914 (fake_score −8.3e4, gradient norm 2.5e5) and is back to normal at the next iteration, with fake_score −0.7 just before; iteration 1441 of the feature matching 0.1 screen is similar (pixel loss 27). Weights change very little in one Adam step, so these point at specific degraded inputs. The generator has two unbounded paths: the SFT modulation (`out * scale + shift`, scale from an unconstrained convolution) and the RGB output.

Input probe: the run 04 generator at iteration 3000 (training weights, train mode) was run on 2468 degraded training samples (617 faces, 4 degradation draws each). Max |output| has median 0.93, 99th percentile 1.07 and maximum 1.17; max |SFT scale| stays at 0.95–1.67 for the worst samples; L1 to GT is at most 0.32. No input makes these weights explode, so the single-iteration explosions are not input-dependent for a given weight state. They come from the weight updates themselves: with Adam (β1 = 0, β2 = 0.99 in `GFPGANModel`), a gradient much larger than its recent history moves every parameter by up to lr / sqrt(1 − β2) = 10 × lr in one step, and gradient norm clipping does not change that, because Adam is invariant to the gradient scale.

### Screening round 10 at 3000 iterations, one factor changed from base 09

Longer screens, because run 04 shows that 1500 iterations often end before the second episode and that the same config gives different episode timing. Reference: the first 3000 iterations of run 04. Factors that reduce the step size or the adversarial pressure: generator lr 5e-5; global discriminator lr 1e-5; global GAN loss weight 0.02.

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 09 (run 04, iterations 1–3000) | 0 | 44, 816–2889 | 8, 817–2883 (min fake_score −8.3e4) | 12.57 → 25.28, then 25.26 | 2.5e5 | 815 (1–815) | Reference |
| Generator lr 5e-5 | 0 | 1, at 2619 (G total 3.0) | 0 | 12.05 → 25.41, rising | 107 | 2618 (1–2618); clean from 2620 to the end (381) | Best of all runs: one small isolated spike, no score violation, gradient norm 2000x smaller, PSNR not lower |
| Global discriminator lr 1e-5 | 0 | 32, 913–2691 | 4, at 913–916 and 2676–2678 (min fake_score −696) | 12.52 → 25.41 | 1.9e3 | 1018 (1644–2661) | Slightly better than the reference (fewer spikes and score violations), far behind generator lr 5e-5 |
| Global GAN loss weight 0.02 | 0 | 35, 586–2685 | 5, at 587–594 and 2172 (min fake_score −612) | 12.65 → 25.44 | 990 | 680 (818–1497) | Slightly better than the reference, far behind generator lr 5e-5 |

New base (base 10): base 09 with generator lr 5e-5 (generator 5e-5, global discriminator 2e-5, gradient clipping 10, no facial component discriminators, feature matching 1.0, global GAN weight 0.05, R1 weight 10 every 4 iterations, batch 2, seed 0). Lowering the generator step size is the only factor that removes the episodes instead of thinning them, consistent with the Adam step explanation above. Run 05 trains base 10 for 10,000 iterations against the 5000-iteration criterion.

**Correction to round 9.** Event clusters (generator spikes or |score| ≥ 100, merged within 20 iterations) recur at the same iterations across seed-0 runs with different settings: 1498–1510 in run 04 and in the feature matching 0.1, discriminator lr 1e-5 and GAN weight 0.02 screens; 2878 in run 04 and 2884 in run 05; 4072 in run 04 and 4079 in run 05; 586–588 in two screens; 816–817 in two runs. With seed 1 the events moved. The seed sets both the network initialization and the random degradations drawn by the data workers, so the events are tied to specific training samples (a face together with its degradation), not to one face alone. Round 9 wrongly concluded that they are not tied to the data. This is also why the input probe above, which drew new degradations, found no explosive input.

No single face is responsible. BasicSR's `EnlargedSampler` shuffles with the epoch number as seed (independent of `manual_seed`), and the data workers are re-created every epoch with `numpy` and `random` seeded by `manual_seed + worker_id`; the color jitter on tensors uses the worker `torch` generator, which is not reproducible across runs. The 29 event clusters of the seed-0 runs cover 106 distinct batches (the batch of the first event iteration and the 3 before it) with 176 distinct faces. The most frequent face appears in 3 of them, and the three most frequent sum to 9; for the same number of random batches over the same iteration range the medians are 3 and 9 (2000 draws). The events depend on the degraded sample and on the state of the networks, not on a particular face.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 05 | base 10 for 10,000 iterations | base 10 (seed 0) | Stopped at 5314 iterations, when the current clean streak (from 5115) could no longer reach 5000 before 10,000. No NaN, 22 generator spikes and 3 score violations in 7 clusters (471–496, 1009–1022, 1625, 2541, 2884, 4079–4095, 5113–5114); max pixel loss 2.6 (914 in run 04); min fake_score −1138 at 4079; generator gradient norm median 3.3, max 3.3e3; longest clean streak 1194 (2885–4078); PSNR 12.22 → 25.53 dB at 4000, then 25.23 at 5250 (no drop > 1 dB) |

Run 05 has half the spikes of run 04 over the same length and no generator output explosion, but discriminator-led episodes still recur every 500–1100 iterations. Most are small (generator loss total 0.8–3 against a median of 0.08, so just above the 10x threshold); the one at 4079–4095 is large.

### Screening round 11 at 3000 iterations, one factor changed from base 10

Factors: generator lr 2.5e-5 (continues the only trend that removed episodes); R1 weight 50 (stronger penalty against discriminator overconfidence; the round 4 test at 50 was confounded by the facial component path); global discriminator lr 1e-5 (tested on base 09 in round 10, now on the smaller generator step).

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Max generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 10 (run 05, iterations 1–3000) | 0 | 17, 471–2884 | 1, at 1014 | 12.22 → 25.32 | 206 | 915 (1626–2540) | Reference |
| Generator lr 2.5e-5 | 0 | 0 | 0 | 11.50 → 24.96, rising more slowly | 91 | 3000 (the whole run) | First run without any violation. PSNR at 3000 is 0.36 dB lower than base 10: slower learning, still rising |
| R1 weight 50 | — | — | — | — | — | — | Stopped after 41 iterations to start the long run of generator lr 2.5e-5 |
| Global discriminator lr 1e-5 | — | — | — | — | — | — | Not run, for the same reason |

New base (base 11): base 10 with generator lr 2.5e-5. Run 06 trains base 11 for 10,000 iterations against the 5000-iteration criterion.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 06 | base 11 for 10,000 iterations | base 11 (seed 0) | **Stable.** 10,000 iterations: no NaN, no \|score\| ≥ 100 (max \|fake_score\| 18.9, max \|real_score\| 7.8), one generator spike at 350 (G total 1.15 against a median of 0.11), no PSNR drop > 1 dB (largest drop between validations 0.18 dB); clean streak 9650 iterations (351–10,000); generator gradient norm median 3.8, max 83; max pixel loss 0.078; PSNR 11.61 → 25.16 dB at 3750, 24.29 at 6750, 24.51 at 10,000 |

## Stable configuration

Base 11 meets the criterion: `options/train_gfpgan_clean.yml` with
- generator lr 2.5e-5, global discriminator lr 2e-5;
- `generator_grad_clip: 10`;
- no facial component discriminators and `comp_style_weight: 0`;
- feature matching weight 1.0, global GAN loss weight 0.05;
- R1 weight 10 every 4 iterations (`net_d_reg_every: 4`);
- batch 2 on one GPU, seed 0.

Open issues:
- Validation PSNR peaks at 25.16 dB at iteration 3750, declines to 24.29 dB at 6750 and partly recovers to 24.51 dB at 10,000, without any drop above 1 dB between validations. With 617 training faces this is consistent with overfitting; the criterion does not cover it.

`options/train_gfpgan_clean.yml` now uses these settings, with batch 2 and 4 data workers per GPU as in run 06.

### Screening round 12 at 3000 iterations: facial component discriminators on base 11

Re-enables the three `FacialComponentDiscriminatorClean` networks on base 11, with the component Gram style loss still off (`comp_style_weight: 0`). Variants: upstream component settings (component GAN loss weight 1, component discriminator lr 2e-4); component GAN loss weight 0.1; component discriminator lr 2.5e-5. Reference: run 06, iterations 1–3000 (one spike at 350, no score violation).

| Variant | NaN/inf | G spikes | \|score\| ≥ 100 | PSNR (dB) | Generator gradient norm | Longest clean streak | Verdict |
|---|---|---|---|---|---|---|---|
| Base 11 (run 06, iterations 1–3000) | 0 | 1, at 350 | 0 | 11.61 → 25.01, rising | median 4.3, max 83 | 2650 (351–3000) | Reference |
| Component discriminators, upstream settings | 0 | 2, at 1156 and 2855 (G total 22 and 26 against a median of 2.2–2.5) | 0 | 10.91 → 22.46 at 2500, then 21.96 | median 33, max 3.0e3 | 1698 (1157–2854) | Worse: the component GAN terms (median about 0.75 each) dominate the generator loss, PSNR is 3 dB lower. Both spikes come from one component: right eye (13.0) at 1156, mouth (20.8) at 2855, while the global fake_score stays small (−18, −0.5) |
| Component discriminators, component GAN loss weight 0.1 | 0 | 0 | 0 | 11.35 → 23.72 at 2250, then 23.67 | median 7.2, max 377 | 3000 (the whole run) | No violation; PSNR 1.3 dB lower than base 11 at 3000 and flat since 2250 |
| Component discriminators, component discriminator lr 2.5e-5 | 0 | 0 | 0 | 11.55 → 23.90, rising | median 7.0, max 797 | 3000 (the whole run) | No violation; PSNR 1.1 dB lower than base 11 at 3000 and still rising |

Both lower-pressure variants are free of violations. The PSNR cost against base 11 is expected: the component losses target eye and mouth realism, not pixel fidelity, so PSNR alone does not measure their benefit.

New base (base 12): base 11 with the three facial component discriminators at component discriminator lr 2.5e-5 (component GAN loss weight 1, component style weight 0). It keeps the upstream component loss weight and has the higher, still rising PSNR. Run 07 trains base 12 for 10,000 iterations against the 5000-iteration criterion.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 07 | base 12 for 10,000 iterations | base 12 (seed 0) | **Stable.** 10,000 iterations with no violation at all (no NaN, no generator spike, no \|score\| ≥ 100, no PSNR drop > 1 dB); clean streak 10,000, the only run clean from the first iteration. Validation PSNR 11.51 → 24.50 dB at 3250, then 23.37 at 10,000 |

## Evaluation metrics

PSNR measures pixel fidelity and favors smooth outputs; it does not measure what the GAN and facial component losses are for (realistic detail), and 32 validation images make differences of a few tenths of a dB noise. In the stability criterion PSNR only detects collapse. Checkpoints are also evaluated with metrics that need no weights with commercial-use restrictions (LPIPS, FID and identity distance rely on ImageNet or ArcFace networks and are not used), by an evaluation script kept outside the repository:
- component PSNR: PSNR inside the left eye, right eye and mouth boxes of the ground truth, averaged;
- landmark distance (LMD): mean distance in pixels between the 478 MediaPipe Face Mesh landmarks of the output and of the ground truth, over the outputs where a face is found (failures are counted separately);
- NIQE (BasicSR implementation, no reference; lower is better).

References on the 32 validation pairs: the degraded inputs have PSNR 24.90 dB, component PSNR 23.54 dB, LMD 8.50 px (no face found in 6 of 32) and NIQE 12.80; the ground truth has NIQE 5.68.

| Run | Iteration | PSNR (dB) | Component PSNR (dB) | LMD (px) | LMD failures | NIQE |
|---|---|---|---|---|---|---|
| 06 (base 11) | 1000 | 21.42 | 20.69 | 11.95 | 7 | 12.87 |
| 06 | 3000 | 25.01 | 23.58 | 9.52 | 4 | 11.19 |
| 06 | 4000 | 25.10 | 23.30 | 7.65 | 5 | 10.56 |
| 06 | 6000 | 24.39 | 22.10 | 7.99 | 6 | 8.54 |
| 06 | 8000 | 24.36 | 22.17 | 9.28 | 4 | 6.96 |
| 06 | 10,000 | 24.51 | 22.33 | 6.83 | 4 | **6.38** |
| 07 (base 12) | 1000 | 15.78 | 15.23 | 12.86 | 30 | 16.28 |
| 07 | 3000 | 24.32 | 22.69 | 9.00 | 5 | 10.97 |
| 07 | 5000 | 23.77 | 20.26 | 9.99 | 5 | 8.82 |
| 07 | 8000 | 23.04 | 20.84 | 8.13 | 7 | 9.82 |
| 07 | 10,000 | 23.37 | 20.90 | 12.20 | 7 | 9.94 |

Reading:
- PSNR peaks early (4000 in run 06) and then falls while NIQE keeps improving, from 12.87 to 6.38, close to the 5.68 of the ground truth. PSNR gives the wrong signal here: the generator is replacing blur with texture. Base 11 also ends below the degraded input on PSNR (24.51 against 24.90) while being much closer to it on NIQE and better on LMD (6.83 against 8.50).
- Base 12 (facial component discriminators at lr 2.5e-5) is stable but worse on every quality metric: NIQE bottoms at 8.82 at 5000 and rises again to 9.94, and component PSNR, the region those discriminators target, stays 1.4–2.4 dB below base 11 from iteration 4000 on. On this dataset they cost quality instead of adding detail.
- LMD failures (3–11 of 32, and 30 at run 07 iteration 1000) make LMD comparisons weak: the mean is taken over the images where MediaPipe finds a face.

Conclusion: `options/train_gfpgan_clean.yml` keeps base 11, without the facial component discriminators. They train stably at component discriminator lr 2.5e-5 (run 07 has no violation in 10,000 iterations) and can be re-enabled, but they need a larger dataset and a quality benchmark before they earn a place in the default config.
- The facial component losses of GFP-GAN are disabled. They were the first source of instability and have not been re-tested on the stable base.
- Stability was shown for one seed and one small dataset.

## The model damages a face that is already good

The background model had this defect and it was measured and corrected in `background_super_resolution.md`: trained only on heavy degradation, it read genuine texture as noise and scored below plain Lanczos once the input was clean. The face path uses the same style of degradation pipeline, so the same failure was plausible and had never been measured.

There is no Lanczos here, because the face model reconstructs at a fixed 512x512 rather than upscaling. The baseline is therefore the degraded input itself, passed through untouched: on a near-identity degradation a healthy restorer should stay at or above that line. Measured on the 256 held-out faces of `ffhq512full/val_gt`, verified disjoint from the 69,674 used for training, over the first 64:

| Regime | Untouched input | Model @100k | Model wins |
|---|---|---|---|
| Training degradation | 20.70 / 0.6133 | **22.85** / 0.6185 | 41 / 64 |
| Near identity | **38.61 / 0.9680** | 26.34 / 0.7504 | **0 / 64** |

In its own regime the model does its job, beating the untouched input by 2.15 dB on average. On a clean input it scores **12.28 dB below doing nothing at all**, and loses on every single face. For scale, the background model lost 1.06 dB in the same situation.

The training-degradation row replaces an earlier one that read 21.14 / 22.69. That row was not reproducible and should not be cited: it was produced before the seeding trap below was understood, so its blur kernels came from whatever ambient state the process happened to be in. The near-identity row is unaffected, because its kernel list has a single entry, and it reproduces to the digit.

**The average gain is not the typical gain.** The +2.15 dB is a mean over a strongly right-skewed distribution; the *median* face gains only +0.74 dB. A minority of badly degraded faces improve enormously and carry the average. Any single number quoted for this model should say which one it is.

A generative restorer resynthesises even a good face, and PSNR and SSIM punish resynthesis harshly, so the metrics alone would not settle whether this is damage or an artefact of the measurement. Inspection settles it: the output shows colour fringing along hair edges, mangles background foliage into coloured noise, and reworks skin and glasses, while the untouched input is plainly closer to the ground truth.

**The practical severity is lower than the numbers suggest.** Most of the visible damage is in hair and background, and `GFPGANer` pastes back only the aligned face region through a feathered mask, so much of it never reaches the output. That reduces the urgency; it does not make the model correct.

### The remedy, measured

`mild_prob` on `FFHQDegradationDataset` draws a sample whose degradation is close to the identity, the same correction already validated for the background. It is implemented in the dataset rather than the model because that is where face degradation is built, and it selects parameters into local names instead of writing them onto `self`: the dataset runs in several worker processes, and mutating instance state per sample would be a race. It defaults to 0, leaving every existing config byte-identical.

Three arms were finetuned from `net_g_100000.pth` for 5,000 iterations, with configurations that differ in exactly one key, verified mechanically rather than by reading them: a control at `mild_prob` 0, a treatment at 0.5, and a third drawing a continuous severity from Beta(0.5, 1), the law CResMD (arXiv:1912.05293) uses for the same purpose. Learning rates were the decayed values in force at iteration 100,000, and `remove_pyramid_loss` was set to 0 so the pyramid loss stayed off as it was at that point: the iteration counter restarts at zero on a finetune, so leaving it at 50,000 would silently have switched a loss back on.

One difference between the code that was measured and the code that ships is worth knowing before reproducing this. The arms ran from a working tree that expressed the mild sample as a continuous severity, whose severity-zero path sets `jpeg_range` to [100, 100]; the committed branch sets it to [99, 100]. Every other parameter of the mild sample is identical. The direction is conservative: the arms trained on samples marginally cleaner than the near-identity regime they were then judged on, which uses [99, 100].

Against the untouched input, on the near-identity regime where the defect lives:

| Model | PSNR / SSIM | vs untouched input | Wins |
|---|---|---|---|
| Model @100k | 26.34 / 0.7504 | -12.276 dB | 0 / 64 |
| Control @5k | 26.22 / 0.7453 | -12.398 dB | 0 / 64 |
| **`mild_prob` 0.5 @5k** | **28.13 / 0.7788** | **-10.487 dB** | 0 / 64 |
| Beta(0.5, 1) @5k | 28.12 / 0.7782 | -10.499 dB | 0 / 64 |

Read as paired per-image deltas, which is what the arms must be judged on:

- **The control earns its cost.** 5,000 further iterations of the unchanged recipe do not repair the defect; they deepen it slightly, by 0.122 dB [0.059, 0.191], losing to the original on 44 of 64 faces. Any improvement in the other arms is therefore attributable to the mixture and not to the extra training.
- **`mild_prob` 0.5 beats the control by 1.911 dB** [1.762, 2.075], median 1.837, **on 64 of 64 faces**, and also wins the degraded regime by 0.138 dB at the median on 44 of 64. There is no trade-off between the two regimes here, which the perception-distortion tradeoff would have permitted but does not require.
- **The continuous law buys nothing.** Beta(0.5, 1) against the two-point draw is +0.012 dB [-0.043, +0.066], p=0.52, 32 of 64 -- a tie bounded to within 0.066 dB, not a tie for want of statistical power. At 1,250 iterations the simpler law was marginally ahead (+0.082 dB, p=0.0015). The extra option, the extra dataset branch and the four-parameter interpolation are not paid for.

**What it does not do.** It closes 1.9 dB of a 12.4 dB hole, about 15%, and the returns diminish sharply: going from 1,250 to 5,000 iterations, four times the compute, moved the treatment only 0.43 dB further. Closing the rest by this route alone is not plausible. The model still loses to its own untouched input on every clean face.

**One cost to record.** On the degraded regime the treatments lose a little SSIM while gaining PSNR (0.6167 and 0.6085 against the control's 0.6182), even as SSIM improves markedly on clean input (0.7788 against 0.7453). The gain is not free on every metric.

The whole experiment cost 3h04 of GPU, of which 27 minutes were wasted: the first attempt was killed for system memory in the middle of writing a checkpoint and left a truncated, unreadable file. Checkpoints were written every 1,250 iterations thereafter, which made a later kill cost almost nothing.

### The rest of the hole is bought at inference, not in training

Training can only go so far, so the remaining trade is exposed as a dial. `gfpgan/utils.py:blend_restoration`
blends the restored face back toward the aligned input: at 1 the output is the restoration, at 0 the input
untouched. This is network interpolation (arXiv:1811.10515) taken to the image, and it costs no training.

The flag for it already existed. `inference_gfpgan.py` has always had `-w/--weight`, default 0.5, advertised as
"Adjustable weights" -- and it did nothing. `GFPGANv1Clean.forward` accepts `**kwargs` and never reads it, and
`RestoreFormer` does the same, so the value was discarded on both architectures. It is inherited from upstream,
not introduced here; `master` has the identical signature and the removed non-clean architecture did not use it
either. No test covered it and no document mentioned it, which is how it survived.

Swept over 64 held-out faces with the `mild_prob` 0.5 model, against the untouched input:

| weight | Degraded: PSNR / SSIM / wins | Near identity: PSNR / SSIM / wins |
|---|---|---|
| 0 | 20.70 / 0.6133 / -- | 38.61 / 0.9680 / -- |
| 0.25 | 21.61 / 0.6403 / **64 of 64** | 35.85 / 0.9449 / 0 of 64 |
| 0.5 | 22.40 / **0.6462** / 62 of 64 | 32.71 / 0.9000 / 0 of 64 |
| **0.75** | 22.94 / 0.6372 / 58 of 64 | 30.17 / 0.8421 / 0 of 64 |
| 1 | **23.01** / 0.6168 / 44 of 64 | 28.12 / 0.7789 / 0 of 64 |

**The default is 0.75 because 1 is dominated, not because 0.75 is a compromise.** Against 1, at 0.75 the
degraded regime is statistically indistinguishable (-0.069 dB, [-0.306, +0.153], p=0.29) while its median and
its SSIM are both better, and the near-identity regime improves by 2.044 dB on 64 of 64 faces. Going further
down to 0.5 does cost the degraded regime for real (-0.540 dB, p=0.0078), so the dial stops there.

**The mean hid this.** On the degraded regime the mean keeps rising to weight 1, which is why full restoration
looks best if only means are read. The win count falls monotonically over the same range, from 64 of 64 at 0.25
to 44 of 64 at 1: at full strength the model loses to its own input on 20 of 64 *degraded* faces. The mean is
carried by a minority of badly degraded faces that gain several decibels; the typical face does better blended.

**Three honest caveats.** First, PSNR and SSIM reward blending toward the input almost by construction when the
input is already close to the ground truth, so the near-identity column overstates how much is really won there;
what the dial trades away is resynthesised detail, which these metrics punish rather than credit. Second,
blending two images can ghost edges that neither shows alone. The measurement says 0.75 is safe on fidelity, not
that it is sharper. Third, the sweep ran on aligned 512 crops fed straight to the network, which is the
`--aligned` path. On the whole-image path the face is additionally warped to 512 and warped back through a
feathered mask, so weight 0 returns the aligned crop rather than the original photograph, and the resampling
that surrounds the blend is not part of what was measured. The best weight also depends on the model -- for the
100k checkpoint 0.75 beats 1 on the mean as well -- which is the argument for a dial rather than a constant.

### A seeding trap found while testing this

`basicsr` draws the blur kernel type with `random.choices`, from the standard library generator rather than numpy or torch. A test that seeds only `np.random` and `torch` leaves that draw to whatever ambient global state the process is in, so two identically configured datasets pick different kernels and disagree. This made the equivalence test pass inside the full suite and fail when run on its own, producing two stable values that swapped places with construction order. Two plausible diagnoses came first and neither survived measurement: a shared `io_backend` dictionary, and a first-call effect from building the `FileClient`. Any test over this dataset must seed all three generators.

The trap was not confined to tests. The training-degradation row of the table above had to be replaced because the measurement that produced it seeded only numpy and torch, so its kernels were drawn from ambient state and the numbers could not be reproduced. A published measurement, not just a flaky test, had been quietly contaminated. `eval_face_mild.py` now seeds all three generators in one function, and re-running it twice returns figures identical to the last decimal.

## Run 13: the published recipe on the licensed Commons corpus

The licensed corpus finally exists -- 1,414 training faces and 64 validation, collected from Wikimedia Commons under
CC0, public domain, CC BY and CC BY-SA and reviewed by hand (`commons_face_corpus.md`). At 2% of FFHQ it cannot make a
good prior, so the question it can answer is whether the published config stays stable on it.

| Run | Change from previous | Settings | Result |
|---|---|---|---|
| 13 | base 12 on the licensed Commons corpus, with `mild_prob` 0.5 | the published config otherwise unchanged: 1,414 training faces, component lr 2.5e-5, batch 2, 10,000 iterations | **Stable.** Clean streak 10,000 from the first iteration, no violation on any of the four checks. Validation PSNR 11.79 → 22.71 over 41 validations with no drop above 1 dB. 1h43 of GPU |

The data regime is the point of interest. 1,414 faces at batch 2 is 707 iterations per epoch, so the model sees each
face about fourteen times over the run, against roughly once in run 09 on 9,927 FFHQ faces. High repetition did not
destabilise it, and the validation PSNR plateaus from about iteration 3,000 without regressing. That plateau is not
comparable to run 09's 21.18: the validation set is 64 faces from this corpus, not from FFHQ.

Two factors changed from run 09 at once, the corpus and `mild_prob` 0.5, because the question was about the recipe as
published rather than about either factor alone. The stable result makes bisection unnecessary: if both together are
stable, neither is destabilising. A negative result would have required taking them apart.

**The stability is held by the gradient clip rather than intrinsic.** `g_grad_norm` before clipping has a median of
3.92 and a maximum of 499, so the run contains gradient spikes two orders of magnitude above typical and
`generator_grad_clip: 10` absorbed every one. No criterion was violated, which is the clip doing its job, but anyone
considering removing it should know this run would likely fail the spike check without it.
