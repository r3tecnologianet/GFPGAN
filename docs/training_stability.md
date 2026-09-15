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
