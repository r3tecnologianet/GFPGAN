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
