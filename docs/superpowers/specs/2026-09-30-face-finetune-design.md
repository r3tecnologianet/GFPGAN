# Fine-tuning the face model for real photographs and dark skin

Design approved in chat on 2026-09-30 (approach 1 of three: staged fine-tuning of the existing generator, rather than
fine-tuning upstream's v1.4 or pretraining a StyleGAN2 prior first).

## Scope and licensing

The starting point is the run 10 generator, `gfpgan_clean_ffhq_100k.pth` (EMA weights at 100,000 iterations), and the
training data is FFHQ. Every model produced here is therefore **for local use only**, like its starting point. Nothing
trained here is committed, released or given a download URL. Evaluation sets live under
`/mnt/dados/gfpgan-clean/eval/`, outside the repository, and are named here without links.

## Goals, in priority order

1. Real photographs: fix the melted mouth and beard detail and the nose deformation seen on unaligned photographs
   (`training_stability.md`, comparison with v1.4).
2. Dark skin: close the quality gap measured in stage 0.
3. Guardrail: no regression on the existing metrics (PSNR, component PSNR, LMD, NIQE) in any skin tone group.

## Stage 0 results (done)

**Alignment does not reproduce FFHQ's.** `FaceHelper` run on 1,000 already aligned FFHQ faces should return the
identity; it returns a face 4.4% too small, shifted about 8 px up, with the mouth keypoint 14 px below the template,
and a per-face jitter of 5° rotation and 6.5% scale (centre shift median 10 px, 90th percentile 16 px). The BlazeFace
nose tip varies by 24 px between faces, and FFHQ's alignment does not use the nose. Measured on 1,000 held-out faces
embedded at 256 px in a 1024 canvas:

| Keypoints | Fit | Scale | Rotation std | Centre shift median / p90 |
|---|---|---|---|---|
| BlazeFace, 4 points, current template | LMEDS | 0.957 ± 0.065 | 5.1° | 10.0 / 16.0 px |
| BlazeFace, eyes and mouth, refitted template | least squares | 1.000 ± 0.064 | 4.1° | 4.8 / 9.4 px |
| Face Landmarker eye and lip centres, refitted template | least squares | 0.999 ± 0.047 | 1.9° | 2.2 / 5.4 px |

Tools and raw results: `tools/align_*.py`, `eval/align_*.json`.

**Skin tone labels.** ITA (individual typology angle) was rejected: uncontrolled illumination puts light-skinned FFHQ
faces at ITA -90 to -60. The 256 validation faces were labelled by eye into three groups (220 light, 22 medium, 14
dark) in `eval/tone_labels_val.csv`. The labels are subjective and 14 dark faces only resolve large differences.

**Baseline, run 10 generator, gain over the degraded input** (95% bootstrap intervals, raw network output at weight 1):

| Group | n | PSNR gain | Component PSNR gain | Worse than input | Cheek ΔL* |
|---|---|---|---|---|---|
| Light | 220 | +1.33 dB [+0.90, +1.79] | +1.45 dB | 87 | +0.08 |
| Medium | 22 | +0.70 dB [-0.03, +1.45] | +0.56 dB | 10 | -0.59 |
| Dark | 14 | -0.14 dB [-0.67, +0.39] | -0.01 dB | 8 | +0.26 |

The model gains nothing on dark skin, and the light and dark intervals do not overlap. It does not lighten skin
(ΔL* near 0): the failure is restoration quality, not colour.

**Pending, blocked outside this design:** the per-group v1.4 baseline (its checkpoint load needs the user to run it)
and the real-photograph sets WebPhoto-Test, CelebChild-Test and LFW-Test (their hosting requires a browser login).

## Alignment fix (inference only)

`FaceHelper` aligns with the Face Landmarker's eye and lip centres (the model is already in `gfpgan/weights`), without
the nose tip, by least-squares similarity to a template refitted on FFHQ. BlazeFace still finds the faces. Acceptance:
the alignment check above reproduces the third row, and the existing face helper tests pass.

## Stage 1: fine-tuning with a harder degradation

- **Initialisation:** generator and EMA from the run 10 weights. The discriminators (global and the three facial
  component ones) were deleted with the experiment folder and start from scratch.
- **Discriminator warm-up:** the first 2,000 iterations update only the discriminators, with the generator frozen.
- **Degradation:** a new option `second_order_prob` in `FFHQDegradationDataset`. With that probability (0.5 here) a
  sample gets the two-pass Real-ESRGAN degradation, built from `basicsr.data.degradations` on the CPU, instead of the
  current single pass. `mild_prob` stays 0.5, as in the published config.
- **Alignment jitter:** training crops are perturbed by the residual error left after the alignment fix: rotation
  ±2°, scale ±5%, shift ±3 px.
- **Schedule:** learning rates at run 10's final values, 50,000 iterations (about 8 hours at 0.6 s/iteration on the
  RTX 3060 Ti), validation per tone group every 5,000.
- **Acceptance:** better NIQE and LMD, and on the real sets once available, with no group losing more than 0.2 dB of
  component PSNR against the baseline above.

## Stage 2: upweighting the faces the model fails on

Tone rebalancing of 70,000 faces is not possible without reliable labels, so this stage uses JTT (Just Train Twice,
Liu et al. 2021, arXiv:2107.09044), which needs none:

1. The stage 1 model restores the whole training set.
2. The 10–20% of faces with the lowest gain over their degraded input form an error set.
3. Fine-tuning continues with the error set sampled 3–5 times more often.

Before training, 200 faces of the error set are labelled by eye to check that dark skin is over-represented there; if
it is not, this stage is re-designed rather than run. Acceptance: the dark group's gain becomes positive with its
interval clear of zero, without the light group losing more than 0.2 dB.

## Testing

- Unit tests for `second_order_prob` (0 reproduces the current pipeline exactly; 1 differs; options are not
  mutated), the alignment jitter, and the JTT sampler weights.
- Training stability is judged by the criterion already used in `training_stability.md`.
- Results are appended to `training_stability.md` as new runs.

## Constraints

- The GPU is shared: no run starts while another process holds it, and no other process is stopped.
- No restricted networks enter the evaluation (no LPIPS, FID or ArcFace), as in `training_stability.md`.
