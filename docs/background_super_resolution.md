# Background super-resolution without restricted weights

`inference_gfpgan.py` restores faces and upscales everything else with Lanczos, because the Real-ESRGAN release
weights are trained on DIV2K, Flickr2K and OST, which are published for academic research only. The code of
Real-ESRGAN is BSD-3-Clause and its architectures are plain PyTorch, so the blocker was never the method: it was
the weights. This document records the plan to train a replacement, the same way the face model was trained.

## What is reused and what is replaced

| Piece | Decision |
|---|---|
| Degradation | `RealESRGANDataset` from basicsr: second-order blur, resize, noise, JPEG and sinc, all synthetic, so it needs no data beyond clean photographs |
| Generator | `SRVGGNetCompact` (1.21M parameters). A background does not need the detail a face does, and inference stays cheap; `RRDBNet` (16.7M) is the fallback if quality is short |
| Discriminator | `UNetDiscriminatorClean` in `gfpgan/archs/discriminator_arch.py`: same shape as the Real-ESRGAN discriminator, plus `return_feats` |
| Perceptual loss | Replaced by discriminator feature matching in `RealESRGANCleanModel`, as in `GFPGANModel`. The model also forces `cri_perceptual` to None, so a stray `perceptual_opt` cannot pull an ImageNet network into training |
| Config | `options/train_realesrgan_clean.yml`, checked by `tests/test_realesrgan_config.py` |

## Data

Megalith-10m (`madebyollin/megalith-10m`): 9.58M links to Flickr photographs filtered to "no known copyright
restrictions", US Government Work, CC0 or Public Domain Mark; the link list itself is MIT. Images are fetched
with `tools/fetch_sr_data.py` (outside the repository) at a controlled rate, and the original JPEG bytes are
stored, never re-encoded: recompressing the ground truth would add the artefacts the model must remove.

Measured on the links before collecting:
- availability: 47 of 50 links answered; the 3 failures are dead links (404, 410). Requests must be paced —
  200 HEAD requests and 60 parallel GETs both returned 0 of 0, and sequential requests then worked, so the zero
  was rate limiting, not link rot;
- size: median 143 KiB per image, about 1024 px on the long side.

Quality gate, with thresholds taken from the measured distribution of 475 photographs rather than guessed:

| Signal | Distribution | Threshold |
|---|---|---|
| Shorter side | median 683 px | ≥ 448 px |
| Sharpness (variance of the Laplacian) | p10 140, median 757 | ≥ 100 |
| Halftone band energy | min 0.47, median 0.65, max 0.76 | ≤ 0.72 |

The halftone signal does not separate print from photograph at this normalisation: a first attempt with a guessed
threshold of 0.45 rejected 455 of 500 images, all of them ordinary photographs. It now only cuts the tail, and the
value is kept in the manifest so the gate can be revisited without downloading again. With these thresholds a
pilot kept 40 of 47 attempts, rejecting 2 dead links, 2 on halftone, 1 blurred and 1 too small.

### A better source for the same photographs: megalith-cc0

`Spawning/megalith-cc0` is Megalith-10m filtered to CC0 alone and rehosted at full resolution on a public S3
bucket under the AWS Open Data Registry. Its card explains the filtering: Public Domain Mark items were dropped
deliberately, because an uploader choosing that mark may not understand its implications, while CC0 is an
unambiguous dedication.

Measured here rather than taken from the card:

| | collected via Megalith-10m | megalith-cc0 |
|---|---|---|
| Short side | median 683 px | median 3000 px; 75% at or above 2000 px, 94% above 1000 px |
| Pixels per image | 0.47 MP | about 10 MP on the sampled rows |
| Compression | 1.97 bits per pixel | 1.59 to 1.98 bits per pixel |
| Licence per row | mixed CC0, PDM, US Gov, "no known restrictions" | `creativecommons.org/publicdomain/zero/1.0/` on every sampled row |
| Fetching | Flickr links, paced, 3 of 50 dead | public S3, no pacing needed |

The bits per pixel are the same on both sides, so the ground truth collected so far was never over-compressed:
the only deficit is resolution, and it is a factor of about 21 in pixels per image. Megalith-10m's `url_highres`
is the Flickr `_b` suffix at 1024 px and cannot be rewritten to a larger size, because every size from `_h`
upward carries its own per-size secret; reaching them needs `flickr.photos.getSizes` and therefore the Flickr
API terms. megalith-cc0 avoids that entirely.

Verified directly: the schema carries `url`, `width`, `height` and `license`; three downloaded files decoded to
exactly the dimensions their metadata claims; rows exist at offset 2,000,000. Not verified: the exact row count,
and whether every row is CC0 rather than only the sampled ones. The CC0 status still rests on Flickr uploaders
having set their licence field correctly, which is a residual risk and not a warranty.

Having `width` and `height` in the index means a resolution gate can run before any bandwidth is spent, which
the current post-download gate cannot do.

### Truncated downloads pass a decode check

Collecting 20,000 images at 48 concurrent connections left 4.6 per cent of the files without an end-of-image
marker: truncated transfers. They are not caught by decoding, because `cv2.imdecode` fills the missing rows and
returns an image rather than None, which is all the fetcher checked. Sampling the tiles of run 05 found 4.25 per
cent of them cut from such files, and they cleared the sharpness gate only just -- median variance of the
Laplacian 104 against a floor of 60.

`tools/fetch_cc0_data.py` now verifies the trailing marker (`FFD9` for JPEG, `IEND` for PNG) before writing, and
counts the rejects separately. The contamination is recorded as a minor one: at 4 per cent it degrades a run
slightly and comes nowhere near explaining why a twenty-one fold increase in pixels produced no gain.

## Status

Smoke training runs: 12 iterations on collected photographs, with pixel, feature matching, GAN and discriminator
losses all present. The stability criterion and the evaluation metrics are the ones already used for the face
model (`docs/training_stability.md`); the analysis script also reads `out_d_real` and `out_d_fake`, which is what
this model logs instead of `real_score` and `fake_score`.

Measured on an RTX 3060 Ti (8 GiB) at `gt_size` 256, scale 4:

| Batch | Result |
|---|---|
| 12 | 0.95–1.05 s/iteration, so 20,000 iterations take about 5.5 hours |
| 24, 32 | CUDA out of memory |

The memory ceiling comes from the degradation pipeline (large blur kernels and differentiable JPEG on the GPU),
not from the 1.21M-parameter generator.

## Run 01: adversarial from the first iteration

20,000 iterations, batch 12, `gt_size` 256, scale 4, on 14,936 photographs, with 64 held-out validation pairs.

**Stable, and that was never the problem:** 19,901 consecutive iterations with no NaN, no generator loss spike, no
discriminator score above 100 and no validation PSNR drop over 1 dB. Unlike the face model, which needed twelve
screening rounds, the first configuration trained clean.

**The quality is bad.** Averages over the 64 validation pairs, all measured with the same code:

| | PSNR (dB) | SSIM | NIQE |
|---|---|---|---|
| Lanczos (what inference uses today) | **25.36** | **0.7488** | 7.07 |
| ours @ 5000 | 23.98 | 0.7021 | 11.56 |
| ours @ 10,000 | 23.92 | 0.6999 | 12.61 |
| ours @ 15,000 | 23.78 | 0.6890 | 12.12 |
| ours @ 20,000 | 23.75 | 0.6845 | 12.44 |
| official realesr-general-x4v3 | 23.62 | 0.7328 | 5.35 |
| official RealESRGAN_x4plus | 24.19 | 0.7212 | **3.97** |

Two readings, and both matter:

- **The validation set flatters interpolation.** Both official models also lose to Lanczos on PSNR and SSIM, so
  those two metrics cannot decide anything here: a generative model that invents plausible texture is punished by
  them. NIQE is where the official models win by a wide margin, and that is the metric to watch.
- **Our outputs are genuinely bad, and get worse with training.** NIQE goes from 11.56 to 12.44 between 5000 and
  20,000 iterations, worse than plain Lanczos (7.07) and far from the official compact model (5.35), which has
  the same architecture and parameter count.

**Defect in the first validation set.** It was built with a bicubic downscale and a JPEG quality argument passed
while writing a `.png` file, where that argument is ignored — so the inputs carry no compression at all, much
milder than the second-order degradation the model trains on. `tools/make_sr_val.py` now builds validation pairs
by running the training degradation itself (the dataset draws the kernels, the model degrades on the GPU), and
both sets are reported from now on.

**The likely cause of the artefacts: a missing training stage.** Real-ESRGAN does not train with a GAN from
scratch. It trains a pixel-loss-only model first (Real-ESRNet) and starts the adversarial phase from those
weights. Run 01 turned the adversarial loss on at iteration 1.

## Run 02: the two-phase recipe, and why it does not save the result

Phase 1 trained 8000 iterations with the pixel loss only, phase 2 trained 5000 more from those weights with the
adversarial and feature matching losses. Both phases were stable (7901 and 4901 clean iterations, no violation of
any criterion). Averages over the 64 validation pairs of each set:

| | light degradation | | | training degradation | | |
|---|---|---|---|---|---|---|
| | PSNR | SSIM | NIQE | PSNR | SSIM | NIQE |
| Lanczos | **25.36** | **0.7488** | 7.07 | **20.96** | 0.5205 | 10.65 |
| phase 1 (pixel only) | 24.03 | 0.7195 | 9.99 | 20.57 | 0.4733 | 20.09 |
| phase 2 (adversarial) | 23.95 | 0.6925 | 15.24 | 20.53 | 0.4593 | 42.41 |
| official realesr-general-x4v3 | 23.62 | 0.7328 | **5.35** | 20.77 | **0.5799** | **7.97** |

The two-phase recipe helps — phase 1 is better than run 01 on every metric — but it does not fix the result, and
it points at a different cause: **phase 1 has no adversarial loss at all and still loses to Lanczos on every
metric.** A model that only minimises L1 cannot invent artefacts; it can only be too weak, and it is. The
official model, with the same architecture and the same 1.21M parameters, was trained on far more data for far
longer.

**Decision: `inference_gfpgan.py` keeps Lanczos for the background.** On this hardware the trained model is worse
than the interpolation it would replace, and shipping it would make inference worse. The training pipeline,
config, model and evaluation stay in the repository: what is missing is training scale, not code.

Order of magnitude of what would be needed: Real-ESRGAN trains its compact model for hundreds of thousands of
iterations on millions of images; this run had 13,000 iterations on 15,000 photographs, about 4.5 hours on one
RTX 3060 Ti.

## What the runs actually hit: a dead initial phase

"Not enough training" was the wrong diagnosis. Cheap probes, none of which needs a training run, found the real
behaviour in minutes:

| Probe | Cost | Result |
|---|---|---|
| Per-layer gradient, one forward/backward | seconds | With the default initialisation the first body layer gets 2.5e-15 and the last 7.8e-3: a ratio of 3e12. Kaiming initialisation (`default_init_weights`, scale 1) brings it to 4.9e-1 against 1.1e-1, ratio 0.22 |
| Output against plain interpolation | seconds | The run 01 and run 02 generators reproduce nearest upsampling to within 0.01 of 255. `SRVGGNetCompact` adds a nearest-upsampled shortcut to the body output, so this means the body contributes nothing |
| L1 of the trivial baselines | seconds | On the validation set: official 0.0591, Lanczos 0.0608, ours 0.06326 — identical to nearest, to five decimals |
| Overfitting four real crops | one minute | Every variant plateaus at exactly the nearest L1 (0.01438) for the first ~1000 iterations, then escapes: by 2000 iterations L1 is 0.00727, below bicubic (0.01246), and the body's contribution grows from 0.0003 to 0.0117 |

So the network is not broken and the recipe is not wrong: it starts in a phase where the body is suppressed and
only the shortcut answers, and it needs on the order of a thousand iterations to leave that phase even on four
fixed images. Both long runs stayed inside it, which is why their outputs equal nearest upsampling and their
pixel loss never moved.

A failed probe worth recording: the first overfit test used `torch.rand` as ground truth. Every variant converged
to 0.277, which is simply the best achievable L1 against uniform noise, and proved nothing.

## Why not simply use a pretrained model

The question comes up every time this document is read, so the answer is recorded here. A survey of the public
super-resolution ecosystem found no release that is cleanly usable in a commercial product, and the reason is
structural rather than case by case: **no major super-resolution repository publishes a licence for its weight
files at all.** The BSD, Apache or MIT text covers the source code and never mentions models, weights or
checkpoints, so the only licence that can be traced is the one on the training data, and that is almost always
academic-only. DIV2K, LSDIR, ImageNet and WED all carry explicit non-commercial clauses; Flickr2K and OST
published no terms whatsoever, which for clearance is worse than a known restriction.

That rules out the entire DF2K-trained family -- SwinIR, HAT, DAT, BSRGAN, EDSR, OmniSR, SAFMN, SPAN, DRCT,
Swin2SR -- along with the official Real-ESRGAN weights this fork already removed. SRFormer is blocked one
level earlier, at the code, under CC BY-NC 4.0.

Community model zoos do not solve it. Many permissively tagged entries are fine-tunes of `RealESRGAN_x4plus` or
of ImageNet-pretrained checkpoints, relabelled by whoever uploaded them. The rule worth carrying forward: **a
licence tag on a model card is an assertion by the uploader, not a fact about the data.**

Whether a dataset's restriction propagates to the weights trained on it is genuinely unsettled -- Keras states
that checkpoints do not inherit the dataset licence, and PyTorch's legal guidance declines to give an answer --
so this is a question of risk appetite, not a settled rule. Training on CC0 data makes it moot, which is why
that is the direction here.

Provenance of these claims: PD12M's CDLA-Permissive-2.0 licence and megalith-cc0's per-row CC0 deed were
verified directly against the dataset APIs and cards. DIV2K's "made available for academic research purpose
only" was read from a mirror quoting the original page, because the ETH page now returns 404. The rest is
research that has not been verified here, and none of it is legal advice.

## Classical upscaling: measured and rejected

Before spending more GPU time, the weight-free options were measured on both validation sets with
`tools/eval_classical.py`. They need no training, no weights and no dataset licence, so any of them that beat
Lanczos could ship immediately.

| | light degradation | | | training degradation | | |
|---|---|---|---|---|---|---|
| | PSNR | SSIM | NIQE | PSNR | SSIM | NIQE |
| nearest | 24.03 | 0.7195 | 9.988 | 20.57 | 0.4733 | 20.085 |
| bicubic | **25.58** | **0.7606** | 7.197 | 20.96 | **0.5229** | **10.212** |
| Lanczos (current) | 25.36 | 0.7488 | **7.073** | 20.96 | 0.5205 | 10.652 |
| Lanczos + unsharp mask, sigma 1.5 amount 0.8 | 24.50 | 0.7294 | 7.031 | 20.73 | 0.4998 | 10.205 |
| Lanczos + `detailEnhance` | 19.16 | 0.5864 | 7.019 | 17.54 | 0.3685 | 10.840 |
| Lanczos + guided filter | 24.46 | 0.7444 | 7.009 | 20.54 | 0.5171 | 10.827 |

No sharpening variant helps: an unsharp mask amplifies the noise and the JPEG artefacts along with the edges, so
it loses PSNR and SSIM without buying NIQE. `detailEnhance` destroys the image.

Bicubic scores above Lanczos on PSNR and SSIM in both sets, but inference keeps Lanczos: the light set was built
with a bicubic downscale, so bicubic upscaling is its matched inverse and the comparison is partly
self-fulfilling, and bicubic is also among the resize modes the second-order degradation draws from. Real
photographs are not downscaled bicubically, and a confounded 0.2 dB is not a reason to change what ships.

## The cause: an uninitialised 34-layer body

`SRVGGNetCompact` never calls `default_init_weights`, so its convolutions keep the PyTorch default for
`nn.Conv2d`, `kaiming_uniform_(a=sqrt(5))`. That gives a standard deviation of `sqrt(1 / (3 * fan_in))` where a
ReLU-family activation needs `sqrt(2 / fan_in)`: a factor of 2.45 per layer, harmless in a shallow network. With
`num_conv: 32` the body is 34 convolutions deep and has no residual connection inside it -- the only shortcut is
the nearest upsampling added to the final output -- so the factor compounds to 2.45^34, about 1e13, which is the
measured gradient ratio of 3e12.

The official `realesr-general-x4v3` weights corroborate it from the other side: their body convolutions have a
standard deviation of 0.053 to 0.085 against the 0.059 that Kaiming predicts, not the 0.024 of the PyTorch
default. A run starting with a 2.5e-15 gradient could not have reached that scale, so upstream initialises.

`SRVGGNetCompactClean` in `gfpgan/archs/srvgg_clean_arch.py` subclasses the architecture and initialises in
`__init__`. The initialisation must live there rather than in `RealESRGANCleanModel`: BasicSR loads
`pretrain_network_g` in `SRModel.__init__` and calls `init_training_settings` afterwards, so initialising in the
model would silently overwrite the phase 1 weights when the adversarial phase starts from them.
`tests/test_realesrgan_config.py` asserts the first body layer still receives gradient.

**What this fixes, and what it does not.** On the real training pipeline with Kaiming initialisation, the body's
contribution grows from 0.00746 at 500 iterations to 0.02673 at 1000, against about 0.0003 in the dead phase: the
body is learning instead of passing the shortcut through. `l_g_pix` cannot settle the question, because each batch
draws a different degradation, so the run below scores on the fixed validation set instead.

## Run 03: the initialisation fix, measured

3000 iterations, pixel loss only, no adversarial loss, on the same 14,936 photographs. Stable: 2901 clean
iterations, no non-finite value, no generator loss spike. Scored on the validation set built with the training
degradation:

| | PSNR | SSIM | NIQE |
|---|---|---|---|
| Lanczos (what inference uses) | 20.96 | 0.5205 | 10.652 |
| **ours, 3000 iterations** | **21.12** | **0.5225** | 12.600 |
| official realesr-general-x4v3 | 20.77 | **0.5799** | **7.971** |
| runs 01 and 02, default initialisation | 20.57 | 0.4733 | 20.09 |

The earlier runs sat at 20.57 and 0.4733, which are nearest upsampling to the decimal, because only the shortcut
trained. This run leaves that plateau and passes Lanczos on both fidelity metrics, in 3000 iterations where 8000
previously moved nothing. **The training pipeline reconstructs correctly once the network receives gradient: what
was missing was the initialisation, not the recipe, the losses or the degradation.**

The margin over Lanczos is small, 0.16 dB and 0.002 SSIM, and the official model remains clearly better overall:
0.5799 SSIM against 0.5225, and NIQE 7.971 against 12.600. The weak NIQE is expected and does not contradict the
result, since a model trained on L1 alone is smooth by construction; it still improved from 20.09 to 12.600. What
remains is perceptual quality, which is what the adversarial phase and more training data are for.

This also answers the question of whether the pipeline needed validating against DIV2K, the dataset the official
weights use: it does not. The pipeline now beats both Lanczos and the official model on PSNR while training only
on CC0 photographs, so no academically licensed data has to enter the project to prove the code works.

## Run 04: the adversarial phase, from weights that had learned

4000 iterations with the adversarial and feature matching losses, starting from the run 03 generator and a fresh
discriminator, as upstream does. Stable: 3901 clean iterations, no non-finite value, no spike. Same validation
set as every run above:

| | PSNR | SSIM | NIQE |
|---|---|---|---|
| Lanczos (what inference uses) | 20.96 | 0.5205 | 10.652 |
| run 03, pixel only | 21.12 | 0.5225 | 12.600 |
| **run 04, adversarial** | **21.25** | 0.5395 | **7.411** |
| official realesr-general-x4v3 | 20.77 | **0.5799** | 7.971 |

**This is the first model here that beats Lanczos on all three metrics**, and it beats the official compact model
on two of the three. NIQE, the metric that was catastrophic in every earlier attempt -- 12.600 at best, 42.41 at
worst -- is now 7.411 against the official 7.971.

The two-phase recipe was never wrong. Run 02 applied exactly this recipe and failed, because its phase 1 was the
dead-body phase and the adversarial phase started from a generator that only passed nearest upsampling through.
Starting it from weights that reconstruct changes the outcome completely, and it shows in the discriminator too:
its scores stayed near zero through all 4000 iterations, where run 01 drove them to plus or minus 10.

Two honest limits. SSIM remains clearly behind the official model, 0.5395 against 0.5799. And NIQE is a
no-reference metric that rewards plausible texture, so beating the official model by 0.56 on it is not proof of
better perceptual quality; only a visual comparison settles that.

### The visual check overturns the table: Lanczos stays

The grid built with `tools/make_sr_grid.py` on the same validation pairs says the opposite of the metrics. Our
output is Lanczos with a little more edge contrast: it keeps the noise, the mottling and the blocking of the
degraded input. The official model removes all of it, leaving clean surfaces and defined edges. The gap between
the two columns is large and obvious, and it favours the official model, which our NIQE of 7.411 against its
7.971 had ranked the other way.

So the no-reference metric did not merely fail to prove perceptual quality, it inverted the ranking. NIQE is
kept in the table as one signal among several, and is no longer treated as the metric to optimise.

**`inference_gfpgan.py` keeps Lanczos.** The measured margin over it is real but cosmetic, and adding a network
to the inference path is not worth that. The bar is not "beats Lanczos on a table", it is "visibly restores what
Lanczos cannot", and run 04 does not clear it.

The likely cause points at the data rather than the recipe: the ground truth used here has a median short side of
683 px and 0.47 MP, so the targets themselves carry little fine texture for a model to learn to synthesise. That
is what the next run tests, with tiles cut from roughly 10 MP photographs.

## Run 05: higher resolution ground truth changes nothing, and refutes the explanation above

58,972 tiles of 512 px cut from 8,674 megalith-cc0 photographs of about 10 MP, against the 14,936 photographs of
0.47 MP used before. Identical protocol to run 03: 3000 iterations, pixel loss only, same config, same validation
set. Stable, 2901 clean iterations.

| | PSNR | SSIM | NIQE |
|---|---|---|---|
| Lanczos | 20.96 | 0.5205 | 10.652 |
| run 03, 0.47 MP corpus | 21.12 | 0.5225 | 12.600 |
| run 05, 10 MP corpus | 21.10 | 0.5220 | 14.175 |
| official realesr-general-x4v3 | 20.77 | **0.5799** | **7.971** |

**A twenty-one fold increase in pixels per image produced no improvement.** PSNR and SSIM are unchanged inside the
noise and NIQE is worse. The explanation recorded above, that the shortfall came from ground truth too small to
carry fine texture, is wrong, and it was written as a prediction that this run would improve on run 03.

### Reading the loss curves weakens that conclusion

Two things came out of the training logs of runs 03 and 05, and both qualify the table above.

| | median l_g_pix, iter 100-1000 | 1000-2000 | 2000-3000 | slope over the last third |
|---|---|---|---|---|
| run 03, 0.47 MP corpus | 0.08403 | 0.07177 | 0.07722 | -0.00685 per 1000 iterations |
| run 05, 10 MP corpus | 0.06205 | 0.05680 | 0.05428 | -0.00984 per 1000 iterations |

**Neither run had converged.** Both losses were still falling steadily at iteration 3000, so 3000 iterations is
undertraining rather than a ceiling in the recipe. Upstream trains this architecture for hundreds of thousands of
iterations.

**Run 05 fits about 30 per cent better and still scores no better.** A model reaching 0.0543 where the other
reaches 0.0772, yet losing on NIQE, points at the benchmark rather than the model: the validation set was built
from the 0.47 MP corpus, so it is matched to run 03's training distribution and mismatched to run 05's. It was
described above as a neutral held-out benchmark, and for run 05 it is not neutral.

So the honest statement is weaker than "refuted": the resolution hypothesis was **not confirmed**, in a test whose
benchmark favoured the older corpus and in which both competitors were undertrained. A fair test needs a
validation set drawn from both corpora and a run long enough to converge.

The next run goes to DIV2K, the corpus the official weights were trained on, under the same 3000-iteration
protocol -- so it inherits the same undertraining, and can only show whether the corpus changes the picture at
equal budget, not what the pipeline reaches at convergence.

## Run 06: the corpus the official weights were trained on changes nothing

5,724 tiles of 512 px cut from the 800 DIV2K training images, the corpus behind the official Real-ESRGAN weights.

**DIV2K is licensed for academic research only**, as its own terms state. It is
used here as a diagnostic to test whether the corpus explains the shortfall, and nothing trained on it is kept or
shipped -- the checkpoints exist only to produce the table below. The licensed corpus remains megalith-cc0.
Same protocol as runs 03 and 05: 3000 iterations, pixel loss only, same config, same validation set. Stable, 2901
clean iterations, no non-finite values and no gradient spikes.

| | corpus | PSNR | SSIM | NIQE |
|---|---|---|---|---|
| Lanczos | -- | 20.96 | 0.5205 | 10.652 |
| run 03 | megalith-cc0, 0.47 MP | **21.12** | 0.5225 | 12.600 |
| run 05 | megalith-cc0, 10 MP | 21.10 | 0.5220 | 14.175 |
| run 06 | DIV2K | 21.07 | 0.5177 | 13.354 |
| official realesr-general-x4v3 | DIV2K and others | 20.77 | **0.5799** | **7.971** |

**The three corpora land within 0.05 dB of PSNR and 0.005 of SSIM of each other**, a spread smaller than the
difference between runs. Training on the exact corpus behind the official weights buys nothing at this budget.

This was run to answer a specific doubt: whether the licensed corpus was itself the reason the results were
weak. It is not. The clean megalith-cc0 data matches the standard academic corpus under an identical protocol,
which is the result needed to keep using it. The remaining gap to the official model -- 0.06 of SSIM and 5 points
of NIQE -- is therefore in the recipe or the training budget, not in the source of the images.

## Undertrained, or a ceiling? Scoring the checkpoints already on disk

Both candidate causes were still open, and the usual way to separate them is an expensive long run. It is not
necessary: runs 03 and 05 each saved a checkpoint every 1000 iterations, so the shape of the curve can be read
from weights already written to disk, in minutes of inference instead of hours of training.

| checkpoint | PSNR | SSIM | NIQE |
|---|---|---|---|
| run 03 @1000 | 20.48 | 0.4430 | 13.272 |
| run 03 @2000 | 20.77 | 0.4879 | 18.357 |
| run 03 @3000 | **21.12** | **0.5225** | 12.600 |
| run 05 @1000 | 20.49 | 0.4456 | 13.580 |
| run 05 @2000 | 20.78 | 0.4903 | 16.964 |
| run 05 @3000 | 21.10 | 0.5220 | 14.175 |
| official realesr-general-x4v3 | 20.77 | 0.5799 | 7.971 |

**Undertrained.** SSIM gains +0.0449 then +0.0346 in run 03, and +0.0447 then +0.0317 in run 05: large, barely
decaying, with no flattening at the last point. PSNR does not decay at all, gaining +0.29 then +0.35 -- consistent
with a body that only started learning once it was initialised. The two corpora track each other to within 0.003
of SSIM at every checkpoint, independently confirming run 06's finding that the corpus is not the variable.

The learning rate makes this reading stronger rather than weaker. The `MultiStepLR` milestone at iteration 2400
halves the rate to 5e-5, so the final +0.0346 was earned with the last 600 iterations running at half speed. The
curve is still climbing steeply under a rate that had already been cut.

This measurement cost one pass over a 64-pair validation set per checkpoint, and it replaced a 20,000-iteration
run that had been queued to answer the same question.

## Run 07: extending run 03, with the prediction written first

Run 03 resumed from its own `3000.state` for another 3000 iterations. Two details of `basicsr` 1.4.2 shape it,
both verified in the installed source rather than assumed:

- `check_resume` (`utils/misc.py:116`) repoints `pretrain_network_g` and `pretrain_network_d` at the iteration's
  own checkpoints, and rewrites `param_key_g` from `params_ema` to `params`. The config leaves those paths null.
- `train.py:102` skips `make_exp_dirs` when resuming, so the experiment name has to stay `stab_sr05_init`.
- `MultiStepLR.load_state_dict` restores `milestones` from the state file, so editing them in the YAML has no
  effect. The only milestone has already fired, so the extension runs at a constant 5e-5. That is the useful
  case: with no decay in the middle, a flattening is a real ceiling and not an artefact of the scheduler.

Extrapolating the decaying SSIM gains gives the prediction, recorded before the run so it can fail:

| iteration | predicted SSIM |
|---|---|
| 4000 | 0.545 |
| 5000 | 0.562 |
| 6000 | 0.575 |

Stop if SSIM at 4000 is below 0.535, which would mean the curve had already flattened. Treat a longer run as paid
for if SSIM at 6000 reaches 0.570, closing on the official 0.5799.

### The prediction, scored

| iteration | predicted SSIM | measured SSIM | PSNR | NIQE |
|---|---|---|---|---|
| 3000 | -- | 0.5225 | 21.12 | 12.600 |
| 4000 | 0.545 | **0.5465** | 21.30 | 8.743 |
| 5000 | 0.562 | 0.5564 | 21.33 | 8.280 |
| 6000 | 0.575 | **0.5606** | **21.35** | **8.264** |
| official realesr-general-x4v3 | -- | 0.5799 | 20.77 | 7.971 |

The stop criterion did not fire, and **the success criterion was not met either**. The prediction was exact at
4000 and optimistic after it: gains decay with a ratio near 0.42, not the 0.77 assumed. Extrapolating on the
measured ratio puts the asymptote at roughly 0.564, so **further iterations at 5e-5 do not pay**. The limit is the
learning rate and the recipe, not the budget.

Two results were not predicted and matter more than the SSIM:

**NIQE collapsed from 12.600 to 8.264**, next to the official 7.971, with no adversarial loss at all -- this phase
is pixel loss only. Against Lanczos at 10.652 it is a clear gain.

**The internal validation PSNR misled.** It fell across the same span (24.5437 at 4000, 24.4931 at 6000) while the
val2 PSNR rose. Two consecutive drops looked like a plateau and were not one.

### The visual check, and a real video frame

The grid at 6000 agrees with the table for the first time. The decisive panel is the white caravan: run 03 at 3000
keeps the same speckle as Lanczos, and 6000 removes it. That was exactly the failure that had kept Lanczos in
`inference_gfpgan.py` -- the official model removed noise and ours did not. Now ours does, though the official
remains visibly cleaner.

Applied to a real 259x195 video frame, whose compression is nothing like the synthetic degradation either model
trained on, **the ranking reverses**. The official model amplifies the compression into hard rectangular blocks,
worst in the sky and along building edges. Ours introduces no blocking but comes out soft, the familiar blur of an
L1-only objective. Neither wins: they fail in opposite directions.

One caveat keeps this from being a result about the official model. `realesr-general-x4v3` ships with a denoising
companion, `realesr-general-wdn-x4v3`, and the intended use blends the two. Only the first was applied here, so
some of that blocking may be the missing blend rather than the model. Comparing against a misconfigured baseline
proves nothing, and this is recorded as unresolved.

## Run 08: the adversarial phase, from a generator that is competent

Run 04 ran this same phase from the 3000-iteration generator and its table looked like a win that the visual grid
then reversed. The diagnosis was that phase 1 was too weak to hand over. This repeats it from the 6000-iteration
generator, fresh discriminator, identical loss weights -- so the only variable changed is the quality of what
phase 1 handed over. 4000 iterations, stable, 3901 clean iterations.

| | PSNR | SSIM | NIQE |
|---|---|---|---|
| Lanczos | 20.96 | 0.5205 | 10.652 |
| phase 1 @6000 | 21.35 | **0.5606** | 8.264 |
| phase 2 @1000 | 21.41 | 0.5592 | 8.754 |
| phase 2 @2000 | **21.42** | 0.5589 | 8.661 |
| phase 2 @3000 | 21.41 | 0.5579 | 8.157 |
| phase 2 @4000 | 21.36 | 0.5554 | **7.884** |
| official realesr-general-x4v3 | 20.77 | 0.5799 | 7.971 |

The adversarial loss trades SSIM for NIQE, monotonically and exactly as it is supposed to: 0.5606 down to 0.5554,
8.264 down to 7.884, passing the official model's NIQE on the way.

**This is the same shape of table that run 04 produced, and this time the visual check does not overturn it.** On
the validation grid phase 2 is close to indistinguishable from phase 1. On the real video frame it is a modest but
real improvement in the right direction: where phase 1 smears the post and softens the building, phase 2 holds the
edges, and it does so without reintroducing the blocking the official model shows on the same frame.

So the honest summary is a small gain, not a leap. What matters is that it is the first clean model that beats
Lanczos on every metric *and* on inspection, on both the synthetic validation set and a real frame. That makes it
the first credible replacement for the Lanczos background in `inference_gfpgan.py`.

Still open before that switch: this is a single real frame, the official baseline is missing its `wdn` companion,
and the validation set's degradation is the one the model trained on.

## The combined pipeline, and a clean photograph that reverses the recommendation

Every comparison up to here measured the two halves apart: the face model on aligned crops, the background model
on validation tiles. `tools/pipeline_combined.py` runs them together the way a user would, with a single face
model across every arm, so the only variable between arms is the background upsampler. `GFPGANer` accepts any
object exposing `.enhance(img, outscale)`, and falls back to a Lanczos upscale when given none, which makes the
current default a free baseline.

On the 259x195 video frame the choice of background barely showed. That is not evidence that it does not matter:
the face fills most of that frame and is identical across arms, so it dominates what the eye reaches for. Measured
globally, swapping Lanczos for the trained model moves the combined output about as much as it moves a standalone
background (mean absolute difference 2.63 against 2.59, 5.6 per cent of pixels past 8 against 4.6). The arm works;
the frame was the wrong test.

Two attempts to isolate the face region and measure the background alone both failed, and neither number should be
reused. Masking by thresholding the pipeline's Lanczos arm against a separately computed Lanczos upscale compared
two different resamplings and marked 69 per cent of the frame as face. Masking by exact equality across arms gave
0.3 per cent, because `inv_soft_mask` is Gaussian and saturates at 1.0 only in a small core, so almost no pixel is
byte-identical. The face and the background are blended continuously and cannot be separated from the outputs.

The right test is a photograph where the background *is* the frame: 1280x960, clean, upscaled 4x, inspected at
three textures.

| region | Lanczos | ours, phase 2 @4000 | official |
|---|---|---|---|
| rock face | soft | **striations washed out, softer than Lanczos** | crisp striations and ridge |
| sand with footprints | soft, structure intact | **grain erased, ridges gone** | grain and sharp footprint edges |
| waterline | soft, foam structured | **foam reduced to a milky blur** | foam structured |

**Consistently official > Lanczos > ours, and ours is worse than doing nothing clever.** On a clean input our model
reads genuine fine texture as noise and removes it. That follows directly from its training distribution: every
sample it ever saw was heavily degraded by blur, noise and JPEG, so it never learned that some inputs should be
left alone. What was trained is a denoiser, not an upscaler.

**This reverses the recommendation recorded above.** Phase 2 @4000 is not a general replacement for the Lanczos
background. It wins only on inputs degraded roughly the way it trained, and it loses on clean ones, which is the
worse failure for a general tool -- degradation is optional, but users supplying good input is not.

The fix is in the data, not the budget or the architecture: the degradation pipeline needs samples at or near
identity, so the model learns to pass a good input through. Until that is tested, `inference_gfpgan.py` keeps
Lanczos.

## Run 09: measuring the defect, then fixing it

"It erases texture on clean input" was a visual judgement, and nothing can be shown to improve it while it stays
one. `val_mild` is the same 64 pairs built by the same machinery as `val2` and from the same corpus, with every
degradation knob neutralised: isotropic blur at sigma 0.1, no noise, no second blur, no sinc, JPEG at 94. Lanczos
scores 24.06 dB on it against 20.96 on `val2`, which confirms it is genuinely the milder set.

| | val2 PSNR | val2 SSIM | val_mild PSNR | val_mild SSIM |
|---|---|---|---|---|
| Lanczos | 20.96 | 0.5205 | **24.06** | **0.6897** |
| phase 1 @6000 | 21.35 | 0.5606 | 23.00 | 0.6475 |
| official | 20.77 | 0.5799 | 22.27 | 0.6977 |

**The sign flips.** The model is 0.39 dB and 0.040 of SSIM above Lanczos where it trained, and 1.06 dB and 0.042
below it where the input is already good. **The official weights invert the same way**, and by more in PSNR, so
this is a property of the Real-ESRGAN degradation recipe rather than of this training run. They do keep the better
SSIM, degrading in structure more gracefully than ours.

### The fix, with a control arm

`mild_prob` draws a batch whose degradation is near the identity; it defaults to 0 and reproduces the inherited
pipeline exactly. Two arms ran from the phase-1 generator at 6000 for 3000 iterations each, identical but for
`mild_prob`, because the learning rate restarts here and the extra iterations would move the numbers on their own.

| | val2 PSNR | val2 SSIM | val_mild PSNR | val_mild SSIM |
|---|---|---|---|---|
| control, `mild_prob` 0 | 21.38 | 0.5667 | 23.03 | 0.6479 |
| **mild, `mild_prob` 0.25** | **21.41** | 0.5651 | **23.77** | **0.6821** |

**The control did not move**: 23.00 to 23.03 on `val_mild`. All of the gain is attributable to `mild_prob` --
0.74 dB and 0.034 of SSIM over the control, closing 72 and 82 per cent of the distance to Lanczos, at no cost on
`val2`.

On the clean photograph from the previous section, downscaled by 4 and reconstructed against the original, the
effect is larger than on the validation set:

| | PSNR | SSIM |
|---|---|---|
| Lanczos | **28.91** | **0.8156** |
| control | 27.52 | 0.7645 |
| **mild** | **28.68** | 0.8002 |
| official | 27.26 | 0.8048 |

The defect reproduces on the real photograph -- the control sits 1.39 dB below Lanczos -- and `mild` closes it to
0.23 dB, a gain of 1.16 dB over the control, and passes the official model in PSNR.

### What this does not establish

Lanczos still wins on clean input. The model stopped being much worse; it did not become better. The two arms are
also visually indistinguishable on the sand and rock crops at 3x magnification: 1.16 dB on a 1280x960 photograph
is real but below what inspection separates, which is the reverse of run 04, where the eye overturned the table.
NIQE moved the wrong way on `val_mild`, 8.88 to 10.83, though it has misjudged twice already here.

Not yet tested: whether a larger `mild_prob` closes the rest, whether the gain survives the adversarial phase, and
whether training from scratch with mixed degradation beats patching a model that learned the bias first.

## Run 10: how much near-identity data, measured

Two more arms under the same protocol, 0.5 and 0.75, turning two points into a dose-response curve. Every arm is
scored on both validation sets and on the clean photograph, reconstructed from a 4x downscale of itself.

| `mild_prob` | val2 PSNR | val2 SSIM | val_mild PSNR | val_mild SSIM | photo PSNR | photo SSIM |
|---|---|---|---|---|---|---|
| 0 | 21.38 | **0.5667** | 23.03 | 0.6479 | 27.52 | 0.7645 |
| 0.25 | **21.41** | 0.5651 | 23.77 | 0.6821 | 28.68 | 0.8002 |
| **0.50** | 21.40 | 0.5608 | 24.06 | 0.6960 | 29.30 | 0.8209 |
| 0.75 | 21.34 | 0.5509 | **24.26** | **0.7043** | **29.56** | **0.8299** |
| Lanczos | 20.96 | 0.5205 | 24.06 | 0.6897 | 28.91 | 0.8156 |
| official | 20.77 | 0.5799 | 22.27 | 0.6977 | 27.26 | 0.8048 |

**The gap closes completely at 0.5 and is passed at 0.75.** The previous section concluded that the model had
"stopped being much worse rather than becoming better"; that was a dose too small, not a ceiling. At 0.5 it
matches Lanczos on `val_mild` PSNR exactly and exceeds its SSIM, and on the clean photograph it wins on both.

**0.5 is the operating point, by marginal return.** From 0 to 0.5 buys 0.048 of `val_mild` SSIM for 0.006 of
`val2` SSIM, about eight to one. From 0.5 to 0.75 buys 0.008 for 0.010, roughly one to one, and `val2` PSNR turns
down as well. The trade-off is real and it becomes unfavourable somewhere just past 0.5.

At `mild_prob` 0.5 the model ties or beats Lanczos on all three sets, which removes the measured reason to keep
Lanczos as the background.

### Magnification exaggerated a difference that 1:1 does not show

Inspected at 3x nearest-neighbour magnification, 0.5 and 0.75 look clearly smoother than Lanczos -- sand grain
softened, rock striations washed out -- which contradicts every metric above. At 1:1, the scale at which a 4x
upscale is actually viewed, that contradiction mostly disappears: the three are close on sand, and on rock Lanczos
is the grainier one while 0.5 and 0.75 are cleaner with better formed shadows.

So the disagreement belonged to the inspection method. Heavy magnification rewards acutance, and some of the grain
Lanczos preserves is upsampling artefact and original JPEG noise rather than recovered detail, which is exactly
what a reference metric against the true original penalises. This is the opposite of run 04, where the eye
correctly overturned a no-reference metric; here the metrics are PSNR and SSIM against the real photograph, and
the magnified view was the misleading one.

The official model still looks the sharpest of the four at both scales while scoring the worst PSNR on all three
sets. It also remains unusable here on licensing grounds, so it stays a reference point rather than an option.
NIQE again moved the wrong way, 8.55 at `mild_prob` 0 against 10.76 at 0.5, having now misjudged three times.

Still open: whether the gain survives the adversarial phase, whether training from scratch with mixed degradation
beats patching a model that learned the bias first, and whether `inference_gfpgan.py` should adopt `mild_prob` 0.5
as its background in place of Lanczos.

## Run 11: the adversarial phase, with the mixture kept on

The gain so far belongs to a pixel-loss model, and the adversarial phase is both what the recipe needs for texture
and what could plausibly undo it: a discriminator rewards convincing detail, which is the pressure that made the
model invent texture over clean input in the first place. Phase 2 therefore ran from the `mild_prob` 0.5
generator, fresh discriminator, 4000 iterations, stable, **with `mild_prob` left at 0.5**. Turning it off would
have let the adversarial phase retrain the bias, and the question is whether the two are compatible.

| | val2 PSNR | val2 SSIM | val2 NIQE | val_mild PSNR | val_mild SSIM | val_mild NIQE | photo PSNR | photo SSIM |
|---|---|---|---|---|---|---|---|---|
| Lanczos | 20.96 | 0.5205 | 10.652 | 24.06 | 0.6897 | 9.068 | 28.91 | 0.8156 |
| `mild_prob` 0.5, pixel | 21.40 | **0.5608** | 8.066 | 24.06 | **0.6960** | 10.760 | 29.30 | **0.8209** |
| **`mild_prob` 0.5, phase 2** | **21.42** | 0.5499 | **7.998** | **24.39** | 0.6936 | **7.704** | **29.37** | 0.8183 |
| official | 20.77 | 0.5799 | 7.971 | 22.27 | 0.6977 | 9.056 | 27.26 | 0.8048 |

**It survives, and improves.** The adversarial model holds the best PSNR on all three sets, keeps SSIM above
Lanczos on both sets where clean input matters, and its `val_mild` NIQE collapses from 10.760 to **7.704**, the
best of the four and ahead of the official model. Nothing fell back toward the 23.0 and 27.5 of the unmixed model,
which was the stated failure condition.

**NIQE agreed for the first time.** It had misjudged three times here -- rewarding invented texture in run 04, and
moving the wrong way in runs 09 and 10. This time it moves with the reference metrics and with inspection. The
lesson is not that NIQE is now trustworthy; it is that a no-reference metric only carries weight when reference
metrics and the eye already point the same way.

The cost is on `val2` SSIM, 0.5608 down to 0.5499, the usual trade of structure for texture on the heavily
degraded set, and the only number where the official weights remain clearly ahead at 0.5799.

Inspected at 1:1, the eye does not contradict the table this time: on sand the adversarial model is slightly
crisper than the pixel one and close to Lanczos, and on rock both of ours are cleaner than Lanczos, which is the
grainier of the three. The official model still looks the sharpest of the four while holding the worst PSNR on
every set -- it reads as sharp because it invents detail, which is exactly what a reference metric against the
true photograph penalises. It remains a reference point rather than an option, on licensing grounds.

This is the strongest model this fork has produced: it beats Lanczos on every metric on the near-identity set
and on the clean photograph, beats it comfortably on the degraded set, and survives inspection at native scale.

Still open: whether training from scratch with mixed degradation beats patching a model that learned the bias
first, and whether `inference_gfpgan.py` should adopt this model as its background in place of Lanczos.

## Run 12: the long run, and the adversarial phase turns harmful

Every run to this point was 3000 to 6000 iterations, which run 07's checkpoint trend had already shown to be
undertraining. This pulls that lever: from scratch, `mild_prob` 0.5, 30000 iterations of pixel loss with
milestones at 18000 and 24000, then 10000 adversarial iterations with a fresh discriminator and the mixture kept
on. Both phases stable, no non-finite losses and no gradient spikes. It also settles the from-scratch question by
construction, since this model never learns the heavy-degradation bias that the earlier ones had corrected
afterwards.

| | val2 PSNR | val2 SSIM | val_mild PSNR | val_mild SSIM | photo PSNR | photo SSIM |
|---|---|---|---|---|---|---|
| Lanczos | 20.96 | 0.5205 | 24.06 | 0.6897 | 28.91 | 0.8156 |
| patched, phase 2 (run 11) | 21.42 | 0.5499 | 24.39 | 0.6936 | 29.37 | 0.8183 |
| **long, phase 1 only** | **21.46** | **0.5764** | **24.57** | **0.7198** | **29.86** | **0.8374** |
| long, phase 2 | 21.07 | 0.5222 | 24.20 | 0.6606 | 28.44 | 0.8008 |
| official | 20.77 | 0.5799 | 22.27 | 0.6977 | 27.26 | 0.8048 |

**Training length was worth more than everything else tried.** Phase 1 alone beats every previous model on every
set, passes the official weights' SSIM on `val_mild` (0.7198 against 0.6977) and lands within 0.0035 of them on
`val2`, the one number where the official model had stayed clearly ahead all along.

The per-iteration trend on `val_mild` shows where it came from, and that it had not finished:

| iteration | PSNR | SSIM |
|---|---|---|
| 6000 | 24.13 | 0.6989 |
| 12000 | 24.43 | 0.7126 |
| 18000 | 24.52 | 0.7172 |
| 24000 | 24.54 | 0.7189 |
| 30000 | **24.57** | **0.7198** |

### The adversarial phase destroyed it, monotonically

The second phase made every fidelity metric worse from its first checkpoint onward, with no beneficial short
window anywhere:

| phase 2 iteration | val2 SSIM | val_mild SSIM | photo SSIM | val2 NIQE |
|---|---|---|---|---|
| 0 (phase 1) | **0.5764** | **0.7198** | **0.8374** | 8.272 |
| 2000 | 0.5734 | 0.7170 | 0.8353 | 8.472 |
| 4000 | 0.5649 | 0.7108 | 0.8311 | 8.306 |
| 6000 | 0.5519 | 0.7003 | 0.8206 | 6.863 |
| 8000 | 0.5186 | 0.6867 | 0.8119 | 7.647 |
| 10000 | 0.5222 | 0.6606 | 0.8008 | **6.576** |

By 10000 iterations the model is *worse than Lanczos* on the clean photograph, having started well above it.

**This contradicts run 11, and the reconciliation is the useful part: the adversarial phase helps a weak
generator and harms a strong one.** Run 11 improved on a base trained for 3000 plus 3000 iterations. From a
30000-iteration base the discriminator has nothing left to add, so it only trades fidelity away.

NIQE improved monotonically in exactly the direction that destroyed everything else, its fourth misjudgement
here. It is not a usable acceptance criterion for this work.

### What the inspection and a second photograph say

At 1:1 the eye agrees with the table, which has not happened before in this document. Phase 1 keeps the sand
grain and footprint ridges that the patched model smooths away, and shows real striations on rock where the
patched model blurs. The official model still carries the most texture, but it is now a contest rather than a
rout, and it holds the worst PSNR on all three sets.

The claim does not generalise without a caveat. On the 256x192 video frame, reconstructed from a 64x48 downscale,
**Lanczos still wins**: 28.48 and 0.8312 against 27.69 and 0.8264. Visually the two are indistinguishable there
and both avoid the hard blocking the official model produces, so the gap corresponds to no visible defect, but it
is real and it is recorded. At that input size there is almost nothing left to reconstruct.

### A note on reading the wrong metric

The internal validation PSNR saturated early -- 26.04 at 14000, 26.18 at 26000 -- and neither learning-rate drop
produced a step. Taken at face value it said the run was finished by iteration 14000 and the remaining 16000
iterations were waste. The external trend says the opposite: `val_mild` SSIM kept climbing to the last checkpoint.
The internal metric is scored only on the heavy degradation pipeline while this model trains half its samples near
identity, so it measures the half that matters least here. Cutting the run short on it, which was considered
twice, would have discarded the best model this fork has produced.

`inference_gfpgan.py` still keeps Lanczos; adopting phase 1 as the background is a change to the product default
and remains a decision to be taken deliberately, not a consequence of this table.

## Run 13: 30000 was the ceiling

Run 12's phase 1 was still climbing at its last checkpoint, so it was resumed to 40000 at the constant 2.5e-5 both
milestones had already produced. The threshold was written before the run: worth it if `val_mild` SSIM cleared
0.7220, a ceiling at 0.7198 or below.

| iteration | val_mild SSIM | val2 SSIM | photo SSIM | val_mild PSNR |
|---|---|---|---|---|
| 30000 | 0.7198 | 0.5764 | 0.8374 | 24.57 |
| 32000 | 0.7199 | 0.5767 | 0.8373 | 24.56 |
| 34000 | 0.7200 | 0.5768 | 0.8376 | 24.56 |
| 36000 | 0.7202 | 0.5770 | 0.8374 | 24.58 |
| 38000 | 0.7204 | 0.5772 | 0.8375 | 24.58 |
| 40000 | 0.7205 | 0.5773 | 0.8379 | 24.56 |

**It did not clear the threshold: 0.7205 against 0.7220.** Ten thousand iterations, an hour and fifty-four
minutes of GPU, bought 0.0007 of SSIM, with PSNR flat across the whole range at 24.56 to 24.58. The other two sets
agree, at 0.0009 and 0.0005.

The reason for writing the number down first is visible in the table: **the curve is monotonically increasing at
every step.** Read without a threshold it looks like a model that is still learning, and the same shape could
have justified another ten thousand iterations, and another. The magnitude says it is noise. 30000 is the ceiling
at this learning rate, and `net_g_30000.pth` stays the recommended model -- not because 40000 is worse, but
because the difference does not exist.

This also closes the training-length question opened in run 07. The answer has two halves: 3000 iterations was
badly undertrained and 30000 was worth the cost, while beyond 30000 at a decayed learning rate there is nothing
left to collect.

## Run 14: the recommendation, tested on 64 photographs instead of one

Everything above rests on two real images: one clean photograph and one video frame. That is enough to expose a
failure and not enough to support a recommendation. `sr-data/val_gt` holds 64 full-resolution photographs,
numbered past the training corpus and verified absent from `meta_info_train.txt`, at 23 distinct sizes around
1024x680. Each was downscaled by 4 and reconstructed, then compared against itself -- the same protocol as the
single photograph, at n=64.

| | PSNR | SSIM | photographs beating Lanczos |
|---|---|---|---|
| Lanczos | 25.36 | 0.7488 | -- |
| **long, phase 1 @30000** | **26.20** | **0.7745** | **64 / 64** |
| patched, phase 2 (run 11) | 25.68 | 0.7491 | 55 / 64 |
| official | 23.62 | 0.7328 | 2 / 64 |

**The win count matters more than the mean.** 64 out of 64 is not an average concealing a split: the model is
better on every held-out photograph, so the single-photograph result was not a fluke and the recommendation no
longer rests on an anecdote. The standard deviation of 4.04 dB across photographs is content variation, not
disagreement about which model wins.

**The official weights lose to Lanczos on 62 of 64 clean photographs.** The beach photograph had already shown
this; at n=64 it generalises. Harming clean input is a property of the Real-ESRGAN degradation recipe rather than
of anything done here, which is what the `mild_prob` mixture exists to correct, and the correction is what puts
phase 1 ahead on every photograph.

The adversarial model ties Lanczos on SSIM at 0.7491 against 0.7488 while winning 55 of 64 on PSNR, which is one
more reason it is not the model to ship.

One limit of this test, by construction: the degradation is a bicubic downscale, so it measures the clean-input
regime specifically. It says nothing about heavily compressed sources, where the 256x192 video frame of run 12
remains the one case Lanczos still wins.

## Run 15: the adversarial phase was worth keeping after all, as a dial

Run 12 spent two hours on an adversarial phase that made every fidelity metric monotonically worse and the
checkpoint was written off. That was the wrong conclusion from the right data. Network interpolation
(arXiv:1811.10515, and ESRGAN arXiv:1809.00219) sets `theta(alpha) = (1 - alpha) * theta_pixel + alpha *
theta_gan` and picks alpha on validation: the two endpoints are fine-tuned from one another, so the blend moves
smoothly between fidelity and texture. It costs no training.

Measured over the same 64 pairs, reporting paired per-image deltas against the pixel model with a bootstrap 95
per cent confidence interval, which is what the earlier runs in this document should have done:

| alpha | val_mild PSNR | delta against alpha 0 | val2 PSNR | delta against alpha 0 |
|---|---|---|---|---|
| 0 (pixel) | 24.567 | -- | 21.464 | -- |
| 0.25 | 24.745 | +0.177 [+0.132, +0.221] | 21.540 | +0.075 [-0.037, +0.173] |
| **0.5** | **24.828** | **+0.260 [+0.185, +0.331]** | 21.374 | -0.090 [-0.276, +0.067] |
| 0.75 | 24.752 | +0.184 [+0.095, +0.271] | 21.243 | -0.222 [-0.443, -0.033] |
| 1 (adversarial) | 24.197 | -0.370 [-0.467, -0.271] | 21.074 | -0.390 [-0.635, -0.177] |

**Alpha 0.5 gains 0.26 dB on the near-identity set with a confidence interval excluding zero, and costs nothing
on the degraded set**, where its interval contains zero and the difference is therefore noise. Neither endpoint
is the best model; the blend is. `scripts/interpolate_sr_weights.py` bakes it into a single checkpoint that
`--bg_model` accepts.

It is a trade, not a free gain: SSIM falls monotonically with alpha, 0.7198 to 0.7080 at 0.5 on `val_mild`. And
NIQE was non-monotonic across the sweep, 9.903 at 0.25 against 9.528 at 0.5 and 6.927 at 0.75, so it would have
selected alpha 1, the one setting that is worse everywhere else. That is its fifth misjudgement here.

### What the literature already knew, and we did not check first

Three findings from a methodology review, recorded because they change how the runs above should be read:

**The adversarial phase degrading fidelity is a theorem, not a discovery.** Blau and Michaeli, *The
Perception-Distortion Tradeoff*, CVPR 2018 (arXiv:1711.06077) prove the tradeoff is monotonic and convex for any
distortion measure and any perceptual divergence. Run 12's two hours bought a published result. What theory does
not give is *where* on the curve a model lands, and run 12's real finding — that it fell below the do-nothing
baseline — is the part worth keeping. PIRM (arXiv:1809.07517) ranks perceptual quality only inside a distortion
band for exactly this reason, and would have framed it in advance.

**`mild_prob` is prior art, three times over.** CutBlur (arXiv:2004.00448) does it spatially and reports the same
experiment we ran, a restoration model fed clean input, improving from 22.61 dB to 27.33 dB. CResMD
(arXiv:1912.05293) samples degradation levels from a Beta(0.5, 1) law deliberately biased toward mild ones, and
states the requirement directly: "when the input image has no degradation, restoration algorithm is expected to
perform identity mapping". Gated degradation models (arXiv:2205.04910) switch individual operations off at
random. Our Bernoulli draw is the crudest of the three and should be cited, not presented as new.

**The gap we found is real even so.** Real-ESRGAN (arXiv:2107.10833) and BSRGAN (arXiv:2103.14006) designed the
degradation pipelines used here and neither acknowledges the clean-input regression; their limitation sections
list aliasing and out-of-distribution degradations only. The problem was worth measuring; only the remedy was
reinvented.
