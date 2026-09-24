# Changes against upstream GFPGAN

Executive summary of this fork against its base, commit `7552a77` of TencentARC/GFPGAN.
Per-component status and evidence are in `LICENSE_CLEANUP.md`; the training work is in `docs/training_stability.md`.

**Goal:** remove every component that carries commercial-use restrictions, and keep the method working.

**Size of the change:** 79 files changed, 6076 insertions and 3147 deletions — 27 files removed, 34 added
and 18 modified. Most of the insertions are documentation and tests; the code is smaller than the original.

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
screening rounds and eleven runs (`docs/training_stability.md`): generator lr 2.5e-5, global discriminator lr 2e-5,
component discriminators lr 2.5e-5, GAN weight 0.05, R1 every 4 iterations, generator gradient clipping 10, and
the component Gram style loss off. Upstream used lr 2e-3, which diverges to NaN within 4 iterations when there is
no pretrained prior to start from.

## 7. Inference — no automatic downloads

`inference_gfpgan.py` no longer downloads weights: `--model_path` is required. The Real-ESRGAN background
upsampler is gone and Lanczos is the default background. `--bg_model` accepts super-resolution weights the user
is licensed to use, reading the architecture and scale from the checkpoint itself (`gfpgan/bg_upsampler.py`);
nothing is downloaded either way. A model trained in this fork for that slot, and the evidence for it, are in
`docs/background_super_resolution.md`.

## 8. Example data and tests

- Removed `inputs/` (photographs of identifiable people with no documented licence) and the FFHQ-named test data.
- Added `tests/test_face_helper.py`, `tests/test_component_boxes.py`, `tests/test_discriminator_arch.py` and
  `tests/test_train_config.py`, all on synthetic images, then `tests/test_realesrgan_config.py`,
  `tests/test_realesrgan_mild.py`, `tests/test_ffhq_mild.py` and `tests/test_bg_upsampler.py` for the background
  work. The suite went from 2 to 130 tests across 20 files, at 79% statement coverage; `.github/workflows/tests.yml`
runs it, which nothing did before.

## 9. New documentation

`LICENSE_CLEANUP.md` (status of each blocker), `docs/training_stability.md` (screening rounds, runs and metrics,
including the comparison against the official GFPGAN v1.4) and `CLAUDE.md`. Later:
`docs/background_super_resolution.md` (the background model, what was measured and what was refuted),
`docs/face_restoration_alternatives.md` (CodeFormer, GPEN, VQFR and CFRNet assessed against this fork) and this
file.

## 10. Upstream documentation — audited for what a public fork may republish

A second review of the documentation, on publication safety alone, found six defects and closed all of them.
Four were already known: `LICENSE_CLEANUP.md` had listed upstream's branding and docs under "kept with
caveats", which deferred the question rather than answering it.

- `experiments/pretrained_models/README.md` was a bare download sheet for the FFHQ StyleGAN2 prior, the
  DFDNet-derived FFHQ landmarks and ArcFace — the three artefacts sections 2, 4 and 5 above removed the code
  for. It now states that nothing is downloaded, and why each is excluded.
- `Comparisons.md` displayed 28 photographs of named public figures, hotlinked from upstream, beside
  manipulated versions of their faces. Section 8 removed `inputs/` for the same reason; this kept the display
  after deleting the files. Replaced by a pointer to what is measured here on held-out faces.
- Upstream maintainers' personal e-mail addresses were this fork's published contact in `README.md`, the
  address for conduct reports in `CODE_OF_CONDUCT.md`, and the `author_email` in `setup.py`. All three now
  point at this repository.
- The download links for the release weights, for upstream's extra-model folders and for FFHQ were stripped
  out of upstream's README, which was kept below the notice with that edit disclosed. The coherence pass in
  section 11 then replaced that text altogether.
- `assets/gfpgan_logo.png` was upstream's mark serving as this fork's header, and `release.yml` hotlinked it
  from upstream's content host into this fork's release notes. Removed. `FAQ.md` (how to finetune release
  weights this repository cannot obtain) and `README_CN.md` (a stub inviting contributions to upstream's
  unfinished translation) went with it.

What the same review checked and found clean: no secrets, no local paths beyond a mount name, no face image
or weight tracked, and `docs/commons_face_corpus.md` describes the corpus entirely by counts, naming no
individual and no source filename.

## 11. Documentation — made to describe this fork rather than upstream's

A third review, on internal coherence, found that the inherited README described a different program and fixed
it at the root instead of line by line.

- `README.md` documented `-v`, `-bg_upsampler` and `-bg_tile`, none of which exist in `inference_gfpgan.py`,
  and omitted `--model_path`, which is required, along with `--arch`, `--channel_multiplier`, `-w/--weight` and
  `--bg_model`. Its installation section told the reader to clone upstream's repository and to
  `pip install facexlib` and `pip install realesrgan`, both dependencies this fork removed, with an unpinned
  `pip install basicsr` where 1.4.2 and `--no-build-isolation` are required. Its training section pointed at
  `options/train_gfpgan_v1.yml` and `train_gfpgan_v1_simple.yml`, which do not exist. Patching those in place
  would have produced text presented as upstream's while describing this fork, so the README now documents this
  fork, and upstream keeps a credit section with the paper, the citation and the licence and no instructions.
- "This branch" throughout the documentation became "this fork": the work is the repository's main line, not a
  side branch of it.
- `LICENSE_CLEANUP.md` cited `gfpgan-spec/specs/license-inventory.md` as its evidence base, a path that exists
  in no published repository, and its verification table still reported 20 passing tests against the present
  62.
- `docs/face_restoration_alternatives.md` measured this fork's data constraint as "9,927 aligned faces", which
  conflated the FFHQ stand-in used for evaluation with the 1,414 licensed faces that actually bound it.
- The "Licensed corpus" entry under "Still open" said training uses FFHQ as a stand-in without mentioning that
  the licensed corpus now exists and that run 13 was trained on it.
- `docs/training_stability.md` numbers its runs 01 to 10 and then 13. The gap is now stated where a reader
  meets it rather than left to look like a missing section.

## 12. Documentation — checked against the code, the configs and the data

A fourth review, on factual accuracy: every number and claim in the documentation re-derived from
`options/train_gfpgan_clean.yml`, the source, the git history of `7552a77`, the training logs and the corpora on
disk. Almost everything held. Four claims did not.

- Two summaries overstated the `-w` sweep. `CLAUDE.md` said weight 0.75 "dominates 1 in both regimes" and
  `README.md` said it "beat 1 on both metrics in both regimes". The sweep itself says otherwise, and says it
  correctly: on the degraded regime 0.75 is *indistinguishable* from 1 on the mean, -0.069 dB with p=0.29, while
  its median and SSIM are better. Both summaries now carry the measured numbers instead of the word.
- `docs/training_stability.md` gave the Beta arm as "+0.012 dB" against the two-point draw, while the table
  immediately above it puts Beta 0.012 dB *below*. The direction was inverted; the conclusion, a bounded tie,
  was not affected.
- `docs/commons_face_corpus.md` called 874 crops "labelled" where 872 carry a verdict, and gave a colour count
  that comes from each crop's first label while the corpus is built with the unanimous rule. Both now say which
  rule they use.
- The licence table's three families leave out 39 files, 0.09% of the candidates, under GFDL, GODL-India, OGL,
  "No restrictions" and bare Attribution. The text invited the reader to sum the rows to 42,971, which they do
  not. Stated.

What the same pass confirmed exactly, against the artefacts rather than against other prose: every learning
rate, loss weight and interval cited from the training config; upstream's 2e-3 learning rate, its R1 interval of
16, its component style weight of 200 and its `-w` default of 0.5 with the help text "Adjustable weights.", all
read from commit `7552a77`; the eight environment versions and the GPU; the corpus at every stage of its funnel
(876, 874, 709, 2,662, 1,757, 815, 1,524, 1,484, 1,478, and 1,414 training with 64 validation, with component
boxes for all 1,478); the crop yields of 117.9, 79.3 and 78.7 per thousand; the licence split of 34,071, 5,058
and 3,803; the download collapse to 0% after 24,000 images and the retry's 100% recovery; the four tie-breaking
rules over the review sheet, which reproduce 709, 711, 710 and 712 as tabulated; and run 13's 41 validations,
11.79 to 22.71 dB, gradient-norm median 3.92 against a maximum of 499, and 1h43 of GPU, which is 1:43:17 in the
log.

That pass had one blind spot, found afterwards by asking whether any of it had leaned on the sibling project.
Every number was re-derived from artefacts in this repository, so a claim whose evidence lives outside it was
classified as a legitimate external reference and never checked. One such claim was wrong: the throughput behind
the "training a StyleGAN2 prior" paragraph of `docs/commons_face_corpus.md`, 28.2 images per second at 512x512,
is above what this repository's own generator reaches with no gradients at all, and eight times a full training
step. It offered a reader a ten-day schedule where the measurement gives eighty-six days. Corrected there, along
with three paths into the sibling project's unpublished tree, which a public reader cannot follow.

Two families of claim resisted a first attempt and were then re-derived as well, after the crop-to-title join
turned out to be recoverable: crop filenames index their fetch record, and the merged directory keeps one prefix
per fetch.

- The series statistics reproduce **exactly** -- 1,388 series, 69 of them holding 1,079 crops, 40.5% -- once the
  trailing sequence number is stripped with punctuation allowed on either side, which is the rule the document
  describes. All three worked examples match to the crop: 93 photographs and 126 crops from the Iberoconf
  session, 44 and 69 from the lecture, 51 and 59 from the editathon opening.
- The cluster concentrations reproduce to within 0.9 points on all seven figures, from a keyword list rebuilt out
  of the five clusters the document names. The list itself was never recorded, so the exact figures cannot be
  recovered, only bracketed; the funnel's shape held under every variant tried. Both the list and this tolerance
  are now written into `docs/commons_face_corpus.md`, along with the arithmetic by which the document's two
  concentration tables confirm each other without any reconstruction at all.

## 13. Two defects inherited from upstream, fixed

Both were found by writing the tests in section 12 rather than by reading the code, and both are in
`gfpgan/models/gfpgan_model.py`. Upstream at `7552a77` has each of them verbatim.

**The facial component crops collapsed below `out_size` 512.** `get_roi_regions` scaled its crop sizes by
`int(out_size / 512)`, which is 0 for every size under 512: the crop sizes became 0 and the crops themselves were
multiplied by 0, so anyone training at 256 with `crop_components` got component discriminators fed blank patches
and three losses that meant nothing, with no error. The ratio is now a float, the sizes round to at least one
pixel, and the pixel values are no longer scaled at all. At 512 this is identical, because the ratio was and is
exactly 1; at 1024 the crops are no longer multiplied by 2, which was never intentional. The component
discriminator still needs a crop of at least 12 pixels, which puts the floor for a usable `out_size` near 128.

**`remove_pyramid_loss` did not remove the pyramid loss.** Past the threshold the weight became 1e-12 rather than
0, to keep the generator's `toRGB` layers from being reported as unused parameters by DistributedDataParallel.
Those layers receive a gradient from nothing else, and Adam divides by the gradient's own magnitude, so a
consistent 1e-12 signal produced a full-size step: for exactly the layers the option governs, the loss went on
training them for the remaining 750,000 iterations of an 800,000 iteration schedule. The weight is now exactly 0,
which freezes them, and the `l_p_*` terms stop being logged, so the log agrees with the option's name. The loss
is still computed at weight zero, which is what keeps the parameters in the graph.

No published measurement changes. The `toRGB` layers are a side channel: perturbing them by ten standard
deviations leaves the restored image bit-identical, which `tests/test_gfpgan_arch.py` now asserts, so their drift
could not have reached any metric in `docs/training_stability.md`.

## Still open

- **No pretrained generative prior.** Upstream starts from a StyleGAN2 trained on FFHQ; here the decoder trains
  from scratch, which is the main reason its restorations are weaker on real photographs.
- **The face model damages an input that is already good.** Measured on 64 held-out faces, it scores 12.28 dB
  below leaving the input untouched once that input is clean, and loses on all 64. `mild_prob` 0.5, now the
  default in `options/train_gfpgan_clean.yml`, recovers 1.9 dB of that on every one of the 64 and gains a little
  on the degraded regime as well, against a control that proves the extra iterations alone do not help. It
  closes only about 15% of the hole and the returns diminish sharply, so the defect stays open. A continuous
  Beta severity law was measured against it and tied to within 0.066 dB. The rest is now traded at inference
  instead: `-w/--weight` blends the restoration back toward the input, and its default of 0.75 buys 2.0 dB on a
  clean face, on 64 of 64, at no measurable cost on a degraded one. That is a mitigation, not a repair -- the
  model still cannot improve on a good input, only avoid spoiling it. Details in
  `docs/training_stability.md`.
- **Licensed corpus.** One exists now: 1,414 training faces from Wikimedia Commons, on which the published
  recipe was confirmed stable (run 13). At 2% of FFHQ it cannot produce a good prior, and the measurement runs
  use FFHQ as a stand-in, so the weights produced here remain for evaluation only.
- **Release weights.** The official checkpoints stay out of this repository, and the build produced here is not a
  release candidate.
