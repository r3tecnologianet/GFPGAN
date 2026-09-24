# Face restoration alternatives: CodeFormer, GPEN, VQFR, CFRNet

An assessment of four published face restoration methods against this fork's constraints: no
commercially-restricted code or weights, no `basicsr.ops.*` or StyleGAN2 CUDA kernels, no DFDNet/PSFRGAN/ParseNet
lineage, and no licensed pretrained face weights, so anything adopted has to be trainable from scratch.

Technical analysis, not legal advice. Every license statement below was read from a primary source; where a
primary source could not be reached, this says so instead of guessing.

## Summary

| Method | Code license | Released weights | Blocked by | Usable here |
|---|---|---|---|---|
| CodeFormer | **S-Lab License 1.0, non-commercial** | No separate license; FFHQ-trained | License, then data | Idea only |
| GPEN | **No LICENSE file at all** | Unlicensed; author cites "commercial issues" | License, then data | Idea only |
| VQFR | **Apache 2.0** | No documented license; FFHQ-trained | Data, and NC kernels in the discriminators | Partly |
| CFRNet | No code published | None exist | Nothing to adopt yet | Idea only |

**The licensing differences turn out not to be the deciding factor.** Three of the four need a codebook or a
generative prior pretrained on roughly 70,000 high-quality faces, and all four were trained on FFHQ, which NVIDIA
licenses CC BY-NC-SA 4.0. This fork has a single 8 GB GPU, 9,927 aligned FFHQ faces it may use only for evaluation, and a licensed
corpus of 1,414. The binding constraint is the
one already recorded as open in `LICENSE_CLEANUP.md`: there is no licensed face corpus and no licensed prior.

## CodeFormer (NeurIPS 2022)

A VQGAN codebook (1024 codes, 256 dims, compression 32, so a 512x512 face becomes a 16x16 token sequence), then a
9-layer transformer that predicts each token as a classification over the codebook, then a controllable feature
transformation inserted at decoder resolutions 32 to 256. The transformation carries a weight `w` applied as
`F + w * (F * scale + shift)`, which trades fidelity against realism at inference time without retraining.

**Blocked.** The repository LICENSE is "S-Lab License 1.0", whose grant covers "Redistribution and use for
**non-commercial purpose**" only. No separate license exists for the released weights in any primary source, so
they inherit that restriction -- including `vqgan_code1024.pth`, the codebook the later stages cannot run without.
The vendored tree also carries `basicsr/ops/{fused_act,upfirdn2d,dcn}` in the import path and a
`facelib/parsing/parsenet.py` headed "Modified from PSFRGAN", both already banned here by name.

The two architecture files are pure PyTorch and need no CUDA ops, so a clean-room reimplementation is feasible in
principle. It is not feasible in practice at this scale: three strictly sequential stages with codebook
pretraining as a hard prerequisite, and the authors report 1.5M/200K/20K iterations on four V100 GPUs. Stages I
and III also use LPIPS, which pulls pretrained VGG -- the same ImageNet restrictiin this fork already removed in
favour of discriminator feature matching.

## GPEN (CVPR 2021)

A U-shaped encoder whose deepest 4x4 feature is flattened into **one** global latent code, while the shallower
feature maps are **concatenated into the noise slots** of an embedded StyleGAN2 decoder. Because concatenation
doubles each convolution's input channels, the decoder is not shape-compatible with a stock StyleGAN2 checkpoint;
the prior has to be pretrained in GPEN's own modified block form.

Contrast with what this repository already does: `GFPGANv1Clean` produces a W+ style code per layer and conditions
the decoder through SFT affine layers applied to half the channels (`sft_half`). GPEN replaces both -- one global
latent instead of W+, concatenation instead of scale-and-shift, no channel split.

**Blocked, and by the strongest form of it.** The repository has no LICENSE file at all: absent from the root,
the GitHub license API returns 404 and reports `license: None`, and there is only one branch. That is not a
restrictive license, it is the absence of any grant, so the default reservation of rights applies. The README
supplies the reason in the authors' own words -- a model was taken down "due to commercial issues", and the
published weight is "not our best model due to commercial issues". Independently, the vendored
`face_model/op/*_kernel.cu` carry the NVIDIA Source Code License-NC header and `face_parse` is PSFRGAN's ParseNet
under CC BY-NC-SA 4.0.

**But the mechanism is the one finding here that fits this fork's actual constraint.** The released
`train_simple.py` trains the whole network jointly from scratch with no pretrained prior, and the authors claim
comparable performance to the paper. Unlike the codebook methods, GPEN's idea does not presuppose a prior that
this fork cannot obtain, and expressing it means changing how the existing `stylegan2_clean_arch.py` decoder is
conditioned -- a small, well-specified edit to code already owned here. That claim of comparable performance is
the authors' and is unverified, and their setup assumes 70,000 faces against the 9,927 FFHQ faces evaluated with here, or the 1,414 licensed ones.

## VQFR (ECCV 2022)

A VQ codebook (1024 codes, 256 dims) at compression 32, chosen by ablation: smaller patches fail to remove
degradation, larger ones recover texture but drift off identity. A parallel decoder then combines a texture
decoder, which sees only quantized features and so stays uncontaminated by the degraded input, with a main branch
that fuses the degraded input's features level by level. A texture warping module uses deformable convolution to
keep generated texture registered to the actual face.

**The only one of the four whose code is permissively licensed.** The LICENSE is Apache 2.0 from THL A29
Limited -- the same entity and license as GFPGAN itself -- with no non-commercial clause. The generator side
imports no restricted kernels, and the deformable convolution runs through `torchvision.ops.deform_conv2d`.

Three qualifications:

- The shipped discriminators are not usable. `stylegan_arch` and `swagan_arch` import vendored `upfirdn2d` whose
  `.cu` files carry the NVIDIA Source Code License-NC header, and `patch_disc_arch` pulls them transitively.
  Since `archs/__init__.py` auto-imports every `*_arch.py`, building any VQFR network imports them. They would
  have to be replaced -- which is precisely what this fork already did once for GFPGAN's discriminators.
- **A discrepancy worth recording.** VQFR's LICENSE and README claim Apache 2.0 over the repository and never
  disclose that NC lineage, while GFPGAN's own LICENSE does disclose its equivalents and reproduces the NVIDIA
  text. For those two files the file header should be treated as controlling, so the repository is not uniformly
  Apache 2.0 in practice.
- The released weights and codebook have no documented license anywhere in the primary sources and are
  FFHQ-derived, so they stand exactly where GFPGAN's own release weights stand: unusable here.

Training from scratch is blocked by data rather than license. Stage-one codebook pretraining is structurally
mandatory, not merely convenient: in stage two the codebook and texture decoder are excluded from every optimizer,
and the frozen high-quality model supplies the ground-truth codes for the code-alignment loss. Skipping stage one
leaves a randomly initialised prior that nothing will ever train. The authors' setup is roughly 800K plus 200K
iterations at batch 16 across 8 GPUs, on 70,000 faces.

## CFRNet (preprint, June 2026)

The name is ambiguous and mostly belongs to other fields. The best-known bearer is Counterfactual Regression
(Shalit et al., ICML 2017), which concerns causal inference, not images; the name is also taken by methods in road
extraction, RGB-T saliency, depth estimation, fundus image restoration and fault diagnosis. No "Coarse-to-Fine
Restoration Network" for faces exists.

A genuine face restoration CFRNet does exist -- arXiv 2606.06850, "Cycle-Consistent Fixed-Point Training for
Real-Time Blind Face Restoration on Consumer Embedded NPUs" -- but it is an unreviewed preprint from June 2026
with no repository, no weights and no citations, and it postdates the available surveys. Its numbers should not be
relied on.

Its architecture is deliberately plain: 2.0M parameters, convolution, ReLU, residual addition and nearest-neighbor
upsampling only, with no attention, no codebook and no generative prior, motivated by what compiles and quantizes
on small NPUs. The contribution is on the training side and costs nothing at inference: the network is trained to
be a fixed-point operator, so re-applying it to an already-restored face does not change it. Three terms --
progressive multi-cycle supervision, an idempotence loss, and a re-degradation cycle loss -- and the cycle count
becomes a quality knob that needs no retraining.

**The idempotence term is directly relevant to a defect measured in this repository.** As recorded in
`background_super_resolution.md`, the background model trained here erases genuine texture from a clean
photograph, because every training sample it saw was heavily degraded and it never learned to leave a good input
alone. An idempotence loss states that requirement directly. The fix proposed there -- adding near-identity
samples to the degradation pipeline -- is the same idea arrived at from the other direction.

Note that CFRNet's full recipe also uses a pretrained VGG-19 perceptual loss and an ArcFace identity loss, both of
which this fork has already rejected. Only the cycle and idempotence terms are adoptable.

## What would actually help

Nothing here is adoptable wholesale. Two cheap experiments are, and neither needs a pretrained prior or a larger
corpus:

1. **The idempotence loss**, on the existing background model. It addresses a failure already measured rather
   than a hypothetical one, it is a loss-side change costing nothing at inference, and it can be tested with the
   same short protocol used for runs 03 to 08.
2. **GPEN-style conditioning**, reimplemented clean-room over `stylegan2_clean_arch.py`: one global latent and
   encoder features concatenated into the noise slots, instead of W+ with channel-split SFT. Its value is that it
   trains jointly from scratch, which is this fork's situation, and it can be compared against the existing
   configuration under the training protocol in `training_stability.md`.

Neither justifies copying any file, snippet or weight from the repositories above.
