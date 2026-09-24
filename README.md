> ## This is a licence-cleanup fork, not upstream GFPGAN
>
> A fork of [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) with every component carrying a
> non-commercial restriction removed or replaced: the NVIDIA StyleGAN2 CUDA ops, DFDNet-derived code,
> ParseNet, the VGG19 perceptual loss and the Real-ESRGAN background upsampler. What each blocker was and
> how it was closed is in [LICENSE_CLEANUP.md](LICENSE_CLEANUP.md).
>
> **What you can use today.** The code and the training pipeline. The background super-resolution weights
> that `--bg_model` takes, trained here on CC0 photographs of scenery, which carry no biometric data and are
> therefore distributable ([evidence](docs/background_super_resolution.md)). And the training recipe, whose
> settings were measured rather than chosen ([stability](docs/training_stability.md)).
>
> **The face weights are yours to supply, and that is permanent.** `inference_gfpgan.py` requires
> `--model_path` and downloads nothing. Every public face restorer rests on FFHQ, which is non-commercial,
> so no released checkpoint can be shipped from here. Collecting a replacement was tried and measured to its
> end: the largest licensed pool on Wikimedia Commons yields 1,414 training faces, 2% of FFHQ
> ([measurement](docs/commons_face_corpus.md)). Scale is not even the binding constraint. A copyright licence
> clears the photographer, not the consent of the person photographed, and that gate closes independently of
> how many images are collected.
>
> **So the honest scope is the code, the pipeline and the background weights** — with a documented recipe for
> anyone who holds face data they are entitled to train on.

[![python lint](https://github.com/r3tecnologianet/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/r3tecnologianet/GFPGAN/actions/workflows/pylint.yml)

GFPGAN restores a degraded face by guiding a StyleGAN2-style decoder with features taken from the input
itself, published as [Towards Real-World Blind Face Restoration with Generative Facial Prior](https://arxiv.org/abs/2101.04061)
(CVPR 2021). This fork keeps the method and replaces the parts that could not be used commercially: the
NVIDIA CUDA operators, the DFDNet-derived landmark code, ParseNet, the VGG19 perceptual loss, the RetinaFace
detector from facexlib and the Real-ESRGAN background upsampler. Face detection, alignment and facial
component landmarks come from **MediaPipe**; training still runs on **BasicSR**.

## Dependencies and installation

Python 3.8 to 3.11, and a CUDA GPU if you have one. Everything works on CPU, more slowly; the MediaPipe
detector and landmarker always run on CPU.

```bash
git clone https://github.com/r3tecnologianet/GFPGAN.git
cd GFPGAN
pip install torch==2.1.2 torchvision==0.16.2 "numpy<2"
pip install --no-build-isolation basicsr==1.4.2
pip install "mediapipe==0.10.14" -r requirements.txt
python setup.py develop
```

The pins are not cosmetic. basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, removed in
torchvision 0.17, which caps torchvision at 0.16 and therefore Python at 3.11 and numpy below 2. mediapipe
1.x needs numpy 2, so 0.10.x is what fits; it cannot run the BlazeFace full-range model, so the default
detector is short range. MediaPipe downloads its models into `gfpgan/weights/` on first use.

## Inference

Nothing is downloaded. Pass weights you hold and are licensed to use:

```bash
python inference_gfpgan.py -i <image_or_folder> -o results --model_path <weights.pth> \
    --arch clean --channel_multiplier 2 -s 2
```

| Flag | Meaning |
|---|---|
| `-i`, `--input` | Image or folder. **Required.** A folder must contain only images |
| `-o`, `--output` | Output folder. Default `results` |
| `--model_path` | Face restoration weights. **Required** |
| `--arch` | `clean` or `RestoreFormer`. Default `clean` |
| `--channel_multiplier` | Channel multiplier of the `clean` architecture. Default 2 |
| `-s`, `--upscale` | Final upsampling scale of the whole image. Default 2 |
| `-w`, `--weight` | How much of the restoration to keep, 0 to 1. Default 0.75 |
| `--bg_model` | Super-resolution weights for the background. Default none, which uses Lanczos |
| `--aligned` | Input faces are already aligned 512x512 crops; skips detection |
| `--only_center_face` | Restore only the centre face |
| `--suffix` | Suffix for the restored face files |
| `--ext` | `auto`, `jpg` or `png`. Default `auto` |

Two of those deserve a paragraph.

**`-w/--weight` is a real dial, and 0.75 rather than 1 is the measured default.** It blends the restored face
back toward the aligned input in `gfpgan/utils.py:blend_restoration`: 1 is the restoration, 0 the aligned 512
crop untouched. Under `--aligned` that crop is the input image; on the whole-image path the face is still
warped to 512 and pasted back through the feathered mask, so `-w 0` is not a no-op on the photograph. The
default comes from a sweep in [docs/training_stability.md](docs/training_stability.md): against 1, weight 0.75
gains 2.04 dB on a clean input on 64 of 64 faces and costs nothing measurable on a degraded one (-0.069 dB,
p=0.29, with a better median and a better SSIM), because the model damages a face that is already good.

**`--bg_model` replaces Lanczos on the background.** The architecture and scale are read from the checkpoint
itself (`gfpgan/bg_upsampler.py`) and nothing is downloaded. The model this slot was built for, and the
evidence for it, are in [docs/background_super_resolution.md](docs/background_super_resolution.md). Two
checkpoints of it can also be blended into one without retraining, which is network interpolation
(arXiv:1811.10515):

```bash
python scripts/interpolate_sr_weights.py --pixel A.pth --gan B.pth --alpha 0.5 -o blended.pth
```

## Training

Set the dataset paths in `options/train_gfpgan_clean.yml`, then:

```bash
python gfpgan/train.py -opt options/train_gfpgan_clean.yml
python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 \
    gfpgan/train.py -opt options/train_gfpgan_clean.yml --launcher pytorch
```

Facial component boxes for a folder of aligned 512x512 faces, which the config's `crop_components` needs:

```bash
python scripts/generate_component_boxes.py -i <aligned_faces_dir> -o <component_boxes.pth>
```

The configuration is not a guess. Its learning rates, GAN weight, R1 interval and gradient clip were selected
over twelve screening rounds and eleven runs, because upstream's learning rate diverges to NaN within four
iterations when there is no pretrained prior to start from. What was tried, what diverged and what was
measured is in [docs/training_stability.md](docs/training_stability.md).

## Tests and lint

```bash
pytest                    # 130 tests; uses CUDA when available
pytest --cov              # add statement coverage (needs pytest-cov)
codespell
flake8 .
isort --check-only --diff gfpgan/ scripts/ inference_gfpgan.py setup.py
yapf -r -d gfpgan/ scripts/ inference_gfpgan.py setup.py
```

The MediaPipe tests download their models on first run and reuse them afterwards.

## What is documented here

| Document | Contents |
|---|---|
| [LICENSE_CLEANUP.md](LICENSE_CLEANUP.md) | Every blocker, its status, and what was removed, added or kept with caveats |
| [docs/changes_vs_upstream.md](docs/changes_vs_upstream.md) | Executive summary of this fork against commit `7552a77` |
| [docs/training_stability.md](docs/training_stability.md) | The screening rounds and runs behind the training config, the comparison against official GFPGAN v1.4, and the `mild_prob` and `-w` measurements |
| [docs/background_super_resolution.md](docs/background_super_resolution.md) | The background model: what was measured, and what was refuted |
| [docs/commons_face_corpus.md](docs/commons_face_corpus.md) | How many licensed faces Wikimedia Commons can supply, measured to exhaustion |
| [docs/face_restoration_alternatives.md](docs/face_restoration_alternatives.md) | CodeFormer, GPEN, VQFR and CFRNet assessed against this fork's constraints |
| [Comparisons.md](Comparisons.md) | Why upstream's version comparison is not reproduced here |

## Upstream GFPGAN

This fork exists because of upstream's work, and the credit is theirs.

> **GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior**
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), Honglun Zhang, Ying Shan
> Applied Research Center (ARC), Tencent PCG
> [[Paper](https://arxiv.org/abs/2101.04061)] [[Project page](https://xinntao.github.io/projects/gfpgan)] [[Repository](https://github.com/TencentARC/GFPGAN)]

```bibtex
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

Upstream's released checkpoints, online demos and model zoo stay with upstream: they are trained on FFHQ, and
this repository neither distributes nor recommends them. Its own README, badges and download counts describe
that repository, not this one, so they are not reproduced here. GFPGAN is built on
[BasicSR](https://github.com/xinntao/BasicSR), which this fork still uses.

## Licence

Apache License 2.0, unchanged from upstream: see [LICENSE](LICENSE). That covers the code in this repository
and nothing else — not the weights you bring to it, and not the images you run it on.

## Contact

Questions about this fork belong in its own issue tracker:
<https://github.com/r3tecnologianet/GFPGAN/issues>. Questions about upstream GFPGAN belong in
[TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN).
