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
>
> Everything below this notice is upstream's README, including the badges, demo links and download counts,
> which report the upstream repository rather than this fork. It is kept with one edit, disclosed here: the
> download links for the official release weights, for upstream's extra-model folders and for FFHQ have been
> stripped from it. The prose stays as a record of what upstream distributes; the links do not, because this
> fork must not hand out weights and datasets it cannot license.

<div align="center">
<!-- <a href="https://twitter.com/_Xintao_" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/17445847/187162058-c764ced6-952f-404b-ac85-ba95cce18e7b.png" width="4%" alt="" />
</a> -->

[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/gfpgan)](https://pypi.org/project/gfpgan/)
[![Open issue](https://img.shields.io/github/issues/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
[![python lint](https://github.com/TencentARC/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/TencentARC/GFPGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/publish-pip.yml)
</div>

1. :boom: **Updated** online demo: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/tencentarc/gfpgan). Here is the [backup](https://replicate.com/xinntao/gfpgan).
1. :boom: **Updated** online demo: [![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/Xintao/GFPGAN)
1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for GFPGAN <a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>; (Another [Colab Demo](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) for the original paper model)

<!-- 3. Online demo: [Replicate.ai](https://replicate.com/xinntao/gfpgan) (may need to sign in, return the whole image)
4. Online demo: [Baseten.co](https://app.baseten.co/applications/Q04Lz0d/operator_views/8qZG6Bg) (backed by GPU, returns the whole image)
5. We provide a *clean* version of GFPGAN, which can run without CUDA extensions. So that it can run in **Windows** or on **CPU mode**. -->

> :rocket: **Thanks for your interest in our work. You may also want to check our new updates on the *tiny models* for *anime images and videos* in [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)** :blush:

GFPGAN aims at developing a **Practical Algorithm for Real-world Face Restoration**.<br>
It leverages rich and diverse priors encapsulated in a pretrained face GAN (*e.g.*, StyleGAN2) for blind face restoration.

:triangular_flag_on_post: **Updates**

- :white_check_mark: Add [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) inference codes.
- :white_check_mark: Add V1.4 model, which produces slightly more details and better identity than V1.3.
- :white_check_mark: Add **V1.3 model**, which produces **more natural** restoration results, and better results on *very low-quality* / *high-quality* inputs. See more in [Model zoo](#european_castle-model-zoo), [Comparisons.md](Comparisons.md)
- :white_check_mark: Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/GFPGAN).
- :white_check_mark: Support enhancing non-face regions (background) with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- :white_check_mark: We provide a *clean* version of GFPGAN, which does not require CUDA extensions.
- :white_check_mark: We provide an updated model without colorizing faces.

---

If GFPGAN is helpful in your photos/projects, please help to :star: this repo or recommend it to your friends. Thanks:blush:
Other recommended projects:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration<br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): An open-source image and video restoration toolbox<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison<br>

---

### :book: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://xinntao.github.io/projects/gfpgan)] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### Installation

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. <br>
The original paper model needs the NVIDIA CUDA extensions this fork removed, so its installation
instructions are not here.

1. Clone repo

    ```bash
    git clone https://github.com/TencentARC/GFPGAN.git
    cd GFPGAN
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr

    # Install facexlib - https://github.com/xinntao/facexlib
    # We use face detection and face restoration helper in the facexlib package
    pip install facexlib

    pip install -r requirements.txt
    python setup.py develop

    # If you want to enhance the background (non-face) regions with Real-ESRGAN,
    # you also need to install the realesrgan package
    pip install realesrgan
    ```

## :zap: Quick Inference

We take the v1.3 version for an example. More models can be found [here](#european_castle-model-zoo).

Download pre-trained models: GFPGANv1.3.pth

```bash
# Download link removed in this fork. Put weights you are licensed to use in experiments/pretrained_models
```

**Inference!**

```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2
```

```console
Usage: python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2 [options]...

  -h                   show this help
  -i input             Input image or folder. Default: inputs/whole_imgs
  -o output            Output folder. Default: results
  -v version           GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3
  -s upscale           The final upsampling scale of the image. Default: 2
  -bg_upsampler        background upsampler. Default: realesrgan
  -bg_tile             Tile size for background sampler, 0 for no tile during testing. Default: 400
  -suffix              Suffix of the restored faces
  -only_center_face    Only restore the center face
  -aligned             Input are aligned faces
  -ext                 Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

The original paper model needs the NVIDIA CUDA extensions this fork removed, so its instructions are not here.

## :european_castle: Model Zoo

| Version | Model Name  | Description |
| :---: | :---:        |     :---:      |
| V1.3 | GFPGANv1.3.pth | Based on V1.2; **more natural** restoration results; better results on very low-quality / high-quality inputs. |
| V1.2 | GFPGANCleanv1-NoCE-C2.pth | No colorization; no CUDA extensions are required. Trained with more data with pre-processing. |
| V1 | GFPGANv1.pth | The paper model, with colorization. |

The comparisons are in [Comparisons.md](Comparisons.md).

Note that V1.3 is not always better than V1.2. You may need to select different models based on your purpose and inputs.

| Version | Strengths  | Weaknesses |
| :---: | :---:        |     :---:      |
|V1.3 |  ✓ natural outputs<br> ✓better results on very low-quality inputs <br> ✓ work on relatively high-quality inputs <br>✓ can have repeated (twice) restorations | ✗ not very sharp <br> ✗ have a slight change on identity |
|V1.2 |  ✓ sharper output <br> ✓ with beauty makeup | ✗ some outputs are unnatural |

Upstream distributes **more models (such as the discriminators)** from its own cloud folders. Those links were removed here.

## :computer: Training

We provide the training codes for GFPGAN (used in our paper). <br>
You could improve it according to your own needs.

**Tips**

1. More high quality faces can improve the restoration quality.
2. You may need to perform some pre-processing, such as beauty makeup.

**Procedures**

(You can try a simple version ( `options/train_gfpgan_v1_simple.yml`) that does not require face component landmarks.)

1. Dataset preparation: FFHQ, whose link was removed here — it is the non-commercial dataset this fork exists to
   avoid depending on

1. Download pre-trained models and other data. Put them in the `experiments/pretrained_models` folder.
    1. Pre-trained StyleGAN2 model, component locations of FFHQ and an ArcFace model. Links removed here, with the
       reason for each in `experiments/pretrained_models/README.md`

1. Modify the configuration file `options/train_gfpgan_v1.yml` accordingly.

1. Training

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

## :scroll: License and Acknowledgement

GFPGAN is released under Apache License Version 2.0.

## BibTeX

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## :e-mail: Contact

Questions about **this fork** belong in its own issue tracker: <https://github.com/r3tecnologianet/GFPGAN/issues>.
Questions about upstream GFPGAN belong in [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN). Upstream
maintainers' personal e-mail addresses stood in this section and were removed: this repository is not theirs to
answer for.
