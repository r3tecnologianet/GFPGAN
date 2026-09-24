# Pre-trained models and other data

This fork downloads nothing. Put the weights you hold and are licensed to use in this folder, and name them
explicitly: `inference_gfpgan.py --model_path <weights.pth>`, `--bg_model <weights.pth>`, or
`pretrain_network_g` in a training config.

Upstream pointed this folder at three downloads. None of them is linked here, and each has a reason:

| What upstream put here | Why it is not linked |
|---|---|
| `StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth` | A generative prior trained on FFHQ, which is non-commercial. This fork trains its decoder from scratch instead, at a measured cost in restoration quality (`LICENSE_CLEANUP.md`) |
| `FFHQ_eye_mouth_landmarks_512.pth` | The output of `scripts/parse_landmark.py`, derived from DFDNet (CC BY-NC-SA 4.0). Both are gone; `scripts/generate_component_boxes.py` derives the boxes from MediaPipe Face Mesh instead |
| `arcface_resnet18.pth` | The identity loss, which the clean config does not use |

To build component boxes for your own aligned 512x512 faces:

```bash
python scripts/generate_component_boxes.py -i <aligned_faces_dir> -o <component_boxes.pth>
```
