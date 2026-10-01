# Face Fine-Tuning, Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix inference alignment, add a second-order degradation and alignment jitter to the face dataset, then fine-tune the run 10 generator for 50,000 iterations and measure it per skin tone group.

**Architecture:** Inference alignment moves from BlazeFace's four keypoints to Face Landmarker eye and lip centres fitted by least squares (`gfpgan/face_helper.py`). Two opt-in options join `FFHQDegradationDataset` (`second_order_prob`, `align_jitter`), both off by default so every existing config degrades exactly as before. The discriminator warm-up needs no code: `net_d_init_iters` already freezes the generator, and a test pins that. The fine-tune itself runs from a launch script outside the repository, because its config holds local data paths.

**Tech Stack:** Python 3.11, PyTorch 2.1.2, BasicSR 1.4.2, MediaPipe 0.10.14, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-face-finetune-design.md`

**Scope:** Stage 1 and the alignment fix only. Stage 2 (JTT) depends on the stage 1 model and has its own pre-training check, so it gets its own plan once stage 1 is accepted.

## Global Constraints

- Every model trained here starts from FFHQ-trained weights and is **for local use only**: nothing trained is committed, released or given a download URL.
- Evaluation data lives under `/mnt/dados/gfpgan-clean/eval/`, outside the repository; no dataset URL enters the repository.
- Do not import `basicsr.ops.*` or `basicsr.archs.stylegan2_arch`; no DFDNet, ParseNet or PSFRGAN-derived code.
- No LPIPS, FID or ArcFace in evaluation.
- The GPU is shared: a run starts only when `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty, and no other process is ever stopped.
- Python: `PY=/mnt/dados/gfpgan-clean/venv/bin/python`; tools: `/mnt/dados/gfpgan-clean/venv/bin/{flake8,isort,yapf,codespell}`.
- Line length 120, single quotes, yapf and isort as configured in `setup.cfg`.
- Repository text in English.
- Commit locally, never push. Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01KahREvu4TReFuUeNPTAxwN
  ```

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `gfpgan/face_helper.py` | modify | Landmarker keypoints, least-squares similarity, refined detection and alignment |
| `tests/conftest.py` | modify | Patched `FaceHelper` passes `landmarker=None` so doubles stay offline |
| `tests/test_face_helper.py` | modify | Tests for the refined alignment and its fallback |
| `gfpgan/data/ffhq_degradation_dataset.py` | modify | `second_order_prob`, `align_jitter`, `transform_box` |
| `tests/test_ffhq_second_order.py` | create | Second-order degradation tests |
| `tests/test_ffhq_align_jitter.py` | create | Alignment jitter tests |
| `tests/test_gfpgan_model.py` | modify | Warm-up freezes the generator; a `params_ema`-only checkpoint initialises both generators |
| `CLAUDE.md` | modify | Alignment description and the two dataset options |
| `/mnt/dados/gfpgan-clean/tools/align_check_helper.py` | create (outside repo) | End-to-end alignment check of the real `FaceHelper` |
| `/mnt/dados/gfpgan-clean/tools/tone_summary.py` | create (outside repo) | Per-group gain over the input from `eval_tone.py` output |
| `/mnt/dados/gfpgan-clean/stab/ft01.sh` | create (outside repo) | Builds `ft01.yml` and runs the fine-tune |
| `docs/training_stability.md` | modify | Run 14 results |

---

### Task 1: Face Landmarker alignment at inference

**Files:**
- Modify: `gfpgan/face_helper.py` (constants after `MEDIAPIPE_KEYPOINT_INDICES`; `FaceHelper.__init__`, `get_face_landmarks`, `align_warp_face`)
- Modify: `tests/conftest.py` (the `patch_network` lambda)
- Modify: `tests/test_face_helper.py`
- Modify: `CLAUDE.md` (Inference pipeline paragraph)
- Create: `/mnt/dados/gfpgan-clean/tools/align_check_helper.py`

**Interfaces:**
- Produces: `LANDMARKER_TEMPLATE_512` (3x2 float32), `landmarker_keypoints(points) -> ndarray (3, 2)`, `similarity_lstsq(src, dst) -> ndarray (2, 3) float32`, `FaceHelper(..., landmarker='auto')` where `'auto'` builds `FaceMeshLandmarker`, `None` disables refinement, and any object with `detect(bgr_uint8) -> (478, 2) | None` is used as is.

- [ ] **Step 1: Write the failing tests**

Change the existing `test_face_helper_align_and_paste` constructor call to `FaceHelper(upscale_factor=2, face_det=_NoDetector(), landmarker=None)`, and update the import line to:

```python
from gfpgan.face_helper import (FACE_TEMPLATE_512, LANDMARKER_LEFT_EYE, LANDMARKER_LIPS, LANDMARKER_RIGHT_EYE,
                                LANDMARKER_TEMPLATE_512, FaceHelper, _nms, _window_positions, landmarker_keypoints,
                                similarity_lstsq)
```

Append to `tests/test_face_helper.py`:

```python
class _OneBoxDetector():
    """One face at a fixed box; its six BlazeFace keypoints are distinct and recognisable."""

    box = np.array([200, 220, 400, 460], np.float32)

    def detect(self, img):
        kps = np.arange(12, dtype=np.float32).reshape(1, 6, 2) + 300
        return self.box[None], np.array([0.9], np.float32), kps


class _FakeLandmarker():
    """Puts the eye and lip centres at fixed crop coordinates and records the crop it was given."""

    centres = np.array([[50, 60], [150, 60], [100, 200]], np.float32)

    def __init__(self, found=True):
        self.found = found
        self.crops = []

    def detect(self, img):
        self.crops.append(img.shape)
        if not self.found:
            return None
        points = np.zeros((478, 2), np.float32)
        for centre, idx in zip(self.centres, (LANDMARKER_LEFT_EYE, LANDMARKER_RIGHT_EYE, LANDMARKER_LIPS)):
            points[list(idx)] = centre
        return points


def test_landmarker_keypoints_are_the_contour_centres():
    points = np.random.RandomState(0).rand(478, 2).astype(np.float32) * 500
    expected = [points[list(idx)].mean(0) for idx in (LANDMARKER_LEFT_EYE, LANDMARKER_RIGHT_EYE, LANDMARKER_LIPS)]
    np.testing.assert_allclose(landmarker_keypoints(points), expected, rtol=1e-6)


def test_similarity_lstsq_recovers_a_known_similarity():
    angle, scale, shift = np.radians(7), 1.3, np.array([12.0, -5.0])
    rot = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = np.array([[10, 20], [200, 25], [90, 180], [40, 300]], np.float64)
    dst = src @ rot.T + shift
    matrix = similarity_lstsq(src, dst)
    np.testing.assert_allclose(matrix[:, :2], rot, atol=1e-5)
    np.testing.assert_allclose(matrix[:, 2], shift, atol=1e-3)


def test_refined_keypoints_come_from_a_padded_crop_around_the_box():
    landmarker = _FakeLandmarker()
    helper = FaceHelper(upscale_factor=1, face_det=_OneBoxDetector(), landmarker=landmarker)
    helper.read_image(np.zeros((800, 800, 3), np.uint8))
    assert helper.get_face_landmarks() == 1
    # pad is half the longer box side (240 / 2 = 120): the crop spans x 80..520 and y 100..580
    assert landmarker.crops == [(480, 440, 3)]
    np.testing.assert_allclose(helper.all_landmarks[0], _FakeLandmarker.centres + [80, 100])


def test_without_landmarks_the_blazeface_keypoints_are_used():
    helper = FaceHelper(upscale_factor=1, face_det=_OneBoxDetector(), landmarker=_FakeLandmarker(found=False))
    helper.read_image(np.zeros((800, 800, 3), np.uint8))
    assert helper.get_face_landmarks() == 1
    expected = (np.arange(12, dtype=np.float32).reshape(6, 2) + 300)[:4]
    np.testing.assert_allclose(helper.all_landmarks[0], expected)


def test_three_keypoints_align_to_the_landmarker_template():
    helper = FaceHelper(upscale_factor=1, face_det=_NoDetector(), landmarker=None)
    helper.read_image(np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8))
    landmark = LANDMARKER_TEMPLATE_512 * 0.5 + np.array([20, 40], np.float32)
    helper.all_landmarks = [landmark]
    helper.align_warp_face()
    affine = helper.affine_matrices[0]
    np.testing.assert_allclose(landmark @ affine[:, :2].T + affine[:, 2], LANDMARKER_TEMPLATE_512, atol=1e-2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -o addopts="" tests/test_face_helper.py -v`
Expected: collection error, `ImportError: cannot import name 'LANDMARKER_LEFT_EYE'`.

- [ ] **Step 3: Implement**

In `gfpgan/face_helper.py`, add `from gfpgan.component_boxes import FaceMeshLandmarker` to the imports, and after `MEDIAPIPE_KEYPOINT_INDICES`:

```python
# Face Mesh indices of the eye and lip contours, in the observer's view (the subject's right eye is on the observer's
# left). Their centres replace BlazeFace's keypoints when the Face Landmarker refines a detection: BlazeFace's nose
# tip varies by 24 px between FFHQ faces, and FFHQ's own alignment does not use the nose.
LANDMARKER_LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LANDMARKER_RIGHT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
LANDMARKER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409)

# Where those three centres fall in a 512x512 FFHQ face: their mean over 1,000 FFHQ faces, each shrunk to 256 px in
# a 1024 canvas and found as below. On 1,000 other faces this alignment reproduces FFHQ's with 1.9 degrees of
# rotation spread and a median centre shift of 2.2 px, against 5.1 degrees and 10 px for the BlazeFace template
# (docs/superpowers/specs/2026-09-30-face-finetune-design.md).
LANDMARKER_TEMPLATE_512 = np.array([[194.967, 242.852], [317.677, 242.872], [256.204, 378.048]], dtype=np.float32)

# The landmarker runs on the detection box enlarged by this fraction of its longer side on every side.
LANDMARKER_CROP_PAD = 0.5


def landmarker_keypoints(points):
    """Left eye, right eye and lip centres (observer's view) from 478 Face Mesh landmarks."""
    points = np.asarray(points, dtype=np.float32)
    return np.stack([points[list(idx)].mean(axis=0) for idx in (LANDMARKER_LEFT_EYE, LANDMARKER_RIGHT_EYE,
                                                                LANDMARKER_LIPS)])


def similarity_lstsq(src, dst):
    """Least-squares similarity transform mapping src onto dst (Umeyama 1991), as a 2x3 matrix.

    Every point weighs the same. With three keypoints a robust estimator such as LMEDS has nothing to reject and
    measured worse (90th percentile centre shift 12.8 px against 9.4 px on BlazeFace's eyes and mouth).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mean_src, dst - mean_dst
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ e @ vt
    scale = (sig * np.diag(e)).sum() / (s**2).sum(axis=1).mean()
    return np.hstack([scale * rot, (mean_dst - scale * rot @ mean_src)[:, None]]).astype(np.float32)
```

In `FaceHelper.__init__`, add the parameter `landmarker='auto'` after `face_det=None`, document it in the docstring Args as:

```
        landmarker (object | str | None): Face landmarker with a ``detect(img)`` method returning (478, 2) points or
            None. ``'auto'`` creates a ``FaceMeshLandmarker``; None aligns with BlazeFace's keypoints only.
            Default: 'auto'.
```

and append to the body:

```python
        if landmarker == 'auto':
            landmarker = FaceMeshLandmarker(model_rootpath=model_rootpath)
        self.landmarker = landmarker
        self.landmarker_template = LANDMARKER_TEMPLATE_512 * (face_size / 512.0)
```

Add this method to `FaceHelper`, before `get_face_landmarks`:

```python
    def _landmarker_keypoints(self, box):
        """Eye and lip centres found by the Face Landmarker on a crop around the box, or None."""
        if self.landmarker is None:
            return None
        h, w = self.input_img.shape[0:2]
        x1, y1, x2, y2 = box[0:4]
        pad = LANDMARKER_CROP_PAD * max(x2 - x1, y2 - y1)
        cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        cx2, cy2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        points = self.landmarker.detect(np.ascontiguousarray(self.input_img[cy1:cy2, cx1:cx2], dtype=np.uint8))
        if points is None:
            return None
        return landmarker_keypoints(points) + np.array([cx1, cy1], dtype=np.float32)
```

In `get_face_landmarks`, replace `landmark = kps[list(MEDIAPIPE_KEYPOINT_INDICES)]` with:

```python
            landmark = self._landmarker_keypoints(box)
            if landmark is None:
                landmark = kps[list(MEDIAPIPE_KEYPOINT_INDICES)]
```

In `align_warp_face`, replace the `affine_matrix = cv2.estimateAffinePartial2D(...)` line with:

```python
            if len(landmark) == len(self.landmarker_template):
                affine_matrix = similarity_lstsq(landmark, self.landmarker_template)
            else:
                affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
```

In `tests/conftest.py`, the `patch_network` lambda body becomes:

```python
            lambda upscale_factor, face_size=512, det_model=None, model_rootpath=None: FaceHelper(
                upscale_factor, face_size=face_size, face_det=face_det, landmarker=None))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest -o addopts="" tests/test_face_helper.py tests/test_utils_enhance.py tests/test_inference_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Measure the real helper end to end**

Create `/mnt/dados/gfpgan-clean/tools/align_check_helper.py`:

```python
"""Alignment of the real FaceHelper against FFHQ's, on faces it has not been fitted to.

FFHQ faces 1000-1999 of the seed 0 shuffle (the template was fitted on 0-999), each shrunk to 256 px in a 1024
canvas. A perfect alignment maps the original 512 frame onto itself.
"""
import glob, json, random
import cv2
import numpy as np
from gfpgan.face_helper import FaceHelper

files = sorted(glob.glob('/mnt/dados/gfpgan-clean/ffhq512full/aligned512/*.png'))
random.Random(0).shuffle(files)
helper = FaceHelper(upscale_factor=1, model_rootpath='gfpgan/weights')
emb = np.array([[0.5, 0, 384], [0, 0.5, 384], [0, 0, 1.0]])
rows, refined, missed = [], 0, 0
for f in files[1000:2000]:
    canvas = np.full((1024, 1024, 3), 128, np.uint8)
    canvas[384:640, 384:640] = cv2.resize(cv2.imread(f), (256, 256), interpolation=cv2.INTER_AREA)
    helper.clean_all()
    helper.read_image(canvas)
    if helper.get_face_landmarks(only_keep_largest=True) == 0:
        missed += 1
        continue
    refined += len(helper.all_landmarks[0]) == 3
    helper.align_warp_face()
    m = np.vstack([helper.affine_matrices[0], [0, 0, 1]]) @ emb
    c = m @ np.array([256, 256, 1.0]) - 256
    rows.append((np.hypot(m[0, 0], m[1, 0]), np.degrees(np.arctan2(m[1, 0], m[0, 0])), np.hypot(c[0], c[1])))
r = np.array(rows)
out = {'n': len(rows), 'missed': missed, 'refined': refined, 'scale_mean': r[:, 0].mean(), 'scale_std': r[:, 0].std(),
       'rot_std': r[:, 1].std(), 'shift_median': np.median(r[:, 2]), 'shift_p90': np.percentile(r[:, 2], 90)}
print(json.dumps(out, default=float))
json.dump(out, open('/mnt/dados/gfpgan-clean/eval/align_check_helper.json', 'w'), default=float)
```

Run: `cd /home/regis/develop/GFPGAN && CUDA_VISIBLE_DEVICES= $PY /mnt/dados/gfpgan-clean/tools/align_check_helper.py 2>/dev/null | tail -1`
Expected: `refined` ≥ 990, `scale_mean` 1.00 ± 0.01, `rot_std` ≤ 2.0, `shift_median` ≤ 2.5, `shift_p90` ≤ 5.5. If not, stop and report: the template or the crop differs from what was measured.

- [ ] **Step 6: Update CLAUDE.md**

In the **Inference pipeline** paragraph, replace `aligns each face to 512x512 with a similarity transform from eyes, nose tip and mouth center` with:

`refines each detection with the MediaPipe Face Landmarker and aligns the face to 512x512 by a least-squares similarity transform from the eye and lip centres (falling back to BlazeFace's eyes, nose tip and mouth centre when no landmarks are found)`

- [ ] **Step 7: Lint and commit**

```bash
B=/mnt/dados/gfpgan-clean/venv/bin
$B/flake8 gfpgan/face_helper.py tests/test_face_helper.py tests/conftest.py
$B/isort --check-only --diff gfpgan/face_helper.py
$B/yapf -d gfpgan/face_helper.py
$PY -m py_compile gfpgan/face_helper.py tests/test_face_helper.py tests/conftest.py
git add gfpgan/face_helper.py tests/test_face_helper.py tests/conftest.py CLAUDE.md
git commit -m "Align faces from Face Landmarker eye and lip centres, which FFHQ's alignment matches

<trailer lines from Global Constraints>"
```

---

### Task 2: Second-order degradation option

**Files:**
- Modify: `gfpgan/data/ffhq_degradation_dataset.py`
- Create: `tests/test_ffhq_second_order.py`
- Modify: `CLAUDE.md` (Training paragraph)

**Interfaces:**
- Produces: dataset option `second_order_prob` (float, default 0); method `FFHQDegradationDataset._second_pass(img, w, h) -> ndarray` (float32 HWC at a size no smaller than `ceil(w / downsample_range[1])` by `ceil(h / downsample_range[1])`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ffhq_second_order.py`:

```python
import cv2
import numpy as np
import pytest
import random
import torch
from basicsr.data import build_dataset

import gfpgan  # noqa: F401  (registers the dataset)


def _face_like(size=128):
    """A smooth image with edges; white noise would make every degradation look alike (see test_ffhq_mild.py)."""
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
    image = np.stack([0.5 + 0.4 * np.sin(6 * x), 0.5 + 0.4 * np.cos(5 * y), 0.5 + 0.3 * np.sin(4 * (x + y))], axis=2)
    image[size // 4:3 * size // 4, size // 4:3 * size // 4] *= 0.6
    return np.clip(image * 255, 0, 255).astype(np.uint8)


@pytest.fixture
def gt_folder(tmp_path):
    for i in range(4):
        cv2.imwrite(str(tmp_path / f'{i:04d}.png'), _face_like())
    return str(tmp_path)


def _opt(gt_folder, **extra):
    opt = dict(
        name='faces',
        type='FFHQDegradationDataset',
        dataroot_gt=gt_folder,
        io_backend=dict(type='disk'),
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        out_size=128,
        use_hflip=False,
        phase='train',
        scale=1,
        blur_kernel_size=41,
        kernel_list=['iso', 'aniso'],
        kernel_prob=[0.5, 0.5],
        blur_sigma=[0.1, 10],
        downsample_range=[0.8, 8],
        noise_range=[0, 20],
        jpeg_range=[60, 100],
        color_jitter_prob=None,
        color_jitter_shift=20,
        color_jitter_pt_prob=None,
        gray_prob=0)
    opt.update(extra)
    return opt


def _psnr(item):
    mse = torch.mean((item['lq'] - item['gt'])**2).item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


def _seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _mean_psnr(ds, n=16):
    _seed()
    return np.mean([_psnr(ds[i % len(ds)]) for i in range(n)])


def test_a_config_without_the_option_is_unchanged(gt_folder):
    absent = build_dataset(_opt(gt_folder))
    zero = build_dataset(_opt(gt_folder, second_order_prob=0))
    assert absent.second_order_prob == 0
    _seed()
    a = absent[0]['lq']
    _seed()
    b = zero[0]['lq']
    assert torch.equal(a, b)


def test_the_second_pass_degrades_further(gt_folder):
    first = _mean_psnr(build_dataset(_opt(gt_folder, second_order_prob=0)))
    second = _mean_psnr(build_dataset(_opt(gt_folder, second_order_prob=1)))
    assert second < first - 0.5, f'second order {second:.2f} dB is not below first order {first:.2f} dB'


def test_a_mild_sample_skips_the_second_pass(gt_folder):
    both = _mean_psnr(build_dataset(_opt(gt_folder, mild_prob=1.0, second_order_prob=1)))
    assert both > 30, f'a mild sample was degraded twice: {both:.2f} dB'


def test_the_total_downsampling_stays_within_range(gt_folder):
    ds = build_dataset(_opt(gt_folder, second_order_prob=1))
    _seed()
    for _ in range(50):
        out = ds._second_pass(np.full((16, 16, 3), 0.5, np.float32), 128, 128)  # already at the 8x limit
        assert out.shape[0] >= 16 and out.shape[1] >= 16


def test_the_second_pass_does_not_mutate_instance_state(gt_folder):
    ds = build_dataset(_opt(gt_folder, second_order_prob=1))
    before = (list(ds.blur_sigma), list(ds.downsample_range), list(ds.noise_range), list(ds.jpeg_range))
    ds[0]
    assert before == (list(ds.blur_sigma), list(ds.downsample_range), list(ds.noise_range), list(ds.jpeg_range))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -o addopts="" tests/test_ffhq_second_order.py -v`
Expected: FAIL, `AttributeError: 'FFHQDegradationDataset' object has no attribute 'second_order_prob'`.

- [ ] **Step 3: Implement**

In `gfpgan/data/ffhq_degradation_dataset.py`, after the imports:

```python
# The second degradation pass of Real-ESRGAN (Wang et al. 2021, arXiv:2107.10833), with its published second-stage
# ranges. Applied at the low-quality size, after the first pass, so the input carries two rounds of blur, resampling,
# noise and compression, as photographs that were copied, resized and re-saved do.
SECOND_BLUR_PROB = 0.8
SECOND_BLUR_SIGMA = [0.2, 1.5]
SECOND_RESIZE_RANGE = [0.3, 1.2]  # output size over input size
SECOND_GAUSSIAN_NOISE_PROB = 0.5
SECOND_NOISE_RANGE = [1, 25]
SECOND_POISSON_SCALE_RANGE = [0.05, 2.5]
SECOND_GRAY_NOISE_PROB = 0.4
SECOND_JPEG_RANGE = [30, 95]
SECOND_INTERPOLATIONS = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)
```

In `__init__`, after `self.mild_prob = ...`:

```python
        # Probability of adding the second degradation pass above. A mild sample never gets it. Defaults to 0, which
        # reproduces the single-pass pipeline exactly.
        self.second_order_prob = opt.get('second_order_prob', 0)
```

Add the method after `color_jitter_pt`:

```python
    def _second_pass(self, img, w, h):
        """Degrade an already degraded low-quality image once more, at its own size.

        The resize is capped so that the total downsampling from the w x h ground truth stays within
        downsample_range, which keeps the second pass from producing inputs smaller than the first pass can.
        """
        if np.random.uniform() < SECOND_BLUR_PROB:
            kernel = degradations.random_mixed_kernels(['iso', 'aniso'], [0.5, 0.5],
                                                       21,
                                                       SECOND_BLUR_SIGMA,
                                                       SECOND_BLUR_SIGMA, [-math.pi, math.pi],
                                                       noise_range=None)
            img = cv2.filter2D(img, -1, kernel)
        factor = np.random.uniform(SECOND_RESIZE_RANGE[0], SECOND_RESIZE_RANGE[1])
        new_w = max(int(img.shape[1] * factor), int(math.ceil(w / self.downsample_range[1])))
        new_h = max(int(img.shape[0] * factor), int(math.ceil(h / self.downsample_range[1])))
        interpolation = SECOND_INTERPOLATIONS[np.random.randint(len(SECOND_INTERPOLATIONS))]
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        if np.random.uniform() < SECOND_GAUSSIAN_NOISE_PROB:
            img = degradations.random_add_gaussian_noise(img, SECOND_NOISE_RANGE, gray_prob=SECOND_GRAY_NOISE_PROB)
        else:
            img = degradations.random_add_poisson_noise(
                img, SECOND_POISSON_SCALE_RANGE, gray_prob=SECOND_GRAY_NOISE_PROB)
        return degradations.add_jpg_compression(img, int(np.random.uniform(SECOND_JPEG_RANGE[0], SECOND_JPEG_RANGE[1])))
```

In `__getitem__`, immediately before the `# resize to original size` comment:

```python
        # second degradation pass, never on a mild sample
        if not mild and self.second_order_prob > 0 and np.random.uniform() < self.second_order_prob:
            img_lq = self._second_pass(img_lq, w, h)
```

In `CLAUDE.md`'s **Training** paragraph, after `(blur, downsample, noise, JPEG, color jitter, grayscale)`, insert: `; with probability \`second_order_prob\` (default 0) a non-mild sample gets Real-ESRGAN's second degradation pass at the low-quality size, capped to \`downsample_range\``.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest -o addopts="" tests/test_ffhq_second_order.py tests/test_ffhq_mild.py tests/test_ffhq_degradation_dataset.py -v`
Expected: all PASS. If `test_the_second_pass_degrades_further` fails, print both means and report; do not lower the 0.5 dB margin without understanding why.

- [ ] **Step 5: Lint and commit**

```bash
B=/mnt/dados/gfpgan-clean/venv/bin
$B/flake8 gfpgan/data/ffhq_degradation_dataset.py tests/test_ffhq_second_order.py
$B/isort --check-only --diff gfpgan/data/ffhq_degradation_dataset.py
$B/yapf -d gfpgan/data/ffhq_degradation_dataset.py
$PY -m py_compile gfpgan/data/ffhq_degradation_dataset.py tests/test_ffhq_second_order.py
git add gfpgan/data/ffhq_degradation_dataset.py tests/test_ffhq_second_order.py CLAUDE.md
git commit -m "Add Real-ESRGAN's second degradation pass to the face dataset, off by default

<trailer lines from Global Constraints>"
```

---

### Task 3: Alignment jitter option

**Files:**
- Modify: `gfpgan/data/ffhq_degradation_dataset.py`
- Create: `tests/test_ffhq_align_jitter.py`
- Modify: `CLAUDE.md` (Training paragraph)

**Interfaces:**
- Consumes: the dataset as left by Task 2.
- Produces: dataset option `align_jitter` (dict with `rotation` in degrees, `scale` as a fraction, `shift` in pixels; default None); module function `transform_box(box, matrix, w, h) -> list[float]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ffhq_align_jitter.py`:

```python
import cv2
import numpy as np
import random
import torch
from basicsr.data import build_dataset

import gfpgan  # noqa: F401  (registers the dataset)
from gfpgan.data.ffhq_degradation_dataset import transform_box

BOX = [40.0, 50.0, 70.0, 80.0]


def _opt(folder, boxes_path=None, **extra):
    opt = dict(
        name='faces',
        type='FFHQDegradationDataset',
        dataroot_gt=folder,
        io_backend=dict(type='disk'),
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        out_size=128,
        use_hflip=False,
        phase='train',
        scale=1,
        blur_kernel_size=41,
        kernel_list=['iso', 'aniso'],
        kernel_prob=[0.5, 0.5],
        blur_sigma=[0.1, 10],
        downsample_range=[0.8, 8],
        noise_range=[0, 20],
        jpeg_range=[60, 100],
        color_jitter_prob=None,
        color_jitter_shift=20,
        color_jitter_pt_prob=None,
        gray_prob=0)
    if boxes_path:
        opt.update(crop_components=True, component_path=boxes_path)
    opt.update(extra)
    return opt


def _seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _folder_with_square(tmp_path):
    """One 128x128 grey image with a white square filling BOX, and component boxes that all point at it."""
    img = np.full((128, 128, 3), 64, np.uint8)
    x1, y1, x2, y2 = [int(v) for v in BOX]
    img[y1:y2, x1:x2] = 255
    cv2.imwrite(str(tmp_path / '0000.png'), img)
    boxes_path = str(tmp_path / 'boxes.pth')
    torch.save({'0000': {'left_eye': BOX, 'right_eye': BOX, 'mouth': BOX}}, boxes_path)
    return str(tmp_path), boxes_path


def test_transform_box_identity_and_shift():
    identity = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    assert transform_box(BOX, identity, 128, 128) == BOX
    shift = np.array([[1, 0, 5], [0, 1, -3]], np.float64)
    assert transform_box(BOX, shift, 128, 128) == [45.0, 47.0, 75.0, 77.0]


def test_transform_box_is_clipped_to_the_image():
    shift = np.array([[1, 0, 100], [0, 1, 0]], np.float64)
    assert transform_box(BOX, shift, 128, 128)[2] == 128.0


def test_zero_jitter_leaves_the_ground_truth_unchanged(tmp_path):
    folder, _ = _folder_with_square(tmp_path)
    plain = build_dataset(_opt(folder))
    zero = build_dataset(_opt(folder, align_jitter=dict(rotation=0, scale=0, shift=0)))
    assert torch.allclose(plain[0]['gt'], zero[0]['gt'], atol=1e-6)


def test_jitter_moves_the_ground_truth(tmp_path):
    folder, _ = _folder_with_square(tmp_path)
    plain = build_dataset(_opt(folder))
    _seed()
    moved = build_dataset(_opt(folder, align_jitter=dict(rotation=2, scale=0.05, shift=3)))[0]['gt']
    assert not torch.allclose(plain[0]['gt'], moved, atol=1e-3)


def test_component_boxes_follow_the_jittered_image(tmp_path):
    folder, boxes_path = _folder_with_square(tmp_path)
    ds = build_dataset(_opt(folder, boxes_path, align_jitter=dict(rotation=2, scale=0.05, shift=3)))
    for seed in range(10):
        _seed(seed)
        item = ds[0]
        white = (item['gt'][0] > 0.9).nonzero().float()  # rows, cols of the square after the warp
        cy, cx = white[:, 0].mean().item(), white[:, 1].mean().item()
        x1, y1, x2, y2 = item['loc_left_eye'].tolist()
        assert x1 <= cx <= x2 and y1 <= cy <= y2
        assert abs((x1 + x2) / 2 - cx) < 1.5 and abs((y1 + y2) / 2 - cy) < 1.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -o addopts="" tests/test_ffhq_align_jitter.py -v`
Expected: collection error, `ImportError: cannot import name 'transform_box'`.

- [ ] **Step 3: Implement**

In `gfpgan/data/ffhq_degradation_dataset.py`, add after the `SECOND_*` constants:

```python
def transform_box(box, matrix, w, h):
    """Axis-aligned box around the four corners of box after a 2x3 affine matrix, clipped to a w x h image."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float64)
    mapped = corners @ np.asarray(matrix, dtype=np.float64)[:, :2].T + np.asarray(matrix, dtype=np.float64)[:, 2]
    (nx1, ny1), (nx2, ny2) = mapped.min(axis=0), mapped.max(axis=0)
    return [float(np.clip(nx1, 0, w)), float(np.clip(ny1, 0, h)), float(np.clip(nx2, 0, w)), float(np.clip(ny2, 0, h))]
```

In `__init__`, after `self.second_order_prob = ...`:

```python
        # Random similarity transform of the ground truth, as {'rotation': degrees, 'scale': fraction, 'shift': px}.
        # Inference crops are not aligned exactly as FFHQ is: with the Face Landmarker alignment the residual is
        # about 2 degrees, 5% scale and a few pixels, and training on perturbed crops of that size keeps the model
        # from depending on a precision it never gets. None, the default, leaves the image untouched.
        self.align_jitter = opt.get('align_jitter')
```

In `__getitem__`, replace the whole `# facial component boxes (observer's view), flipped together with the image` block (through the `loc_left_eye, loc_right_eye, loc_mouth = [...]` statement) with:

```python
        # facial component boxes (observer's view), flipped together with the image
        if self.crop_components:
            boxes = self.component_boxes[osp.splitext(osp.basename(gt_path))[0]]
            if status[0]:
                boxes = flip_component_boxes(boxes, w)

        # alignment jitter, applied to the ground truth before any degradation and carried to the boxes
        if self.align_jitter:
            angle = np.random.uniform(-self.align_jitter['rotation'], self.align_jitter['rotation'])
            factor = np.random.uniform(1 - self.align_jitter['scale'], 1 + self.align_jitter['scale'])
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, factor)
            matrix[:, 2] += np.random.uniform(-self.align_jitter['shift'], self.align_jitter['shift'], 2)
            img_gt = cv2.warpAffine(
                img_gt, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if self.crop_components:
                boxes = {name: transform_box(boxes[name], matrix, w, h) for name in ('left_eye', 'right_eye', 'mouth')}

        if self.crop_components:
            loc_left_eye, loc_right_eye, loc_mouth = [
                torch.tensor(boxes[name], dtype=torch.float32) for name in ('left_eye', 'right_eye', 'mouth')
            ]
```

In `CLAUDE.md`'s **Training** paragraph, after the `second_order_prob` insertion from Task 2, add: `; \`align_jitter\` ({rotation, scale, shift}, default off) warps the ground truth by a random similarity transform before degradation and moves the component boxes with it`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest -o addopts="" tests/test_ffhq_align_jitter.py tests/test_ffhq_second_order.py tests/test_ffhq_mild.py tests/test_ffhq_degradation_dataset.py tests/test_gfpgan_model_components.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
B=/mnt/dados/gfpgan-clean/venv/bin
$B/flake8 gfpgan/data/ffhq_degradation_dataset.py tests/test_ffhq_align_jitter.py
$B/isort --check-only --diff gfpgan/data/ffhq_degradation_dataset.py
$B/yapf -d gfpgan/data/ffhq_degradation_dataset.py
$PY -m py_compile gfpgan/data/ffhq_degradation_dataset.py tests/test_ffhq_align_jitter.py
git add gfpgan/data/ffhq_degradation_dataset.py tests/test_ffhq_align_jitter.py CLAUDE.md
git commit -m "Add alignment jitter to the face dataset, moving the component boxes with the image

<trailer lines from Global Constraints>"
```

---

### Task 4: Pin the warm-up and the fine-tune initialisation

**Files:**
- Modify: `tests/test_gfpgan_model.py`

**Interfaces:**
- Consumes: `GFPGANModel` options `net_d_init_iters`, `path.pretrain_network_g`, `path.param_key_g` (existing).
- Produces: nothing new; the fine-tune config in Task 5 relies on these two behaviours.

- [ ] **Step 1: Write the tests**

Append to `tests/test_gfpgan_model.py`:

```python
def _data():
    return {'lq': torch.rand((2, 3, 32, 32)) * 2 - 1, 'gt': torch.rand((2, 3, 32, 32)) * 2 - 1}


def test_the_generator_is_frozen_during_the_discriminator_warm_up():
    """The fine-tune starts its discriminators from scratch and relies on net_d_init_iters to train them alone."""
    opt = _get_opt(feature_matching_weight=1.0)
    opt['train']['net_d_init_iters'] = 5
    model = GFPGANModel(opt)
    model.feed_data(_data())
    g_before = [p.detach().clone() for p in model.net_g.parameters()]
    d_before = [p.detach().clone() for p in model.net_d.parameters()]
    model.optimize_parameters(current_iter=1)
    assert all(torch.equal(b, p) for b, p in zip(g_before, model.net_g.parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(d_before, model.net_d.parameters()))
    assert 'l_g_pix' not in model.log_dict
    model.optimize_parameters(current_iter=6)
    assert any(not torch.equal(b, p) for b, p in zip(g_before, model.net_g.parameters()))


def test_a_checkpoint_with_only_ema_weights_initialises_both_generators(tmp_path):
    """The exported run 10 file holds params_ema alone, and the fine-tune starts from it."""
    source = GFPGANModel(_get_opt(feature_matching_weight=1.0)).net_g
    path = tmp_path / 'ema_only.pth'
    torch.save({'params_ema': source.state_dict()}, path)
    opt = _get_opt(feature_matching_weight=1.0)
    opt['path'].update(pretrain_network_g=str(path), param_key_g='params_ema')
    model = GFPGANModel(opt)
    for net in (model.net_g, model.net_g_ema):
        for (name, a), b in zip(source.state_dict().items(), net.state_dict().values()):
            assert torch.equal(a.cpu(), b.cpu()), name
```

- [ ] **Step 2: Run them**

Run: `$PY -m pytest -o addopts="" tests/test_gfpgan_model.py -v`
Expected: PASS without code changes (the behaviour exists). If either fails, stop: Task 5's config depends on both, so report instead of working around it.

- [ ] **Step 3: Full suite, lint and commit**

```bash
$PY -m pytest
B=/mnt/dados/gfpgan-clean/venv/bin; $B/flake8 tests/test_gfpgan_model.py; $B/codespell
git add tests/test_gfpgan_model.py
git commit -m "Pin the discriminator warm-up and EMA-only initialisation the fine-tune relies on

<trailer lines from Global Constraints>"
```
Expected: the whole suite passes on the GPU, with no skips.

---

### Task 5: Run the stage 1 fine-tune (run 14)

**Files:**
- Create: `/mnt/dados/gfpgan-clean/stab/ft01.sh` (outside the repository)

**Interfaces:**
- Consumes: `second_order_prob`, `align_jitter` (Tasks 2–3); `net_d_init_iters`, `param_key_g` (Task 4); `/mnt/dados/gfpgan-clean/stab/base12_ffhq_full.yml`; `/mnt/dados/gfpgan-clean/tools/analyze.py`.
- Produces: `experiments/stab_run14_finetune/models/net_g_{5000..50000}.pth`, `stab/stab_run14_finetune.log`, moved afterwards to `/mnt/dados/gfpgan-clean/experiments/`.

- [ ] **Step 1: Write the launch script**

```bash
#!/bin/bash
# Run 14: stage 1 fine-tune of the run 10 generator (docs/superpowers/specs/2026-09-30-face-finetune-design.md).
# Learning rates are run 10's final values (its base lrs halved twice); discriminators start from scratch and train
# alone for 2,000 iterations; the pyramid loss stays off, as it was for run 10's second half.
# Usage: ft01.sh [ITERS] [NAME]
set -o pipefail
T=/mnt/dados/gfpgan-clean; ST=$T/stab; D=$T/ffhq512full
ITERS=${1:-50000}
NAME=${2:-stab_run14_finetune}
. $T/venv/bin/activate
cd /home/regis/develop/GFPGAN

# wait for the GPU without touching whatever holds it
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
echo "=== gpu free, starting at $(date +%H:%M)"

python - "$ITERS" "$NAME" <<PY
import sys, yaml
o = yaml.safe_load(open('$ST/base12_ffhq_full.yml'))
iters, o['name'] = int(sys.argv[1]), sys.argv[2]
tr = o['datasets']['train']
tr.update(mild_prob=0.5, second_order_prob=0.5, align_jitter=dict(rotation=2, scale=0.05, shift=3))
o['path'].update(pretrain_network_g='$T/export/gfpgan-clean-ffhq-100k/gfpgan_clean_ffhq_100k.pth',
                 param_key_g='params_ema', strict_load_g=True)
t = o['train']
t['optim_g']['lr'] = 2.5e-5 / 4
t['optim_d']['lr'] = 2.0e-5 / 4
t['optim_component']['lr'] = 2.5e-5 / 4
t['scheduler']['milestones'] = [10**9]
t['total_iter'] = iters
t['net_d_init_iters'] = 2000
t['remove_pyramid_loss'] = 0
o['val'].update(val_freq=2500, save_img=False)
o['logger']['save_checkpoint_freq'] = 5000
yaml.safe_dump(o, open(f'$ST/{o["name"]}.yml', 'w'), sort_keys=False)
PY
LOG=$ST/$NAME.log
python gfpgan/train.py -opt $ST/$NAME.yml > $LOG 2>&1
rc=$?
err=$(grep -m1 -E "Error|Traceback" $LOG | cut -c1-150)
echo "RUN14 exit=$rc iters=$(grep -cE 'iter:\s+[0-9,]+,' $LOG) ${err:+ERR: $err}"
python $T/tools/analyze.py $LOG --required 5000
echo "RUN14 DONE"
```

- [ ] **Step 2: Smoke-test 50 iterations**

Run: `bash /mnt/dados/gfpgan-clean/stab/ft01.sh 50 smoke_run14`
Expected (the name avoids `debug`, which would make BasicSR validate every 8 iterations): `RUN14 exit=0 iters=50`; `stab/smoke_run14.log` shows `Loading GFPGANv1Clean model from .../gfpgan_clean_ffhq_100k.pth, with param key: [params_ema]`, finite `l_d` values, and no `l_g_pix` (the generator is still in its warm-up). Then remove the smoke run: `rm -r experiments/smoke_run14 /mnt/dados/gfpgan-clean/stab/smoke_run14.{yml,log}`.

- [ ] **Step 3: Launch the run in the background**

Run: `nohup bash /mnt/dados/gfpgan-clean/stab/ft01.sh > /mnt/dados/gfpgan-clean/stab/run14.out 2>&1 &`
Expected: `run14.out` prints `=== gpu free` and the log advances at about 0.6 s/iteration (about 8.5 hours in total). The repository disk has 71 GB free; ten checkpoints with training states fit, and Step 5 moves them to `/mnt/dados`.

- [ ] **Step 4: Check the warm-up at iteration 2,000**

Run: `grep -E 'iter:\s+2,000,' /mnt/dados/gfpgan-clean/stab/stab_run14_finetune.log`
Expected: `l_d_real` and `l_d_fake` (or `real_score` and `fake_score`) have separated: the discriminator scores real above fake. If they have not, report before the generator has trained for long; the spec's lr choice would need revisiting.

- [ ] **Step 5: When it finishes, confirm stability and move the experiment**

Expected in `run14.out`: `RUN14 exit=0 iters=50000` and `analyze.py` reporting no violation.

```bash
mkdir -p /mnt/dados/gfpgan-clean/experiments
mv /home/regis/develop/GFPGAN/experiments/stab_run14_finetune /mnt/dados/gfpgan-clean/experiments/
```

---

### Task 6: Evaluate run 14 against the acceptance criteria

**Files:**
- Create: `/mnt/dados/gfpgan-clean/tools/tone_summary.py` (outside the repository)

**Interfaces:**
- Consumes: `/mnt/dados/gfpgan-clean/tools/eval_tone.py` (exists), `eval/tone_labels_val.csv`, `eval/baseline_val/per_image.csv` (stage 0 baseline, model name `ours100k`).
- Produces: `eval/run14_val/per_image.csv`, a per-group table for every 5,000-iteration checkpoint.

- [ ] **Step 1: Write the summary tool**

```python
"""Per skin tone group, each model's gain over the degraded input, with 95% bootstrap intervals.

Usage: python tone_summary.py PER_IMAGE.csv [PER_IMAGE.csv ...]
"""
import csv, sys
import numpy as np

T = '/mnt/dados/gfpgan-clean/eval'
tone = {r['name']: r['tone'] for r in csv.DictReader(l for l in open(f'{T}/tone_labels_val.csv') if not l.startswith('#'))}
rows = {}
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        rows[(r['model'], r['name'])] = r
models = sorted({m for m, _ in rows if m != 'LQ'})
rng = np.random.default_rng(0)


def interval(v):
    boot = [rng.choice(v, len(v)).mean() for _ in range(4000)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


print('model\tgroup\tn\tpsnr_gain\tcomp_psnr_gain\tniqe\tlmd\tlmd_fail')
for m in models:
    for g in ('light', 'medium', 'dark'):
        names = [n for n in tone if tone[n] == g and (m, n) in rows and ('LQ', n) in rows]
        out = [m, g, str(len(names))]
        for k in ('psnr', 'comp_psnr'):
            v = np.array([float(rows[(m, n)][k]) - float(rows[('LQ', n)][k]) for n in names])
            lo, hi = interval(v)
            out.append(f'{v.mean():+.2f} [{lo:+.2f},{hi:+.2f}]')
        out.append(f"{np.mean([float(rows[(m, n)]['niqe']) for n in names]):.2f}")
        lmds = [float(rows[(m, n)]['lmd']) for n in names if rows[(m, n)]['lmd'] not in ('', 'None')]
        out += [f'{np.mean(lmds):.2f}', str(len(names) - len(lmds))]
        print('\t'.join(out))
```

- [ ] **Step 2: Evaluate every checkpoint**

```bash
cd /home/regis/develop/GFPGAN
T=/mnt/dados/gfpgan-clean; D=$T/ffhq512full; E=$T/experiments/stab_run14_finetune/models
$T/venv/bin/python $T/tools/eval_tone.py $T/stab/base12_ffhq_full.yml $D/component_boxes.pth $D/val_lq $D/val_gt \
  $T/eval/run14_val $(for i in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000; do echo "it$i=$E/net_g_$i.pth"; done)
$T/venv/bin/python $T/tools/tone_summary.py $T/eval/baseline_val/per_image.csv $T/eval/run14_val/per_image.csv
```
Expected: a table with `ours100k` and `it5000` … `it50000`, three groups each.

- [ ] **Step 3: Apply the acceptance criteria**

From the spec, a checkpoint is accepted when, against `ours100k`: NIQE is lower and LMD is lower overall, and no group's component PSNR gain falls by more than 0.2 dB. Pick the earliest accepted checkpoint with the best dark-group component PSNR gain. If none is accepted, the result is negative: record it in Task 7 as such, and do not start stage 2.

- [ ] **Step 4: Real photographs, if the user has downloaded them**

Only if `/mnt/dados/gfpgan-clean/eval/real/` contains image folders: run `inference_gfpgan.py` with `--model_path` set to the run 10 export and to the accepted checkpoint, on each folder, `-o $T/eval/real_out/<model>/<set>`, and build a side-by-side grid of 16 faces per set for visual review with the user. Skip this step otherwise and say so in Task 7.

---

### Task 7: Record run 14

**Files:**
- Modify: `docs/training_stability.md` (new section after "Run 13")

- [ ] **Step 1: Write the section**

Append a section `## Run 14: stage 1 fine-tune of the run 10 generator` containing: the change from run 10 (one table row in the existing format: initialisation, warm-up, `second_order_prob` 0.5, `align_jitter` 2°/5%/3 px, `mild_prob` 0.5, lrs, 50,000 iterations, wall time); the stability result from `analyze.py`; the per-group table from Task 6 for `ours100k` and the accepted checkpoint (or the best one, if none was accepted); the alignment fix numbers from Task 1 Step 5; and which acceptance criterion passed or failed. State that the weights are FFHQ-trained and local only.

- [ ] **Step 2: Commit**

```bash
B=/mnt/dados/gfpgan-clean/venv/bin; $B/codespell docs/training_stability.md
git add docs/training_stability.md
git commit -m "Record run 14, the stage 1 fine-tune, per skin tone group

<trailer lines from Global Constraints>"
```
