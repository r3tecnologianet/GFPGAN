import pytest
import torch
from basicsr.utils import DiffJPEG, USMSharp

from gfpgan.models.realesrgan_clean_model import RealESRGANCleanModel


def _degradation_opt(mild_prob):
    """The degradation half of a training config, which is all feed_data reads."""
    return {
        'scale': 4,
        'mild_prob': mild_prob,
        # feed_data crops after degrading, so gt_size is required even though it plays no part in the degradation.
        'gt_size': 256,
        'resize_prob': [0.2, 0.7, 0.1],
        'resize_range': [0.15, 1.5],
        'gaussian_noise_prob': 0.5,
        'noise_range': [1, 30],
        'poisson_scale_range': [0.05, 3],
        'gray_noise_prob': 0.4,
        'jpeg_range': [30, 95],
        'second_blur_prob': 0.8,
        'resize_prob2': [0.3, 0.4, 0.3],
        'resize_range2': [0.3, 1.2],
        'gaussian_noise_prob2': 0.5,
        'noise_range2': [1, 25],
        'poisson_scale_range2': [0.05, 2.5],
        'gray_noise_prob2': 0.4,
        'jpeg_range2': [30, 95],
    }


def _model(mild_prob, device):
    """Build only what feed_data touches.

    Constructing the model properly would need a generator, a discriminator and an optimizer, none of which take
    part in building the low quality input. __new__ skips that and the attributes below are what the inherited
    feed_data actually uses.
    """
    model = RealESRGANCleanModel.__new__(RealESRGANCleanModel)
    model.opt = _degradation_opt(mild_prob)
    model.is_train = True
    model.device = device
    model.usm_sharpener = USMSharp().to(device)
    model.jpeger = DiffJPEG(differentiable=False).to(device)
    # feed_data ends by passing the batch through a shuffling queue, read as an attribute rather than an option.
    # One entry, the batch size used here, makes it a trivial buffer that returns the pair it was just given.
    model.queue_size = 1
    return model


def _image(device, size=256):
    """A smooth ground truth with an edge in it.

    White noise must not be used here. Resizing picks between area, bilinear and bicubic at random, and JPEG
    cannot represent white noise at all, so on noise every degradation destroys the signal equally and the mild
    and heavy branches land within 0.15 dB of each other: the test would pass or fail for reasons unrelated to
    what it claims to check. On an image-like input the two branches separate by about 15 dB.
    """
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
    image = torch.stack([
        0.5 + 0.4 * torch.sin(6 * x),
        0.5 + 0.4 * torch.cos(5 * y),
        0.5 + 0.3 * torch.sin(4 * (x + y)),
    ])
    image[:, size // 4:3 * size // 4, size // 4:3 * size // 4] *= 0.6
    return image.unsqueeze(0).clamp(0, 1).to(device)


def _data(device, kernel_size=21):
    """An image-like ground truth plus the blur kernels the dataset would draw."""
    torch.manual_seed(0)
    gt = _image(device)
    kernel = torch.zeros(1, kernel_size, kernel_size, device=device)
    kernel[:, :, :] = 1.0 / (kernel_size * kernel_size)  # a heavy box blur, clearly not the identity
    return {'gt': gt, 'kernel1': kernel.clone(), 'kernel2': kernel.clone(), 'sinc_kernel': kernel.clone()}


def _psnr_against_downscaled(model):
    """How close the low quality input is to a plain downscale of the ground truth."""
    reference = torch.nn.functional.interpolate(model.gt, scale_factor=1 / 4, mode='bicubic').clamp(0, 1)
    mse = torch.mean((model.lq.clamp(0, 1) - reference)**2).item()
    return 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()


@pytest.mark.parametrize('device', ['cuda' if torch.cuda.is_available() else 'cpu'])
def test_mild_branch_is_really_mild(device):
    """The mild branch must neutralise the kernels too, not only the options.

    The kernels come from the dataset rather than from the options, so swapping the option dictionary alone would
    leave the sample blurred and the branch would change nothing it claims to change.
    """
    mild = _model(1.0, device)
    mild.feed_data(_data(device))
    mild_psnr = _psnr_against_downscaled(mild)

    heavy = _model(0.0, device)
    heavy.feed_data(_data(device))
    heavy_psnr = _psnr_against_downscaled(heavy)

    assert mild_psnr > 25, f'the mild branch is not near-identity: {mild_psnr:.2f} dB'
    assert mild_psnr > heavy_psnr + 6, f'mild {mild_psnr:.2f} dB is not clearly milder than heavy {heavy_psnr:.2f}'


@pytest.mark.parametrize('device', ['cuda' if torch.cuda.is_available() else 'cpu'])
def test_mild_prob_zero_restores_the_inherited_pipeline(device):
    """The default must reproduce the inherited behaviour exactly, so existing configs are untouched."""
    model = _model(0.0, device)
    before = dict(model.opt)
    model.feed_data(_data(device))
    assert model.opt == before


@pytest.mark.parametrize('device', ['cuda' if torch.cuda.is_available() else 'cpu'])
def test_options_are_restored_after_a_mild_batch(device):
    """A mild batch must not leave the degradation permanently weakened for later batches."""
    model = _model(1.0, device)
    before = {key: model.opt[key] for key in RealESRGANCleanModel.MILD_DEGRADATION}
    model.feed_data(_data(device))
    after = {key: model.opt[key] for key in RealESRGANCleanModel.MILD_DEGRADATION}
    assert after == before
