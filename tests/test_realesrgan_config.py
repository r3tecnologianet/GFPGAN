import torch
import yaml
from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY

import gfpgan  # noqa: F401, registers archs, datasets and models

CLEAN_ARCHS = {'SRVGGNetCompactClean', 'UNetDiscriminatorClean'}
RESTRICTED_KEYS = ('perceptual_opt', 'network_identity', 'vgg', 'RealESRGAN_x4plus', 'DIV2K', 'Flickr2K', 'OST')


def test_train_realesrgan_clean_config():
    path = 'options/train_realesrgan_clean.yml'
    with open(path, mode='r') as f:
        lines = f.read().splitlines()
    # comments explain which restricted datasets and weights are avoided, so only the settings are checked
    text = '\n'.join(line for line in lines if not line.strip().startswith('#'))
    opt = yaml.safe_load(text)

    # only clean, registered components
    network_types = {v['type'] for k, v in opt.items() if k.startswith('network')}
    assert network_types == CLEAN_ARCHS
    for arch in network_types:
        ARCH_REGISTRY.get(arch)
    MODEL_REGISTRY.get(opt['model_type'])
    DATASET_REGISTRY.get(opt['datasets']['train']['type'])

    # no restricted losses, networks or datasets
    for key in RESTRICTED_KEYS:
        assert key not in text, key
    assert opt['path']['pretrain_network_g'] is None
    assert opt['path']['pretrain_network_d'] is None
    assert opt['train']['feature_matching_weight'] > 0


def test_srvgg_clean_gradient_reaches_the_first_layer():
    """The 34-layer body must be initialised, or the first layer gets no gradient and only the shortcut trains.

    With the PyTorch default initialisation the measured ratio was 3e-13; with Kaiming it is about 0.2.
    """
    net = ARCH_REGISTRY.get('SRVGGNetCompactClean')(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4)
    net(torch.rand(1, 3, 32, 32)).mean().backward()
    first = net.body[0].weight.grad.abs().mean().item()
    last = net.body[-1].weight.grad.abs().mean().item()
    assert first > 0 and last > 0, (first, last)
    assert first / last > 1e-3, f'gradient vanishes through the body: {first:.3e} against {last:.3e}'


def test_realesrgan_clean_model_ignores_perceptual_loss():
    """A stray perceptual_opt must not pull an ImageNet network into training."""
    model_cls = MODEL_REGISTRY.get('RealESRGANCleanModel')
    source = model_cls.init_training_settings.__doc__ or ''
    assert 'cri_perceptual' in model_cls.init_training_settings.__code__.co_names, source
