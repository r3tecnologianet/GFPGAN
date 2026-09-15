import yaml
from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY

import gfpgan  # noqa: F401, registers archs, datasets and models

CLEAN_ARCHS = {'GFPGANv1Clean', 'StyleGAN2DiscriminatorClean', 'FacialComponentDiscriminatorClean'}
RESTRICTED_KEYS = ('perceptual_opt', 'network_identity', 'identity_weight', 'vgg', 'FFHQ_eye_mouth',
                   'StyleGAN2_512_Cmul1', 'arcface_resnet18', 'eye_enlarge_ratio')


def test_train_gfpgan_clean_config():
    path = 'options/train_gfpgan_clean.yml'
    with open(path, mode='r') as f:
        text = f.read()
    opt = yaml.safe_load(text)

    # only clean, registered components; the facial component discriminators are optional
    network_types = {v['type'] for k, v in opt.items() if k.startswith('network')}
    assert {'GFPGANv1Clean', 'StyleGAN2DiscriminatorClean'} <= network_types <= CLEAN_ARCHS
    for arch in network_types:
        ARCH_REGISTRY.get(arch)
    MODEL_REGISTRY.get(opt['model_type'])
    DATASET_REGISTRY.get(opt['datasets']['train']['type'])

    # no restricted losses, networks or weights
    for key in RESTRICTED_KEYS:
        assert key not in text, key
    assert opt['network_g']['decoder_load_path'] is None
    assert opt['train']['feature_matching_weight'] > 0
    if 'FacialComponentDiscriminatorClean' in network_types:
        assert opt['datasets']['train']['crop_components'] is True
