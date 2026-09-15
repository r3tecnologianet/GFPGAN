import torch

from gfpgan.archs.discriminator_arch import StyleGAN2DiscriminatorClean
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.models.gfpgan_model import GFPGANModel


def _get_opt(feature_matching_weight):
    return {
        'name': 'UnitTest',
        'model_type': 'GFPGANModel',
        'num_gpu': 1 if torch.cuda.is_available() else 0,
        'is_train': True,
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'network_g': {
            'type': 'GFPGANv1Clean',
            'out_size': 32,
            'num_style_feat': 512,
            'channel_multiplier': 1,
            'decoder_load_path': None,
            'fix_decoder': False,
            'num_mlp': 8,
            'input_is_latent': True,
            'different_w': True,
            'narrow': 0.5,
            'sft_half': True
        },
        'network_d': {
            'type': 'StyleGAN2DiscriminatorClean',
            'out_size': 32,
            'narrow': 0.25
        },
        'path': {
            'pretrain_network_g': None,
            'pretrain_network_d': None,
            'strict_load_g': True,
            'resume_state': None
        },
        'train': {
            'optim_g': {
                'type': 'Adam',
                'lr': 2e-3
            },
            'optim_d': {
                'type': 'Adam',
                'lr': 2e-3
            },
            'scheduler': {
                'type': 'MultiStepLR',
                'milestones': [600000, 700000],
                'gamma': 0.5
            },
            'total_iter': 10,
            'warmup_iter': -1,
            'pixel_opt': {
                'type': 'L1Loss',
                'loss_weight': 0.1,
                'reduction': 'mean'
            },
            'L1_opt': {
                'type': 'L1Loss',
                'loss_weight': 1,
                'reduction': 'mean'
            },
            'gan_opt': {
                'type': 'GANLoss',
                'gan_type': 'wgan_softplus',
                'loss_weight': 0.1
            },
            'feature_matching_weight': feature_matching_weight,
            'r1_reg_weight': 10,
            'net_d_iters': 1,
            'net_d_init_iters': 0,
            'net_d_reg_every': 1,
            'pyramid_loss_weight': 1,
            'remove_pyramid_loss': 50000
        }
    }


def test_gfpgan_model_feature_matching():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = {'lq': torch.rand((2, 3, 32, 32)) * 2 - 1, 'gt': torch.rand((2, 3, 32, 32)) * 2 - 1}

    # ------------------ with feature matching -------------------- #
    model = GFPGANModel(_get_opt(feature_matching_weight=1.0))
    assert isinstance(model.net_g, GFPGANv1Clean)
    assert isinstance(model.net_d, StyleGAN2DiscriminatorClean)
    assert model.cri_perceptual is None
    assert model.device == device

    params_before = [p.detach().clone() for p in model.net_g.parameters()]
    model.feed_data(data)
    model.optimize_parameters(current_iter=1)
    assert 'l_g_fm' in model.log_dict
    for key in ('l_g_fm', 'l_g_gan', 'l_g_pix', 'l_d', 'l_d_r1'):
        assert torch.isfinite(torch.tensor(model.log_dict[key])), key
    assert any(not torch.equal(b, p) for b, p in zip(params_before, model.net_g.parameters()))

    # ------------------ without feature matching -------------------- #
    model = GFPGANModel(_get_opt(feature_matching_weight=0))
    model.feed_data(data)
    model.optimize_parameters(current_iter=1)
    assert 'l_g_fm' not in model.log_dict
    assert 'l_g_gan' in model.log_dict
    assert 'g_grad_norm' not in model.log_dict

    # ------------------ with generator gradient clipping -------------------- #
    opt = _get_opt(feature_matching_weight=1.0)
    opt['train']['generator_grad_clip'] = 1.0
    model = GFPGANModel(opt)
    model.feed_data(data)
    model.optimize_parameters(current_iter=1)
    assert 'g_grad_norm' in model.log_dict
    assert torch.isfinite(torch.tensor(model.log_dict['g_grad_norm']))
