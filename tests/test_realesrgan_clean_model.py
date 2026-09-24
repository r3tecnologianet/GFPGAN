"""The loss path of the background training model.

What this fork changed in Real-ESRGAN's training is the guidance: the VGG19/ImageNet perceptual loss is gone and
discriminator feature matching takes its place. test_realesrgan_mild covers the degradation branch; this covers
the losses, which is the part that was rewritten.
"""
import torch

import gfpgan  # noqa: F401  (registers the clean archs)
from gfpgan.models.realesrgan_clean_model import RealESRGANCleanModel

SIZE = 32
SCALE = 2


def _opt(tmp_path, feature_matching_weight=1.0, perceptual_opt=None):
    for sub in ('models', 'training_states', 'visualization'):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    opt = {
        'name': 'UnitTest',
        'model_type': 'RealESRGANCleanModel',
        'scale': SCALE,
        'num_gpu': 1 if torch.cuda.is_available() else 0,
        'is_train': True,
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'queue_size': 2,
        'l1_gt_usm': False,
        'percep_gt_usm': False,
        'gan_gt_usm': False,
        'network_g': {
            'type': 'SRVGGNetCompactClean',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 8,
            'num_conv': 2,
            'upscale': SCALE
        },
        'network_d': {
            'type': 'UNetDiscriminatorClean',
            'num_in_ch': 3,
            'num_feat': 8
        },
        'path': {
            'pretrain_network_g': None,
            'pretrain_network_d': None,
            'strict_load_g': True,
            'resume_state': None,
            'models': str(tmp_path / 'models'),
            'training_states': str(tmp_path / 'training_states'),
            'visualization': str(tmp_path / 'visualization'),
            'experiments_root': str(tmp_path)
        },
        'train': {
            'ema_decay': 0.999,
            'optim_g': {
                'type': 'Adam',
                'lr': 1e-4
            },
            'optim_d': {
                'type': 'Adam',
                'lr': 1e-4
            },
            'scheduler': {
                'type': 'MultiStepLR',
                'milestones': [600000],
                'gamma': 0.5
            },
            'total_iter': 10,
            'warmup_iter': -1,
            'pixel_opt': {
                'type': 'L1Loss',
                'loss_weight': 1.0,
                'reduction': 'mean'
            },
            'gan_opt': {
                'type': 'GANLoss',
                'gan_type': 'vanilla',
                'real_label_val': 1.0,
                'fake_label_val': 0.0,
                'loss_weight': 0.1
            },
            'feature_matching_weight': feature_matching_weight,
            'net_d_iters': 1,
            'net_d_init_iters': 0
        },
        'val': {
            'val_freq': 1,
            'save_img': False
        },
        'logger': {
            'print_freq': 1
        }
    }
    if perceptual_opt is not None:
        opt['train']['perceptual_opt'] = perceptual_opt
    return opt


def _feed(model, batch=2):
    """Set what optimize_parameters reads, skipping the synthetic degradation covered elsewhere."""
    device = model.device
    model.gt = torch.rand((batch, 3, SIZE * SCALE, SIZE * SCALE), device=device)
    model.gt_usm = model.gt.clone()
    model.lq = torch.rand((batch, 3, SIZE, SIZE), device=device)


def test_the_perceptual_loss_is_refused_even_when_a_config_asks_for_it(tmp_path):
    """A stray option must not pull an ImageNet network into training."""
    model = RealESRGANCleanModel(_opt(tmp_path, perceptual_opt={'type': 'PerceptualLoss', 'layer_weights': {}}))
    assert model.cri_perceptual is None


def test_feature_matching_is_logged_when_weighted(tmp_path):
    model = RealESRGANCleanModel(_opt(tmp_path, feature_matching_weight=1.0))
    _feed(model)
    model.optimize_parameters(current_iter=1)
    assert 'l_g_fm' in model.log_dict
    for key in ('l_g_pix', 'l_g_gan', 'l_d_real', 'l_d_fake'):
        assert key in model.log_dict, key
        assert torch.isfinite(torch.tensor(model.log_dict[key])), key


def test_feature_matching_is_absent_at_weight_zero(tmp_path):
    model = RealESRGANCleanModel(_opt(tmp_path, feature_matching_weight=0))
    _feed(model)
    model.optimize_parameters(current_iter=1)
    assert 'l_g_fm' not in model.log_dict
    assert 'l_g_gan' in model.log_dict, 'only the feature matching term is optional'


def test_the_generator_is_updated(tmp_path):
    model = RealESRGANCleanModel(_opt(tmp_path))
    _feed(model)
    before = [p.detach().clone() for p in model.net_g.parameters()]
    model.optimize_parameters(current_iter=1)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.net_g.parameters()))


def test_the_discriminator_is_updated(tmp_path):
    model = RealESRGANCleanModel(_opt(tmp_path))
    _feed(model)
    before = [p.detach().clone() for p in model.net_d.parameters()]
    model.optimize_parameters(current_iter=1)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.net_d.parameters()))


def test_the_ema_weights_follow_the_generator(tmp_path):
    """--bg_model loads params_ema, so the averaged copy has to be maintained during training."""
    model = RealESRGANCleanModel(_opt(tmp_path))
    assert hasattr(model, 'net_g_ema')
    before = [p.detach().clone() for p in model.net_g_ema.parameters()]
    _feed(model)
    model.optimize_parameters(current_iter=1)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.net_g_ema.parameters()))


def test_the_output_has_the_checkpoint_scale(tmp_path):
    model = RealESRGANCleanModel(_opt(tmp_path))
    _feed(model)
    model.optimize_parameters(current_iter=1)
    assert model.output.shape == (2, 3, SIZE * SCALE, SIZE * SCALE)


def test_saving_writes_both_networks(tmp_path):
    model = RealESRGANCleanModel(_opt(tmp_path))
    model.save(epoch=0, current_iter=3)
    written = {p.name for p in (tmp_path / 'models').iterdir()}
    assert 'net_g_3.pth' in written and 'net_d_3.pth' in written
    assert sorted(torch.load(tmp_path / 'models' / 'net_g_3.pth')) == ['params', 'params_ema']
