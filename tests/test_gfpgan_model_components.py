"""The facial component path, validation and saving of GFPGANModel.

`options/train_gfpgan_clean.yml` configures the three component discriminators, so this is the code every run in
`docs/training_stability.md` went through. test_gfpgan_model covers the global discriminator and feature
matching; this covers what the shipped config adds on top.

The component tests run at the real 512, because `get_roi_regions` computes `face_ratio` as
`int(out_size / 512)`: below 512 that is 0, which silently zeroes both the crops and the component losses. The
rest run at 32, where they are cheap.
"""
import pytest
import torch

from gfpgan.models.gfpgan_model import GFPGANModel

SMALL = 32


class _Dataset():
    opt = {'name': 'validation'}


class _Loader(list):
    """The smallest thing nondist_validation accepts: an iterable whose dataset has a name."""

    dataset = _Dataset()


def _opt(tmp_path, size=SMALL, components=True, comp_style_weight=0.0, remove_pyramid_loss=50000, metrics=True):
    for sub in ('models', 'training_states', 'visualization'):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    opt = {
        'name': 'UnitTest',
        'model_type': 'GFPGANModel',
        'num_gpu': 1 if torch.cuda.is_available() else 0,
        'is_train': True,
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'network_g': {
            'type': 'GFPGANv1Clean',
            'out_size': size,
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
            'out_size': size,
            'narrow': 0.25
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
                'milestones': [600000],
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
            'feature_matching_weight': 1.0,
            'r1_reg_weight': 10,
            'net_d_iters': 1,
            'net_d_init_iters': 0,
            'net_d_reg_every': 1,
            'pyramid_loss_weight': 1,
            'remove_pyramid_loss': remove_pyramid_loss
        },
        'val': {
            'val_freq': 1,
            'save_img': False
        },
        'logger': {
            'print_freq': 1
        }
    }
    if metrics:
        opt['val']['metrics'] = {'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False}}
    if components:
        for key in ('network_d_left_eye', 'network_d_right_eye', 'network_d_mouth'):
            opt[key] = {'type': 'FacialComponentDiscriminatorClean'}
        opt['train']['optim_component'] = {'type': 'Adam', 'lr': 2e-3}
        opt['train']['gan_component_opt'] = {
            'type': 'GANLoss',
            'gan_type': 'vanilla',
            'real_label_val': 1.0,
            'fake_label_val': 0.0,
            'loss_weight': 1.0
        }
        opt['train']['comp_style_weight'] = comp_style_weight
    return opt


def _data(size=SMALL, batch=2, with_locations=True):
    data = {'lq': torch.rand((batch, 3, size, size)) * 2 - 1, 'gt': torch.rand((batch, 3, size, size)) * 2 - 1}
    if with_locations:
        # boxes in the observer's view, as gfpgan/component_boxes.py writes them for a 512 frame
        scale = size / 512
        data['loc_left_eye'] = torch.tensor([[150.0, 230.0, 240.0, 320.0]] * batch) * scale
        data['loc_right_eye'] = torch.tensor([[280.0, 230.0, 370.0, 320.0]] * batch) * scale
        data['loc_mouth'] = torch.tensor([[200.0, 330.0, 320.0, 450.0]] * batch) * scale
    return data


def _val_loader(size=SMALL):
    return _Loader([{
        'lq': torch.rand((1, 3, size, size)) * 2 - 1,
        'gt': torch.rand((1, 3, size, size)) * 2 - 1,
        'lq_path': ['face.png']
    }])


@pytest.fixture(scope='module')
def model512(tmp_path_factory):
    """One 512 model for the component tests, since building it costs more than a test does."""
    return GFPGANModel(_opt(tmp_path_factory.mktemp('components'), size=512))


def test_the_component_discriminators_are_built_and_optimised(tmp_path):
    model = GFPGANModel(_opt(tmp_path))
    assert model.use_facial_disc
    # one optimizer for the generator, one for the global discriminator, three for the components
    assert len(model.optimizers) == 5


def test_component_losses_are_logged_for_all_three_regions(model512):
    model512.feed_data(_data(size=512, batch=1))
    model512.optimize_parameters(current_iter=1)
    for region in ('left_eye', 'right_eye', 'mouth'):
        assert f'l_g_gan_{region}' in model512.log_dict, region
        assert f'l_d_{region}' in model512.log_dict, region
        assert torch.isfinite(torch.tensor(model512.log_dict[f'l_d_{region}'])), region


def test_the_regions_are_cropped_to_the_sizes_the_discriminators_expect(model512):
    model512.feed_data(_data(size=512, batch=1))
    model512.output = model512.gt
    model512.get_roi_regions(eye_out_size=80, mouth_out_size=120)
    assert model512.left_eyes.shape == (1, 3, 80, 80)
    assert model512.right_eyes.shape == (1, 3, 80, 80)
    assert model512.mouths.shape == (1, 3, 120, 120)
    assert model512.left_eyes_gt.shape == model512.left_eyes.shape


def test_the_gram_style_loss_is_off_at_weight_zero(model512):
    """The shipped config sets comp_style_weight to 0, because larger component steps destabilised training."""
    model512.feed_data(_data(size=512, batch=1))
    model512.optimize_parameters(current_iter=1)
    assert 'l_g_comp_style_loss' not in model512.log_dict


def test_the_gram_style_loss_is_computed_when_asked_for(model512):
    """The weight is read from the options on every iteration, so turning it on needs no new model."""
    model512.opt['train']['comp_style_weight'] = 200.0
    try:
        model512.feed_data(_data(size=512, batch=1))
        model512.optimize_parameters(current_iter=1)
        # one key for the three regions together, not one per region
        assert 'l_g_comp_style_loss' in model512.log_dict
        assert torch.isfinite(torch.tensor(model512.log_dict['l_g_comp_style_loss']))
    finally:
        model512.opt['train']['comp_style_weight'] = 0.0


def test_the_gram_matrix_is_square_in_the_channel_dimension(model512):
    gram = model512._gram_mat(torch.rand((2, 8, 16, 16), device=model512.device))
    assert gram.shape == (2, 8, 8)


def test_the_pyramid_loss_stops_contributing_but_keeps_its_layers_in_the_graph(tmp_path):
    """Past the threshold the weight is zero, so the terms leave the log while the layers keep their gradients.

    The generator's `toRGB` layers exist only to produce the pyramid, so they receive a gradient from nothing
    else. Dropping the loss outright would leave them unused, which DistributedDataParallel reports as an error;
    computing it at weight zero keeps them in the graph. The threshold is compared against the iteration, which
    is why a finetune restarting at 0 puts the full weight back unless the option is set to 0 as well.
    """
    model = GFPGANModel(_opt(tmp_path, components=False, remove_pyramid_loss=5))
    model.feed_data(_data(with_locations=False))
    # one loss per pyramid level, named after the level's resolution
    levels = ['l_p_8', 'l_p_16', 'l_p_32']
    model.optimize_parameters(current_iter=1)
    assert [k for k in model.log_dict if k.startswith('l_p_')] == levels

    model.optimize_parameters(current_iter=6)
    assert [k for k in model.log_dict if k.startswith('l_p_')] == [], 'the log shows the loss as off'
    assert 'l_g_pix' in model.log_dict, 'the pixel loss is not what gets switched off'
    to_rgb_grads = [p.grad for name, p in model.net_g.named_parameters() if name.startswith('toRGB')]
    assert to_rgb_grads and all(g is not None for g in to_rgb_grads), 'the layers must stay in the graph'
    assert all(float(g.abs().max()) == 0.0 for g in to_rgb_grads), 'and receive exactly nothing'


def test_the_crop_sizes_follow_the_generator_size(tmp_path):
    """Upstream scaled by `int(out_size / 512)`, which is 0 below 512: blank crops and meaningless losses.

    The component discriminators cannot consume crops this small, which is why the tests above run at 512. What
    is checked here is that the geometry is right rather than zero.
    """
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.feed_data(_data())
    model.output = model.gt
    model.get_roi_regions(eye_out_size=80, mouth_out_size=120)
    assert model.left_eyes.shape == (2, 3, 5, 5), 'round(80 * 32 / 512)'
    assert model.mouths.shape == (2, 3, 8, 8), 'round(120 * 32 / 512)'
    assert float(model.left_eyes.abs().max()) > 0, 'the crop must not be scaled away'
    assert float(model.mouths.abs().max()) <= float(model.gt.abs().max()), 'nor scaled up'


def test_a_crop_size_never_rounds_down_to_zero(tmp_path):
    """At out_size 32 the mouth would round to 7.5; the floor of one pixel keeps roi_align valid at any size."""
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.feed_data(_data())
    model.output = model.gt
    model.get_roi_regions(eye_out_size=1, mouth_out_size=1)
    assert model.left_eyes.shape[-1] == 1
    assert model.mouths.shape[-1] == 1


def test_validation_reports_the_configured_metric(tmp_path):
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.nondist_validation(_val_loader(), current_iter=1, tb_logger=None, save_img=False)
    assert 'psnr' in model.metric_results
    assert torch.isfinite(torch.tensor(model.metric_results['psnr']))


def test_validation_can_save_its_images(tmp_path):
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.nondist_validation(_val_loader(), current_iter=1, tb_logger=None, save_img=True)
    assert list((tmp_path / 'visualization').rglob('*.png')), 'save_img must leave something to look at'


def test_validation_without_metrics_still_runs(tmp_path):
    model = GFPGANModel(_opt(tmp_path, components=False, metrics=False))
    model.nondist_validation(_val_loader(), current_iter=1, tb_logger=None, save_img=False)


def test_dist_validation_delegates_on_rank_zero(tmp_path):
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.dist_validation(_val_loader(), current_iter=1, tb_logger=None, save_img=False)
    assert 'psnr' in model.metric_results


def test_test_uses_the_ema_weights(tmp_path):
    """Inference loads params_ema, so validation has to judge the same weights."""
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.feed_data(_data(with_locations=False))
    assert hasattr(model, 'net_g_ema')
    model.test()
    assert model.output.shape == (2, 3, SMALL, SMALL)


def test_save_writes_every_network_and_the_training_state(tmp_path):
    model = GFPGANModel(_opt(tmp_path))
    model.save(epoch=0, current_iter=7)
    written = {p.name for p in (tmp_path / 'models').iterdir()}
    assert 'net_g_7.pth' in written
    assert 'net_d_7.pth' in written
    for region in ('left_eye', 'right_eye', 'mouth'):
        assert f'net_d_{region}_7.pth' in written, region
    assert (tmp_path / 'training_states' / '7.state').is_file()


def test_the_generator_checkpoint_carries_both_parameter_sets(tmp_path):
    """params for resuming, params_ema for inference: gfpgan/utils.py prefers the second."""
    model = GFPGANModel(_opt(tmp_path, components=False))
    model.save(epoch=0, current_iter=7)
    assert sorted(torch.load(tmp_path / 'models' / 'net_g_7.pth')) == ['params', 'params_ema']


def test_a_model_without_component_discriminators_skips_them(tmp_path):
    model = GFPGANModel(_opt(tmp_path, components=False))
    assert not model.use_facial_disc
    assert len(model.optimizers) == 2
    model.feed_data(_data(with_locations=False))
    model.optimize_parameters(current_iter=1)
    assert 'l_g_gan_left_eye' not in model.log_dict
