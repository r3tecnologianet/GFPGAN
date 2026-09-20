import torch
from basicsr.models.realesrgan_model import RealESRGANModel
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F


@MODEL_REGISTRY.register()
class RealESRGANCleanModel(RealESRGANModel):
    """Real-ESRGAN training without components that carry commercial-use restrictions.

    The synthetic degradation pipeline is inherited from ``RealESRGANModel``: ``feed_data`` builds the low quality
    input on the GPU from blur kernels, resizing, noise, JPEG compression and sinc filtering, twice in a row.

    What changes here is how the generator is guided. Upstream adds a VGG19/ImageNet perceptual loss, whose weights
    are not licensed for commercial use. This model replaces it with discriminator feature matching
    (``feature_matching_weight``), the same substitution used by ``GFPGANModel`` in this repository: the L1
    distance between the discriminator features of the output and of the ground truth. The discriminator must
    therefore accept ``return_feats`` (see ``UNetDiscriminatorClean``).

    ``perceptual_opt`` is ignored even if a config provides it, so that a stray option cannot silently pull an
    ImageNet network into training.
    """

    # Degradation options for a near-identity sample: no noise, no second blur, near-lossless JPEG and no
    # rescaling beyond the downscale to the target size. Ranges keep a minimum width because basicsr asserts
    # strict inequality on several of them.
    MILD_DEGRADATION = {
        'resize_prob': [0, 0, 1],
        'resize_range': [1, 1.01],
        'gaussian_noise_prob': 0.0,
        'noise_range': [0, 0.01],
        'poisson_scale_range': [0, 0.01],
        'gray_noise_prob': 0.0,
        'jpeg_range': [94, 95],
        'second_blur_prob': 0.0,
        'resize_prob2': [0, 0, 1],
        'resize_range2': [1, 1.01],
        'gaussian_noise_prob2': 0.0,
        'noise_range2': [0, 0.01],
        'poisson_scale_range2': [0, 0.01],
        'gray_noise_prob2': 0.0,
        'jpeg_range2': [94, 95],
    }

    @staticmethod
    def _delta_kernels(kernel):
        """A kernel that filter2D leaves unchanged, so the mild branch really is unblurred.

        The blur and sinc kernels are drawn by the dataset, not read from the options, so swapping the option
        dictionary alone would still blur the mild sample and the branch would measure nothing.
        """
        delta = torch.zeros_like(kernel)
        centre = kernel.shape[-1] // 2
        delta[..., centre, centre] = 1.0
        return delta

    def feed_data(self, data):
        """Build the low quality input, drawing a near-identity degradation with probability ``mild_prob``.

        Training only ever shows this model heavily degraded inputs, so it learns to treat fine texture as noise
        and erases it when the input is already good: measured on a near-identity validation set, it falls 1.06 dB
        of PSNR and 0.042 of SSIM below plain Lanczos, having been above it on the usual set. Mixing in samples
        that need almost no restoration states the missing requirement, that a good input should survive.

        ``mild_prob`` defaults to 0, which reproduces the inherited pipeline exactly.
        """
        mild_prob = self.opt.get('mild_prob', 0)
        if not self.is_train or mild_prob <= 0 or torch.rand(1).item() >= mild_prob:
            super(RealESRGANCleanModel, self).feed_data(data)
            return

        data = dict(data)
        for key in ('kernel1', 'kernel2', 'sinc_kernel'):
            if key in data:
                data[key] = self._delta_kernels(data[key])
        saved = {key: self.opt[key] for key in self.MILD_DEGRADATION if key in self.opt}
        self.opt.update(self.MILD_DEGRADATION)
        try:
            super(RealESRGANCleanModel, self).feed_data(data)
        finally:
            self.opt.update(saved)

    def init_training_settings(self):
        super(RealESRGANCleanModel, self).init_training_settings()
        if self.opt.get('mild_prob', 0) > 0:
            missing = [key for key in self.MILD_DEGRADATION if key not in self.opt]
            assert not missing, f'mild_prob is set but these degradation options are absent: {missing}'
        self.feature_matching_weight = self.opt['train'].get('feature_matching_weight', 0)
        self.cri_perceptual = None

    def optimize_parameters(self, current_iter):
        # the ground truth used by each loss; the sharpened copy is the upstream default
        l1_gt = self.gt_usm if self.opt.get('l1_gt_usm', True) else self.gt
        gan_gt = self.gt_usm if self.opt.get('gan_gt_usm', True) else self.gt

        # ----------- optimize net_g ----------- #
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total = l_g_total + l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.feature_matching_weight > 0:
                fake_pred, fake_feats = self.net_d(self.output, return_feats=True)
                with torch.no_grad():
                    _, real_feats = self.net_d(gan_gt, return_feats=True)
                l_g_fm = sum(F.l1_loss(fake, real.detach()) for fake, real in zip(fake_feats, real_feats))
                l_g_fm = l_g_fm * self.feature_matching_weight
                l_g_total = l_g_total + l_g_fm
                loss_dict['l_g_fm'] = l_g_fm
            else:
                fake_pred = self.net_d(self.output)

            l_g_gan = self.cri_gan(fake_pred, True, is_disc=False)
            l_g_total = l_g_total + l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # ----------- optimize net_d ----------- #
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        fake_d_pred = self.net_d(self.output.detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if getattr(self, 'ema_decay', 0) > 0:
            self.model_ema(decay=self.ema_decay)
