from basicsr.archs.arch_util import default_init_weights
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SRVGGNetCompactClean(SRVGGNetCompact):
    """SRVGGNetCompact with the weight initialisation its depth requires.

    ``SRVGGNetCompact`` never initialises its convolutions, so they keep the PyTorch default for ``nn.Conv2d``,
    ``kaiming_uniform_(a=sqrt(5))``, whose standard deviation is ``sqrt(1 / (3 * fan_in))`` against the
    ``sqrt(2 / fan_in)`` that a ReLU-family activation needs. Per layer that is a factor of 2.45, which is
    harmless in a shallow network. This body is 34 convolutions deep with no residual connection inside it --
    the only shortcut is the nearest upsampling added to the final output -- so the factor compounds: the
    gradient reaching the first layer was measured at 2.5e-15 against 7.8e-3 at the last, a ratio of 3e12.

    Training in that state moves only the shortcut, so the network reproduces nearest upsampling and the pixel
    loss barely falls. Two runs of 20,000 and 13,000 iterations ended with outputs equal to nearest to within
    0.01 of 255; see ``docs/background_super_resolution.md``. Kaiming initialisation brings the ratio to 0.22.

    The initialisation belongs here rather than in the model because BasicSR loads ``pretrain_network_g`` in
    ``__init__`` and calls ``init_training_settings`` afterwards: initialising there would silently overwrite
    the phase 1 weights when the adversarial phase starts from them.
    """

    def __init__(self, **kwargs):
        super(SRVGGNetCompactClean, self).__init__(**kwargs)
        default_init_weights(self.body, scale=1.0)
