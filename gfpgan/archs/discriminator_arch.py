import math
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F


class ResDownBlock(nn.Module):
    """Residual block that halves the spatial size.

    Main path: conv3x3 -> LeakyReLU -> conv3x3 -> LeakyReLU -> 2x average pooling.
    Shortcut: 2x average pooling -> conv1x1. The two paths are summed and scaled by 1/sqrt(2).
    All operations support double backward, as required by the R1 penalty.
    """

    def __init__(self, in_channels, out_channels):
        super(ResDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.avg_pool2d(out, 2)
        skip = self.skip(F.avg_pool2d(x, 2))
        return (out + skip) / math.sqrt(2)


@ARCH_REGISTRY.register()
class StyleGAN2DiscriminatorClean(nn.Module):
    """Residual global discriminator implemented with standard PyTorch operations (no custom CUDA kernels).

    Args:
        out_size (int): Input image size, a power of 2 >= 8. Default: 512.
        channel_multiplier (int): Channel multiplier for resolutions >= 64. Default: 1.
        narrow (float): Channel scale for all resolutions. Default: 1.
        stddev_group (int): Group size of the minibatch standard deviation layer. Default: 4.
    """

    def __init__(self, out_size=512, channel_multiplier=1, narrow=1, stddev_group=4):
        super(StyleGAN2DiscriminatorClean, self).__init__()
        log_size = int(math.log(out_size, 2))
        if 2**log_size != out_size or out_size < 8:
            raise ValueError(f'out_size must be a power of 2 >= 8, but received {out_size}.')

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }
        channels = {k: min(v, int(512 * narrow)) for k, v in channels.items()}

        self.from_rgb = nn.Conv2d(3, channels[out_size], 1, 1, 0)
        self.blocks = nn.ModuleList()
        in_channels = channels[out_size]
        for i in range(log_size, 2, -1):
            out_channels = channels[2**(i - 1)]
            self.blocks.append(ResDownBlock(in_channels, out_channels))
            in_channels = out_channels

        self.stddev_group = stddev_group
        self.final_conv = nn.Conv2d(in_channels + 1, channels[4], 3, 1, 1)
        self.final_linear1 = nn.Linear(channels[4] * 4 * 4, channels[4])
        self.final_linear2 = nn.Linear(channels[4], 1)

    def minibatch_stddev(self, x):
        b, c, h, w = x.shape
        group = min(b, self.stddev_group)
        if b % group != 0:
            group = 1
        y = x.view(group, -1, c, h, w)
        y = torch.sqrt((y - y.mean(0, keepdim=True)).pow(2).mean(0) + 1e-8)
        y = y.mean(dim=[1, 2, 3], keepdim=True)  # (b / group, 1, 1, 1)
        y = y.repeat(group, 1, h, w)
        return torch.cat([x, y], dim=1)

    def forward(self, x, return_feats=False):
        """Forward function.

        Args:
            x (Tensor): Images with shape (b, 3, out_size, out_size).
            return_feats (bool): Whether to also return the output of each residual block. Default: False.

        Returns:
            Tensor | tuple: Logits with shape (b, 1), or (logits, list of block features) if return_feats.
        """
        out = F.leaky_relu(self.from_rgb(x), negative_slope=0.2)
        feats = []
        for block in self.blocks:
            out = block(out)
            if return_feats:
                feats.append(out)
        out = self.minibatch_stddev(out)
        out = F.leaky_relu(self.final_conv(out), negative_slope=0.2)
        out = F.leaky_relu(self.final_linear1(out.flatten(1)), negative_slope=0.2)
        out = self.final_linear2(out)
        if return_feats:
            return out, feats
        return out


@ARCH_REGISTRY.register()
class UNetDiscriminatorClean(nn.Module):
    """U-Net discriminator with spectral normalization, for the background super-resolution model.

    Same shape as the discriminator described in the Real-ESRGAN paper: three strided convolutions down, three
    bilinear upsamples back with skip connections, then two extra convolutions and a per-pixel logit map. The
    forward can also return the downsampling features, which the training model uses for feature matching in
    place of a VGG perceptual loss.

    Args:
        num_in_ch (int): Channel number of the input. Default: 3.
        num_feat (int): Channel number of the first layer. Default: 64.
        skip_connection (bool): Whether to add the skip connections. Default: True.
    """

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorClean, self).__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, return_feats=False):
        """Forward function.

        Args:
            x (Tensor): Images with shape (b, num_in_ch, h, w); h and w must be multiples of 8.
            return_feats (bool): Whether to also return the three downsampling features. Default: False.

        Returns:
            Tensor | tuple: Logit map with shape (b, 1, h, w), or (logits, list of features) if return_feats.
        """
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)

        up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(up), negative_slope=0.2)
        if self.skip_connection:
            x4 = x4 + x2
        up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(up), negative_slope=0.2)
        if self.skip_connection:
            x5 = x5 + x1
        up = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(up), negative_slope=0.2)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2)
        out = self.conv9(out)
        if return_feats:
            return out, [x1, x2, x3]
        return out


@ARCH_REGISTRY.register()
class FacialComponentDiscriminatorClean(nn.Module):
    """Small patch discriminator for facial components (eyes, mouth), with standard PyTorch operations.

    Args:
        num_feat (int): Channel number of the first layer. Default: 64.
    """

    def __init__(self, num_feat=64):
        super(FacialComponentDiscriminatorClean, self).__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.block1 = ResDownBlock(num_feat, num_feat * 2)
        self.block2 = ResDownBlock(num_feat * 2, num_feat * 4)
        self.conv_last = nn.Conv2d(num_feat * 4, 1, 3, 1, 1)

    def forward(self, x, return_feats=False, **kwargs):
        """Forward function.

        Args:
            x (Tensor): Component patches with shape (b, 3, h, w).
            return_feats (bool): Whether to return the features of the two residual blocks. Default: False.

        Returns:
            tuple: Patch logits with shape (b, 1, h / 4, w / 4), and a list of two features or None.
        """
        feat = F.leaky_relu(self.conv_first(x), negative_slope=0.2)
        feat1 = self.block1(feat)
        feat2 = self.block2(feat1)
        out = self.conv_last(feat2)
        if return_feats:
            return out, [feat1, feat2]
        return out, None
