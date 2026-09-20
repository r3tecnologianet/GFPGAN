"""Background super-resolution for the inference pipeline.

`GFPGANer` upscales the background with whatever object is passed as `bg_upsampler`, and falls back to a Lanczos
resize when given none. Lanczos is the default because the Real-ESRGAN release weights are trained on
non-commercial data; this module lets a user supply super-resolution weights they are licensed to use.

No weights are downloaded and no path is built in: the model path is supplied by the caller, exactly as
`--model_path` already works for the face model.
"""
import cv2
import numpy as np
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact


def build_from_state(state_dict):
    """Build the network the checkpoint describes, including its scale.

    Reading the shape rather than trusting an argument means the scale cannot disagree with the weights.
    ``SRVGGNetCompact`` stores a flat ``body.N`` list whose last convolution emits ``num_out_ch * scale**2``
    channels for the pixel shuffle, so both the depth and the scale are recoverable.
    """
    indices = [int(k.split('.')[1]) for k in state_dict if k.startswith('body.') and k.split('.')[1].isdigit()]
    if not indices or 'body.0.weight' not in state_dict:
        raise ValueError('not an SRVGGNetCompact checkpoint: no body.N convolutions found')
    last = max(indices)
    num_feat = state_dict['body.0.weight'].shape[0]
    num_conv = (last - 2) // 2
    out_channels = state_dict[f'body.{last}.weight'].shape[0]
    scale = int(round((out_channels / 3)**0.5))
    if scale < 1 or 3 * scale * scale != out_channels:
        raise ValueError(f'cannot infer an integer scale from {out_channels} output channels')
    net = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=scale, act_type='prelu')
    return net, scale


class SRBackgroundUpsampler:
    """Adapts a super-resolution checkpoint to the interface `GFPGANer` expects of a background upsampler.

    `GFPGANer` calls ``enhance(img, outscale=...)`` and uses the first element of the result, so this returns
    ``(image, None)`` to match.
    """

    def __init__(self, model_path, device=None, param_key=None):
        state = torch.load(model_path, map_location='cpu')
        if param_key is not None:
            state = state[param_key]
        else:
            # Trained checkpoints carry both; params_ema is what inference should use.
            for key in ('params_ema', 'params'):
                if isinstance(state, dict) and key in state:
                    state = state[key]
                    break
        self.net, self.scale = build_from_state(state)
        self.net.load_state_dict(state, strict=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.net = self.net.eval().to(self.device)

    @torch.no_grad()
    def enhance(self, img, outscale=None):
        """Upscale a BGR uint8 image by the checkpoint's own scale, then resize if another one is asked for."""
        height, width = img.shape[:2]
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.).unsqueeze(0)
        out = self.net(x.to(self.device)).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        out = (out * 255.).round().astype(np.uint8)
        if outscale is not None and float(outscale) != float(self.scale):
            out = cv2.resize(out, (int(width * outscale), int(height * outscale)), interpolation=cv2.INTER_LANCZOS4)
        return out, None
