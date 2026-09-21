import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from torchvision.transforms.functional import normalize

from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.face_helper import FaceHelper

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def blend_restoration(restored, original, weight):
    """Blend a restored face back toward the face that went in.

    A generative restorer resynthesises even a good face, so on an input that is already clean this model scores
    12.4 dB below simply doing nothing, losing on 64 of 64 held-out faces. Training on a mixture that includes
    near-identity samples (`mild_prob`) recovers about 15% of that and the returns diminish sharply, so the rest
    has to be bought somewhere else. This is that dial, applied at inference and costing no training: at 0 the
    output is the input untouched, at 1 it is the restoration, and in between the trade is continuous. It is
    network interpolation (arXiv:1811.10515) taken to the image, the technique this branch already uses to blend
    the background upsampler's two checkpoints. Evidence for both is in `docs/training_stability.md`.

    The trade is real in both directions: lowering the weight protects a good input and weakens the restoration
    of a badly degraded one. There is no value that is best for every image.

    Args:
        restored (ndarray): the restored face, uint8 BGR.
        original (ndarray): the aligned input face it was restored from, same shape and dtype.
        weight (float): how much of the restoration to keep, in [0, 1].
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f'weight must be in [0, 1], got {weight}')
    if restored.shape != original.shape:
        raise ValueError(f'cannot blend shapes {restored.shape} and {original.shape}')
    if weight == 1.0:
        return restored
    if weight == 0.0:
        return original
    # addWeighted saturates and rounds; doing this in numpy would truncate and darken every blended pixel.
    return cv2.addWeighted(restored, weight, original, 1.0 - weight, 0.0)


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | RestoreFormer. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
        det_model (str): MediaPipe face detector. Option: blaze_face_short_range | blaze_face_full_range.
            Default: blaze_face_short_range.
    """

    def __init__(self,
                 model_path,
                 upscale=2,
                 arch='clean',
                 channel_multiplier=2,
                 bg_upsampler=None,
                 device=None,
                 det_model='blaze_face_short_range'):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()
        else:
            raise ValueError(f'Unsupported arch {arch}. Option: clean | RestoreFormer.')
        # initialize face helper (MediaPipe BlazeFace detection, Apache 2.0 model)
        self.face_helper = FaceHelper(
            upscale, face_size=512, det_model=det_model, model_rootpath=os.path.join(ROOT_DIR, 'gfpgan/weights'))

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'gfpgan/weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.75):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face keypoints for each face
            self.face_helper.get_face_landmarks(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                # weight is deliberately not passed to the network. Neither architecture here uses it: both
                # accept **kwargs and discard it, so the flag has never done anything. It is applied below, on
                # the image, where it can.
                output = self.gfpgan(cropped_face_t, return_rgb=False)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                restored_face = blend_restoration(restored_face.astype('uint8'), cropped_face, weight)
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine()
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
