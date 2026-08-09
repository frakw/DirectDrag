# *************************************************************************
# Live single-pair evaluation helpers for the Gradio "Evaluate" button:
#   - IF  (1 - LPIPS)   — image fidelity / how well the background is preserved
#   - MD  (Mean Distance) — drag accuracy, via DIFT feature matching
# GScore lives in gscore.py at the project root (LLM-as-judge, separate concern).
#
# Both metrics load fairly heavy models the first time they're used (LPIPS+AlexNet
# for IF; a whole extra Stable Diffusion 2.1 model for MD's DIFT features, on top of
# whatever base model DirectDrag itself is already running). They're loaded lazily
# and cached as module-level singletons so repeated Evaluate clicks don't reload them,
# but the first click after starting the app (or after checking IF/MD for the first
# time) will be slow and use extra VRAM.
# *************************************************************************

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars can still be set another way.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.transforms import PILToTensor

# Override with a local path or a different repo id via this env var if
# "stabilityai/stable-diffusion-2-1" isn't reachable (e.g. no internet access to
# huggingface.co, or you've pre-downloaded it somewhere else / are behind a
# firewall). Set it in your .env file, e.g.:
#   DIRECTDRAG_DIFT_MODEL=/local/path/to/stable-diffusion-2-1
DIFT_MODEL_ID = os.environ.get("DIRECTDRAG_DIFT_MODEL", "stabilityai/stable-diffusion-2-1")

_lpips_model = None
_dift_featurizer = None


def _get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex').to(_get_device())
    return _lpips_model


def compute_if(original_image: np.ndarray, result_image: np.ndarray) -> float:
    """
    IF = 1 - LPIPS(original, result). Both images are numpy arrays (H, W, 3), uint8.
    Result is resized to match the original's dimensions if they differ (e.g. because
    the pipeline downscaled the working resolution).
    """
    device = _get_device()

    def to_tensor(img):
        t = torch.from_numpy(img).float() / 127.5 - 1
        t = rearrange(t, "h w c -> 1 c h w")
        return t.to(device)

    ori_tensor = to_tensor(original_image)
    res_tensor = to_tensor(result_image)
    if res_tensor.shape[-2:] != ori_tensor.shape[-2:]:
        res_tensor = F.interpolate(res_tensor, size=ori_tensor.shape[-2:], mode='bilinear')

    ori_224 = F.interpolate(ori_tensor, (224, 224), mode='bilinear')
    res_224 = F.interpolate(res_tensor, (224, 224), mode='bilinear')

    model = _get_lpips_model()
    with torch.no_grad():
        lp = model(ori_224, res_224)
    return float(1 - lp.item())


def _get_dift_featurizer():
    """
    Lazily loads the DIFT feature extractor (a separate Stable Diffusion 2.1 model,
    per DirectDrag's own run_eval_IF_MD.py) used to locate where each handle point
    ended up in the result image.
    """
    global _dift_featurizer
    if _dift_featurizer is None:
        from run_eval_IF_MD import SDFeaturizer
        try:
            _dift_featurizer = SDFeaturizer(DIFT_MODEL_ID)
        except Exception as e:
            raise RuntimeError(
                f"Could not load the DIFT model '{DIFT_MODEL_ID}' for MD. This is "
                f"usually a network issue (this machine can't reach huggingface.co) "
                f"or an offline/HF_HUB_OFFLINE environment. If you have this model "
                f"pre-downloaded locally, set DIRECTDRAG_DIFT_MODEL in your .env to "
                f"its local folder path. Original error: {e}"
            ) from e
    return _dift_featurizer


def compute_md(original_image: np.ndarray, result_image: np.ndarray, points, prompt: str = ""):
    """
    MD = mean Euclidean distance between each pair's target point and where the
    handle point's feature actually ended up in the result image (found via DIFT
    cosine-similarity matching, same approach as run_eval_IF_MD.py).

    `points` is the same [[x1,y1],[x2,y2],...] list used throughout the app
    (alternating handle/target pairs, in the *original_image*'s pixel coordinates).
    `prompt` defaults to "" since DirectDrag is prompt-free.

    Returns None if there are no complete point pairs to measure.
    """
    if not points or len(points) < 2:
        return None

    device = _get_device()
    featurizer = _get_dift_featurizer()

    ori_pil = Image.fromarray(original_image)
    res_pil = Image.fromarray(result_image).resize(ori_pil.size, Image.BILINEAR)

    ori_tensor = (PILToTensor()(ori_pil) / 255.0 - 0.5) * 2
    res_tensor = (PILToTensor()(res_pil) / 255.0 - 0.5) * 2
    _, height, width = ori_tensor.shape

    ft_ori = featurizer.forward(ori_tensor, prompt, t=261, up_ft_index=1, ensemble_size=8)
    ft_ori = F.interpolate(ft_ori, (height, width), mode='bilinear')
    ft_res = featurizer.forward(res_tensor, prompt, t=261, up_ft_index=1, ensemble_size=8)
    ft_res = F.interpolate(ft_res, (height, width), mode='bilinear')

    cos = nn.CosineSimilarity(dim=1)
    handle_points = [points[i] for i in range(0, len(points) - 1, 2)]
    target_points = [points[i] for i in range(1, len(points), 2)]

    distances = []
    num_channel = ft_ori.size(1)
    for (hx, hy), (tx, ty) in zip(handle_points, target_points):
        hx, hy = int(hx), int(hy)
        tx, ty = int(tx), int(ty)
        hy = min(max(hy, 0), height - 1)
        hx = min(max(hx, 0), width - 1)

        src_vec = ft_ori[0, :, hy, hx].view(1, num_channel, 1, 1)
        cos_map = cos(src_vec, ft_res).detach().cpu().numpy()[0]
        max_row, max_col = np.unravel_index(cos_map.argmax(), cos_map.shape)

        dist = ((tx - max_col) ** 2 + (ty - max_row) ** 2) ** 0.5
        distances.append(dist)

    if not distances:
        return None
    return float(sum(distances) / len(distances))