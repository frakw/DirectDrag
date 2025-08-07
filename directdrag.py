import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL
from pipeline import DirectDragger

from torchvision.utils import save_image
from pytorch_lightning import seed_everything


def clamp_points(points, image_height, image_width, r2):
    """將拖曳點往內推，使其距離邊界至少 r2 像素，避免越界。"""
    clamped = []
    for y, x in points:
        y = max(r2, min(y, image_height - r2 - 1))
        x = max(r2, min(x, image_width - r2 - 1))
        clamped.append([y, x])
    return clamped

def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

def get_original_points(handle_points: List[torch.Tensor],
                        full_h: int,
                        full_w: int,
                        sup_res_w,
                        sup_res_h,
                        ) -> List[torch.Tensor]:
    """
    Convert local handle points and target points back to their original UI coordinates.

    Args:
        sup_res_h: Half original height of the UI canvas.
        sup_res_w: Half original width of the UI canvas.
        handle_points: List of handle points in local coordinates.
        full_h: Original height of the UI canvas.
        full_w: Original width of the UI canvas.

    Returns:
        original_handle_points: List of handle points in original UI coordinates.
    """
    original_handle_points = []

    for cur_point in handle_points:
        original_point = torch.round(
            torch.tensor([cur_point[1] * full_w / sup_res_w, cur_point[0] * full_h / sup_res_h]))
        original_handle_points.append(original_point)

    return original_handle_points


def run_directdrag(source_image,
                 image_with_clicks,
                 points,
                 inversion_strength,
                 lam,
                 latent_lr,
                 model_path,
                 vae_path,
                 lora_path,
                 drag_end_step,
                 track_per_step,
                 r1,
                 r2,
                 d,
                 max_drag_per_track,
                 max_track_no_change,
                 feature_idx=3,
                 result_save_path='',
                 return_intermediate_images=False,
                 drag_loss_threshold=0,
                 save_intermedia=False,
                 compare_mode=False,
                 once_drag=False,
                 enable_soft_mask=True,
                 enable_latent_warpage_function=True,
                 enable_readout_guided_feature_alignment=True,
                 soft_mask_sigma=30,
                 latent_warpage_function_ratio=0.15,
                 readout_guided_feature_alignment_multiplier=350
                 ):
    mask = None
    prompt = ""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    height, width = source_image.shape[:2]
    n_inference_step = 50
    guidance_scale = 1.0
    seed = 42

    points = clamp_points(points, height, width, r2)
    dragger = DirectDragger(device, model_path, prompt, height, width, inversion_strength, r1, r2, d,
                          drag_end_step, track_per_step, lam, latent_lr,
                          n_inference_step, guidance_scale, feature_idx, compare_mode, vae_path, lora_path, seed,
                          max_drag_per_track, drag_loss_threshold, once_drag, max_track_no_change,
                          enable_soft_mask,enable_latent_warpage_function,enable_readout_guided_feature_alignment,
                          soft_mask_sigma,latent_warpage_function_ratio,readout_guided_feature_alignment_multiplier)

    source_image = preprocess_image(source_image, device)

    gen_image, intermediate_features, new_points_handle, intermediate_images = \
        dragger.direct_drag(source_image, points,
                          mask,
                          return_intermediate_images=return_intermediate_images)

    new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)
    if save_intermedia:
        drag_image = [dragger.latent2image(i.cuda()) for i in intermediate_features]
        save_images_with_pillow(drag_image, base_filename='drag_image')

    gen_image = F.interpolate(gen_image, (height, width), mode='bilinear')

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)

    new_points = []
    for i in range(len(new_points_handle)):
        new_cur_handle_points = new_points_handle[i].numpy().tolist()
        new_cur_handle_points = [int(point) for point in new_cur_handle_points]
        new_points.append(new_cur_handle_points)
        new_points.append(points[i * 2 + 1])

    print(f'points {points}')
    print(f'new points {new_points}')

    if return_intermediate_images:
        os.makedirs(result_save_path, exist_ok=True)
        for i in range(len(intermediate_images)):
            intermediate_images[i] = F.interpolate(intermediate_images[i], (height, width), mode='bilinear')
            intermediate_images[i] = intermediate_images[i].cpu().permute(0, 2, 3, 1).numpy()[0]
            intermediate_images[i] = (intermediate_images[i] * 255).astype(np.uint8)

        for i in range(len(intermediate_images)):
            intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
            intermediate_images_path = os.path.join(result_save_path, f'output_image_{i}.png')
            cv2.imwrite(intermediate_images_path, intermediate_images[i])

    return out_image, new_points
