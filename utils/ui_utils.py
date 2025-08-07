# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************


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

from .lora_utils import train_lora


# -------------- general UI functionality --------------
def clear_all(length=512):
    return gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           [], None, None


def mask_image(image,
               mask,
               color=[255, 0, 0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0, out)
    return out


def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


def show_cur_points(img,
                    sel_pix,
                    bgr=False):
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            red = (255, 0, 0) if not bgr else (0, 0, 255)
            cv2.circle(img, tuple(point), 5, red, -1)
        else:
            # draw a blue circle at the handle point
            blue = (0, 0, 255) if not bgr else (255, 0, 0)
            cv2.circle(img, tuple(point), 5, blue, -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


# clear all handle/target points
def undo_point(original_image, selected_points):
    selected_points = selected_points[:-1]
    img = original_image.copy()
    points = []
    for idx, point in enumerate(selected_points):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img, selected_points

def undo_pair(original_image, selected_points):
    if len(selected_points) % 2 == 0:
        selected_points = selected_points[:-2]
    else:
        selected_points = selected_points[:-1]
    img = original_image.copy()
    points = []
    for idx, point in enumerate(selected_points):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img, selected_points



def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file or a directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents


def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress(),
                         use_gradio_progress=True):
    if not os.path.exists(lora_path):
        os.makedirs(lora_path)

    clear_folder(lora_path)

    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress,
        use_gradio_progress)
    return "Training LoRA Done!"



def save_images_with_pillow(images, base_filename='image'):
    for index, img in enumerate(images):
        # Convert array to Image object and save
        img_pil = Image.fromarray(img)
        folder_path = f'./save'
        filename = os.path.join(folder_path, "{}_{}.png".format(base_filename, index))
        img_pil.save(filename)
        print(f"Saved: {filename}")


def save_image_mask_points(mask, points, image_with_points, output_dir='./saved_data'):
    """
    Saves the mask and points to the specified directory.

    Args:
      mask: The mask data as a numpy array.
      points: The list of points collected from the user interaction.
      image_with_points: The image with points clicked by the user.
      output_dir: The directory where to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save mask
    mask_path = os.path.join(output_dir, f"mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    # Save points
    points_path = os.path.join(output_dir, f"points.json")
    with open(points_path, 'w') as f:
        json.dump({'points': points}, f)

    image_with_points_path = os.path.join(output_dir, "image_with_points.jpg")
    Image.fromarray(image_with_points).save(image_with_points_path)

    return


def save_drag_result(output_image, new_points, result_path):
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    result_dir = f'{result_path}'
    os.makedirs(result_dir, exist_ok=True)
    output_image_path = os.path.join(result_dir, 'output_image.png')
    cv2.imwrite(output_image_path, output_image)

    img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    cv2.imwrite(new_points_image_path, img_with_new_points)

    points_path = os.path.join(result_dir, f'new_points.json')
    with open(points_path, 'w') as f:
        json.dump({'points': new_points}, f)


def save_intermediate_images(intermediate_images, result_dir):
    for i in range(len(intermediate_images)):
        intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
        intermediate_images_path = os.path.join(result_dir, f'output_image_{i}.png')
        cv2.imwrite(intermediate_images_path, intermediate_images[i])


def create_video(image_folder, data_folder, fps=2, first_frame_duration=2, last_frame_extra_duration=2):
    """
    Creates an MP4 video from a sequence of images using OpenCV.
    """
    img_folder = Path(image_folder)
    img_num = len(list(img_folder.glob('*.png')))

    # Path to the original image with points
    data_folder = Path(data_folder)
    original_path = data_folder / 'image_with_points.jpg'
    output_path = img_folder / 'dragging.mp4'
    # Collect all image paths
    img_files = [original_path]

    # Load the first image to determine the size
    frame = cv2.imread(str(img_files[0]))
    height, width, layers = frame.shape
    size = (int(width), int(height))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 format
    video = cv2.VideoWriter(str(output_path), fourcc, int(fps), size)

    for _ in range(int(fps * first_frame_duration)):
        video.write(frame)

    # Add images to video
    for i in range(img_num - 2):
        video.write(cv2.imread(str(img_folder / f'output_image_{i}.png')))

    last_frame = cv2.imread(str(img_folder / 'output_image.png'))
    for _ in range(int(fps * last_frame_extra_duration)):
        video.write(last_frame)

    video.release()