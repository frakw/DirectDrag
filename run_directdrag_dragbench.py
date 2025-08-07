# GoodDrag: https://github.com/zewei-Zhang/GoodDrag
# DragBench: https://github.com/Yujun-Shi/DragDiffusion/releases/tag/v0.1.1
# Usage: python dragbench_goodrag.py [dataset_folder]
# Note: you should use extract_drag_bench.py first, link: https://gist.github.com/frakw/4a259ece6e8a506057ebbbddc2ad5a73
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
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from utils.ui_utils import train_lora_interface, show_cur_points, create_video
from directdrag import run_directdrag
import time
import torch
import random

use_existing_lora = True  # Set to True if you want to use existing LoRA files, otherwise it will train a new one.

use_prompt = False

use_dragtext = False
use_dragtext_mask = False
use_dragtext_lora = False

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 如果用 GPU
    torch.cuda.manual_seed_all(seed)  # 多 GPU 時

    torch.backends.cudnn.deterministic = True  # 設定為 deterministic mode
    torch.backends.cudnn.benchmark = False     # 關閉 auto-tuning（不然會影響重現性）

    os.environ["PYTHONHASHSEED"] = str(seed)

def benchmark_dataset_dragbench(dataset_folder):
    dataset_path = Path(dataset_folder)
    categories = os.listdir(dataset_path)
    start_time = time.time()
    drag_total_time = 0
    train_lora_total_time = 0
    sample_count = 1
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue
        samples = os.listdir(category_path)
        for sample in samples:
            sample_path = Path(os.path.join(category_path, sample))
            if os.path.isfile(sample_path):
                continue
            print(f'Benchmarking {sample_path}...')
            print(f'Sample Count {sample_count}...')
            sample_count += 1
            original_image, mask, points, image_with_points, prompt = load_data_dragbench(sample_path)
            train_lora_start_time = time.time()
            train_lora_one_image_dragbench(sample_path,original_image, "")
            train_lora_end_time = time.time()
            train_lora_total_time += (train_lora_end_time - train_lora_start_time)
            drag_start_time = time.time()
            drag_one_image_dragbench(sample_path,original_image, mask, points, image_with_points, prompt)
            drag_end_time = time.time()
            drag_total_time += (drag_end_time - drag_start_time)
            #try:
            #    bench_one_image_dragbench(sample_path)
            #except Exception as e:
            #    print(f'An error occured while benchmarking {sample_path}: {e}.')
    end_time = time.time()
    print(f"***************\n"*2)
    print(f"use time sum: {end_time-start_time}")
    print(f"drag per sample: {drag_total_time/205}")
    print(f"train lora per sample: {train_lora_total_time/205}")
    print(f"***************\n"*2)
    logg = f"***************\n"*2 + \
            f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}\n" +\
            f"{'./dragbench_result'}:  \n" + \
            f"use time sum: {end_time-start_time} \n" + \
            f"drag per sample: {drag_total_time/205}\n" +\
            f"train lora per sample: {train_lora_total_time/205}\n\n\n"
    with open("./run_drag_result.txt", 'a') as f:
        f.write(logg)

def load_data_dragbench(folder):
    """Load the original image, mask, and points from the specified folder."""
    folder_path = Path(folder)

    # Load original image
    original_image_path = folder_path / 'original_image.png'
    original_image = Image.open(original_image_path)
    original_image = np.array(original_image)

    # Load mask
    mask_path = folder_path / 'mask.png'
    mask = Image.open(mask_path)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Load json
    json_path = folder_path / 'drag_instruction.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        points = json_data['points']
        prompt = json_data['prompt']

    image_points_path = folder_path / 'user_drag.png'
    image_with_points = Image.open(image_points_path)
    image_with_points = np.array(image_with_points)

    return original_image, mask, points, image_with_points, prompt

def train_lora_one_image_dragbench(folder,original_image,prompt):
    if use_prompt:
        lora_path = f'./prompt_dragbench_lora_data/{folder.parts[-2]}/{folder.parts[-1]}'
    else:
        lora_path = f'./no_prompt_dragbench_lora_data/{folder.parts[-2]}/{folder.parts[-1]}'
    lora_file_path = os.path.join(lora_path, 'pytorch_lora_weights.safetensors')
    model_path = 'runwayml/stable-diffusion-v1-5'
    if use_existing_lora and os.path.exists(lora_file_path):
        print(f'Using existing LoRA at {lora_file_path}.')
    else:
        print(f'Training Lora.')
        train_lora_interface(original_image=original_image, prompt=prompt, model_path=model_path,
                            vae_path='stabilityai/sd-vae-ft-mse',
                            lora_path=lora_path, lora_step=70, lora_lr=0.0005, lora_batch_size=4, lora_rank=16,
                            use_gradio_progress=False,
                            use_dragtext_lora=use_dragtext_lora,)
        print(f'Training Lora Done! Begin dragging.')

def drag_one_image_dragbench(folder, original_image, mask, points, image_with_points, prompt):
    """
    Test the saved data by running the drag model.

    Args:
      folder: The folder where the original image, mask, and points are saved.
    """
    
    model_path = 'runwayml/stable-diffusion-v1-5'

    if use_prompt:
        lora_path = f'./prompt_dragbench_lora_data/{folder.parts[-2]}/{folder.parts[-1]}'
    else:
        lora_path = f'./no_prompt_dragbench_lora_data/{folder.parts[-2]}/{folder.parts[-1]}'

    return_intermediate_images = False

    result_dir = f'./dragbench_result/{Path(folder).parts[-2]}/{Path(folder).parts[-1]}'
    os.makedirs(result_dir, exist_ok=True)

    print(prompt)
    if use_prompt == False:
        prompt = ''
    output_image, new_points = run_directdrag(
        source_image=original_image,
        image_with_clicks=image_with_points,
        #mask=None,
        #prompt=prompt,
        points=points,
        inversion_strength=0.75,
        lam=0.1,
        latent_lr=0.02,
        model_path=model_path,
        vae_path='stabilityai/sd-vae-ft-mse',
        lora_path=lora_path,
        drag_end_step=7,
        track_per_step=10,
        save_intermedia=False,
        compare_mode=False,
        r1=4,
        r2=12,
        d=4,
        max_drag_per_track=3,
        drag_loss_threshold=0,
        once_drag=False,
        max_track_no_change=5,
        result_save_path=result_dir,
        return_intermediate_images=return_intermediate_images,
        enable_soft_mask=True,
        enable_readout_guided_feature_alignment=True,
        enable_latent_warpage_function=True,
        soft_mask_sigma=30,
        readout_guided_feature_alignment_multiplier=350,
        latent_warpage_function_ratio=0.15
    )

    print(f'Drag finished!')
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image_path = os.path.join(result_dir, 'dragged_image.png')
    cv2.imwrite(output_image_path, output_image)

    img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    cv2.imwrite(new_points_image_path, img_with_new_points)

    image_with_points_path = os.path.join(result_dir, 'image_with_points.png')
    image_with_points = Image.fromarray(image_with_points)
    image_with_points.save(image_with_points_path)

    points_path = os.path.join(result_dir, f'new_points.json')
    with open(points_path, 'w') as f:
        json.dump({'points': new_points}, f)

    #if return_intermediate_images:
    #    create_video(result_dir, folder)


def main(argv):
    dataset_folder = argv[1]
    if len(argv) > 2:
        if argv[2] == '--single':
            original_image, mask, points, image_with_points, prompt = load_data_dragbench(dataset_folder)
            drag_one_image_dragbench(Path(dataset_folder),original_image, mask, points, image_with_points, prompt)
            return
    if dataset_folder == '':
        benchmark_dataset_dragbench('drag_bench_data')
    else:
        benchmark_dataset_dragbench(dataset_folder)


if __name__ == '__main__':
    fix_seed(42)
    main(sys.argv)
    
