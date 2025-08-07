import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision.transforms import PILToTensor
import lpips
import clip
import csv
from einops import rearrange
import time
from pytorch_lightning import seed_everything

# code credit: https://github.com/Tsingularity/dift/blob/main/src/models/dift_sd.py
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
import gc
from PIL import Image

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            
            if i > np.max(up_ft_indices):
                break
            
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            
            if i in up_ft_indices:
                up_ft[i] = sample.detach()
            
        output = {}
        output['up_ft'] = up_ft
        return output

class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(latents_noisy, 
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output


class SDFeaturizer:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1'):
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        # onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self, 
                img_tensor,
                prompt, 
                t=261, 
                up_ft_index=1, 
                ensemble_size=8):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False)[0] # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)
        unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        return unet_ft




def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w").to(device)
    return image

def fix_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':
    seed_everything(42)
    fix_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--drag_bench_root', default='./drag_bench_data', required=True, help='Root path for dragged results')
    parser.add_argument('--eval_root', required=True, help='Root path for dragged results')
    parser.add_argument('--All_IF_MD_filename', default='All_IF_MD.csv', help='CSV to save metrics')
    parser.add_argument('--Average_IF_MD_filename', default='Average_IF_MD.txt', help='TXT to save averages')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drag_bench_root = args.drag_bench_root
    all_category = [
        'art_work', 'land_scape', 'building_city_view', 'building_countryside_view',
        'animals', 'human_head', 'human_upper_body', 'human_full_body',
        'interior_design', 'other_objects',
    ]

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')
    

    all_lpips, all_clip, all_md = [], [], []

    with open(args.All_IF_MD_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SampleName', '1-lpips(IF)', 'clip sim', 'mean distance(MD)'])

        for cat in all_category:
            for file_name in os.listdir(os.path.join(drag_bench_root, cat)):
                if file_name.startswith('.'):
                    continue

                # Load paths
                ori_path = os.path.join(drag_bench_root, cat, file_name, 'original_image.png')
                drag_path = os.path.join(args.eval_root, cat, file_name, 'dragged_image.png')
                meta_path = os.path.join(drag_bench_root, cat, file_name, 'meta_data.pkl')

                if not os.path.exists(ori_path) or not os.path.exists(drag_path) or not os.path.exists(meta_path):
                    continue

                # Load and resize images
                ori_img = Image.open(ori_path)
                drag_img = Image.open(drag_path).resize(ori_img.size, Image.BILINEAR)
                meta_data = pickle.load(open(meta_path, 'rb'))
                prompt = meta_data['prompt']
                points = meta_data['points']

                # LPIPS & CLIP
                ori_tensor = preprocess_image(np.array(ori_img), device)
                drag_tensor = preprocess_image(np.array(drag_img), device)

                with torch.no_grad():
                    ori_224 = F.interpolate(ori_tensor, (224, 224), mode='bilinear')
                    drag_224 = F.interpolate(drag_tensor, (224, 224), mode='bilinear')
                    lp = loss_fn_alex(ori_224, drag_224)
                    all_lpips.append(lp.item())

                ori_clip = clip_preprocess(ori_img).unsqueeze(0).to(device)
                drag_clip = clip_preprocess(drag_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat_ori = clip_model.encode_image(ori_clip)
                    feat_drag = clip_model.encode_image(drag_clip)
                    feat_ori /= feat_ori.norm(dim=-1, keepdim=True)
                    feat_drag /= feat_drag.norm(dim=-1, keepdim=True)
                    clip_sim = (feat_ori * feat_drag).sum()
                    all_clip.append(clip_sim.cpu().numpy())

                # MD evaluation
                handle_points, target_points = [], []
                for i, pt in enumerate(points):
                    pt_tensor = torch.tensor([pt[1], pt[0]])
                    (handle_points if i % 2 == 0 else target_points).append(pt_tensor)

                ori_vis_tensor = (PILToTensor()(ori_img) / 255.0 - 0.5) * 2
                drag_vis_tensor = (PILToTensor()(drag_img) / 255.0 - 0.5) * 2
                _, H, W = ori_vis_tensor.shape

                ft_ori = dift.forward(ori_vis_tensor, prompt, t=261, up_ft_index=1, ensemble_size=8)
                ft_ori = F.interpolate(ft_ori, (H, W), mode='bilinear')
                ft_drag = dift.forward(drag_vis_tensor, prompt, t=261, up_ft_index=1, ensemble_size=8)
                ft_drag = F.interpolate(ft_drag, (H, W), mode='bilinear')

                cos = nn.CosineSimilarity(dim=1)
                sample_distance = 0.0
                for hp, tp in zip(handle_points, target_points):
                    num_channel = ft_ori.size(1)
                    src_vec = ft_ori[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
                    cos_map = cos(src_vec, ft_drag).cpu().numpy()[0]
                    max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape)
                    dist = (tp - torch.tensor(max_rc)).float().norm()
                    all_md.append(dist)
                    sample_distance += dist
                
                sample_id = f'{cat}/{file_name}'
                writer.writerow([sample_id, f'{(1 - lp).item():.4f}', f'{clip_sim.item():.4}', f'{sample_distance.item():.2}'])
                
                print(f"{sample_id}, IF: {(1 - lp).item():.4f}, CLIP_SIM: {clip_sim.item():.4f}, MD: {sample_distance.item():.2f}")
                torch.cuda.empty_cache() 

    # Write summary
    with open(args.Average_IF_MD_filename, 'w') as f:
        f.write(f"avg lpips: {np.mean(all_lpips):.4f}\n")
        f.write(f"avg 1-lpips: {1 - np.mean(all_lpips):.4f}\n")
        f.write(f"avg clip sim: {np.mean(all_clip):.4f}\n")
        f.write(f"mean distance: {np.mean(all_md):.2f}\n")



