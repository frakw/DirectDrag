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


import torch
import numpy as np
import copy

import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from utils.drag_utils import point_tracking, check_handle_reach_target, interpolate_feature_patch, point_tracking_clipdrag_v1, point_tracking_betterdrag, point_tracking_draglora,interpolate_feature_patch_safe
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from diffusers import DDIMScheduler, AutoencoderKL
from pytorch_lightning import seed_everything
from accelerate import Accelerator


from omegaconf import OmegaConf
from utils.readout_guidance import rg_helpers, rg_operators, rg_pipeline

import os
from bitsandbytes.optim import Adam8bit

from utils.unet_drag.unet_2d_condition import UNet2DConditionModel  # for memory
import math
import cv2

from utils.continuous_drag import drag_stretch_multipoint_ratio_interp
import torchvision.transforms.functional as Fu
from utils.shift_test import shift_matrix
use_fastdrag_unet = False
use_fastdrag_kv_copy = False
use_soft_mask = True
soft_mask_sigma = 30

use_drag_stretch = True
drag_stretch_ratio = 0.15

method = 'original'  # 'original', 'clipdrag', 'draglora' 'betterdrag'
use_readout_guidance = True  # True, False
readout_guidance_factor = 350

output_drag_stretch_only = False  # True, False

# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks
def override_forward(self):
    def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
            last_up_block_idx: int = None,
            iter_cur=0, save_kv=True
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
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

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                if use_fastdrag_unet:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        iter_cur=iter_cur, save_kv=save_kv
                    )
                else:
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

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if use_fastdrag_unet:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    iter_cur=iter_cur, save_kv=save_kv
                )
            else:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if use_fastdrag_unet:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        iter_cur=iter_cur, save_kv=save_kv
                    )
                else:
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
            all_intermediate_features.append(sample)
            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward



def point_tracking_start(features, F1, handle_points, handle_points_init, target_points, r2):
    if method == 'clipdrag':
        handle_points = point_tracking_clipdrag_v1(features,
            F1, handle_points, handle_points_init, target_points, 3)
        return handle_points
    elif method == 'original':
        handle_points = point_tracking(features,
                                       F1, handle_points, handle_points_init,target_points, r2)
        return handle_points
    elif method == 'draglora':
        handle_points, _ = point_tracking_draglora(features,
                                       F1, handle_points, handle_points_init, target_points, 3, True)
        return handle_points

    elif method == 'betterdrag':
        handle_points = point_tracking_betterdrag(features,
                                       F1, handle_points, handle_points_init, target_points, 3)
        return handle_points


class GoodDragger:
    def __init__(self, device, model_path: str, prompt: str,
                 full_height: int, full_width: int,
                 inversion_strength: float,
                 r1: int = 4, r2: int = 12, beta: int = 4,
                 drag_end_step: int = 10, track_per_denoise: int = 10,
                 lam: float = 0.2, latent_lr: float = 0.01,
                 n_inference_step: int = 50, guidance_scale: float = 1.0, feature_idx: int = 3,
                 compare_mode: bool = False,
                 vae_path: str = "default", lora_path: str = '', seed: int = 42,
                 max_drag_per_track: int = 10, drag_loss_threshold: float = 4.0, once_drag: bool = False,
                 max_track_no_change: int = 10):
        self.device = device
        self.vae_path = vae_path
        self.lora_path = lora_path
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                  beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False, steps_offset=1)

        is_sdxl = 'xl' in model_path
        self.is_sdxl = is_sdxl
        if is_sdxl:
            self.model = StableDiffusionXLPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)
            self.model.unet.config.addition_embed_type = None
        else:
            self.model = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)
        
        if use_fastdrag_unet:
            lcm_model_path = "SimianLuo/LCM_Dreamshaper_v7"
            self.model.unet = UNet2DConditionModel.from_pretrained(
                lcm_model_path,
                subfolder="unet",).to(self.device)
        
        self.modify_unet_forward()
        if vae_path != "default":
            self.model.vae = AutoencoderKL.from_pretrained(
                vae_path
            ).to(self.device, self.model.vae.dtype)

        if lora_path != "":
            self.set_lora()

        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        #if self.lora_path == "": #save memory
        #    self.model.unet.enable_gradient_checkpointing()



        seed_everything(seed)
        self.seed = seed

        self.prompt = prompt
        self.full_height = full_height
        self.full_width = full_width
        self.sup_res_h = int(0.5 * full_height)
        self.sup_res_w = int(0.5 * full_width)

        self.n_inference_step = n_inference_step
        self.n_actual_inference_step = round(inversion_strength * self.n_inference_step)
        self.guidance_scale = guidance_scale

        self.unet_feature_idx = [feature_idx]

        self.r_1 = r1
        self.r_2 = r2
        self.lam = lam
        self.beta = beta

        self.lr = latent_lr
        self.compare_mode = compare_mode

        self.t2 = drag_end_step
        self.track_per_denoise = track_per_denoise
        self.total_drag = int(track_per_denoise * self.t2)

        self.model.scheduler.set_timesteps(self.n_inference_step)

        self.do_drag = True
        self.drag_count = 0
        self.max_drag_per_track = max_drag_per_track

        self.drag_loss_threshold = drag_loss_threshold * ((2 * self.r_1) ** 2)
        self.once_drag = once_drag
        self.no_change_track_num = 0
        self.max_no_change_track_num = max_track_no_change

    def set_lora(self):
        if self.lora_path == "":
            print("applying default parameters")
            self.model.unet.set_default_attn_processor()
        else:
            print("applying lora: " + self.lora_path)
            self.model.unet.load_attn_procs(self.lora_path)

    def modify_unet_forward(self):
        self.model.unet.forward = override_forward(self.model.unet)

    def get_handle_target_points(self, points):
        handle_points = []
        target_points = []

        for idx, point in enumerate(points):
            cur_point = torch.tensor(
                [point[1] / self.full_height * self.sup_res_h, point[0] / self.full_width * self.sup_res_w])
            cur_point = torch.round(cur_point)
            if idx % 2 == 0:
                handle_points.append(cur_point)
            else:
                target_points.append(cur_point)
        print(f'handle points: {handle_points}')
        print(f'target points: {target_points}')
        return handle_points, target_points

    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.model.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

        latents = self.model.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def forward_unet_features(self, z, t, encoder_hidden_states):
        if use_fastdrag_unet:
            unet_output, all_intermediate_features = self.model.unet(
                z,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_intermediates=True,
                save_kv=use_fastdrag_kv_copy,
            )
        else:
            unet_output, all_intermediate_features = self.model.unet(
                z,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_intermediates=True,
            )

        all_return_features = []
        for idx in self.unet_feature_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (self.sup_res_h, self.sup_res_w), mode='bilinear')
            all_return_features.append(feat)        
        return_features = torch.cat(all_return_features, dim=1)

        del all_intermediate_features
        torch.cuda.empty_cache()

        return unet_output, return_features

    @torch.no_grad()
    def invert(
            self,
            image: torch.Tensor,
            prompt,
            return_intermediates=False,
            **kwds,
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(prompt)
        else:
            text_embeddings = self.get_text_embeddings(prompt)

        latents = self.image2latent(image)

        if self.guidance_scale > 1.:
            unconditional_embeddings = self.get_text_embeddings([''] * batch_size)
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("Valid timesteps: ", self.model.scheduler.timesteps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.model.scheduler.timesteps), desc="DDIM Inversion")):
            if self.n_actual_inference_step is not None and i >= self.n_actual_inference_step:
                continue

            if self.guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            t_ = self.model.scheduler.timesteps[-(i + 2)]

            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=text_embeddings,iter_cur=len(self.model.scheduler.timesteps)-i-1, save_kv=use_fastdrag_kv_copy)
            if self.guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + self.guidance_scale * (noise_pred_con - noise_pred_uncon)

            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            return latents, latents_list
        return latents

    def get_original_features(self, init_code, text_embeddings):
        timesteps = self.model.scheduler.timesteps
        strat_time_step_idx = self.n_inference_step - self.n_actual_inference_step
        original_step_output = {}
        features = {}
        cur_latents = init_code.detach().clone()
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps[strat_time_step_idx:],
                                       desc="Denosing for mask features")):
                if i <= self.t2:
                    model_inputs = cur_latents
                    noise_pred, F0 = self.forward_unet_features(model_inputs, t, encoder_hidden_states=text_embeddings)
                    cur_latents = self.model.scheduler.step(noise_pred, t, model_inputs, return_dict=False)[0]
                    original_step_output[t.item()] = cur_latents.cpu()
                    features[t.item()] = F0.cpu()

        del noise_pred, cur_latents, F0
        torch.cuda.empty_cache()

        return original_step_output, features

    def get_noise_features(self, input_latents, t, text_embeddings):
        unet_output, F1 = self.forward_unet_features(input_latents, t, encoder_hidden_states=text_embeddings)
        return unet_output, F1

    def cal_motion_supervision_loss(self, handle_points, target_points, F1, x_prev_updated, original_prev,
                                    interp_mask, original_features, original_points, alpha=None):
        drag_loss = 0.0
        for i_ in range(len(handle_points)):
            pi, ti = handle_points[i_], target_points[i_]
            norm_dis = (ti - pi).norm()
            if norm_dis < 2.:
                continue

            di = (ti - pi) / (ti - pi).norm() * min(self.beta, norm_dis)

            original_features.requires_grad_(True)
            pi = original_points[i_]
            f0_patch = original_features[:, :, int(pi[0]) - self.r_1:int(pi[0]) + self.r_1 + 1,
                       int(pi[1]) - self.r_1:int(pi[1]) + self.r_1 + 1].detach()

            pi = handle_points[i_]
            f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], self.r_1)
            drag_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch)

        print(f'Loss from drag: {drag_loss}')

        
        if use_soft_mask:
            # 讓變動懲罰根據 soft_mask 平滑地下降
            stability_weight = 1.0 - interp_mask  # 還是要讓背景穩定
            loss = drag_loss + self.lam * (stability_weight * (x_prev_updated - original_prev).abs()).sum()
        else:
            loss = drag_loss + self.lam * ((x_prev_updated - original_prev)
                                           * (1.0 - interp_mask)).abs().sum()
        

        print('Loss total=%f' % loss)
        return loss, drag_loss

    def track_step(self, original_feature, original_feature_, F1, F1_, handle_points, handle_points_init):
        if self.compare_mode:
            handle_points = point_tracking(original_feature,
                                           F1, handle_points, handle_points_init, self.r_2)
        else:
            handle_points = point_tracking(original_feature_,
                                           F1_, handle_points, handle_points_init, self.r_2)
        return handle_points

    def compare_tensor_lists(self, lst1, lst2):
        if len(lst1) != len(lst2):
            return False
        return all(torch.equal(t1, t2) for t1, t2 in zip(lst1, lst2))

    def gooddrag_step(self, init_code, t, t_, text_embeddings, handle_points, target_points,
                      features, handle_points_init, original_step_output, interp_mask,rg_controller,rg_latents,rg_feat_f0,step_idx):
        drag_latents = init_code.clone().detach()
        drag_latents.requires_grad_(True)

        first_drag = True
        need_track = False
        track_num = 0
        cur_drag_per_track = 0
        self.compare_mode = True
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision='fp16'
        )

        optimizer = torch.optim.Adam([drag_latents], lr=self.lr)

        target_embeddings = text_embeddings
        drag_latents, self.model.unet, optimizer = accelerator.prepare(drag_latents, self.model.unet, optimizer)
        while track_num < self.track_per_denoise:
            optimizer.zero_grad()
            #unet_output, F1 = self.forward_unet_features(drag_latents, t, text_embeddings)
            unet_output, F1 = self.forward_unet_features(drag_latents, t, target_embeddings)
            x_prev_updated = self.model.scheduler.step(unet_output, t, drag_latents, return_dict=False)[0]

            if (need_track or first_drag) and (not self.compare_mode):
                with torch.no_grad():
                    #_, F1_ = self.forward_unet_features(x_prev_updated, t_, text_embeddings)
                    _, F1_ = self.forward_unet_features(x_prev_updated, t_, target_embeddings)

            if first_drag:
                first_drag = False
                if self.compare_mode:
                    handle_points = point_tracking_start(features[t.item()].cuda(),
                                F1, handle_points, handle_points_init, target_points, self.r_2)
                    #handle_points = point_tracking(features[t.item()].cuda(),
                    #                               F1, handle_points, handle_points_init, self.r_2)
                    #handle_points = point_tracking_clipdrag_v1(features[t.item()].cuda(),
                    #            F1, handle_points, handle_points_init, target_points, 3)
                else:
                    handle_points = point_tracking_start(features[t_.item()].cuda(),
                                F1_, handle_points, handle_points_init, target_points, self.r_2)
                    #handle_points = point_tracking(features[t_.item()].cuda(),
                    #                               F1_, handle_points, handle_points_init, self.r_2)
                    #handle_points = point_tracking_clipdrag_v1(features[t_.item()].cuda(),
                    #            F1_, handle_points, handle_points_init, target_points, 3)

                print(f'After denoise new handle points: {handle_points}, drag count: {self.drag_count}')

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                self.do_drag = False
                print('Reached the target points')
                break

            #if self.no_change_track_num == self.max_no_change_track_num:
            #    self.do_drag = False
            #    print('Early stop.')
            #    break

            del unet_output
            if need_track and (not self.compare_mode):
                del _
            torch.cuda.empty_cache()

            loss, drag_loss = self.cal_motion_supervision_loss(handle_points, target_points, F1, x_prev_updated,
                                                               original_step_output[t.item()].cuda(), interp_mask,
                                                               original_features=features[t.item()].cuda(),
                                                               original_points=handle_points_init)
            
            if use_readout_guidance:
                appled_readout_guidance = True
                rg_feat_f1 = rg_controller.collect_and_resize_feats()
                feats = torch.cat([rg_feat_f0, rg_feat_f1], dim=0) 
                feats = feats.half() #special add in gooddrag
                log = False
                emb = rg_helpers.embed_timestep(self.model.unet, init_code, t)
                latents_scale = (init_code.detach().min(), init_code.detach().max())
                # Compute the loss over both the uncond and cond branch
                #for gt_idx in [0, b//2]:
                gt_idx = 0
                log_branch = (log and gt_idx != 0)
                batch_idx = gt_idx + 1
                #latents_scale = (latents.detach().min(), latents.detach().max())
                rg_loss = rg_operators.loss_guidance(rg_controller, feats, batch_idx, gt_idx, edits=rg_controller.edits, log=log_branch, emb=emb, latents_scale=latents_scale, t=t, i=step_idx)
                
                rg_loss = rg_loss * readout_guidance_factor

                print('rg_loss= ',rg_loss)

                loss += rg_loss
            
            if output_drag_stretch_only:
                pass
            else:
                accelerator.backward(loss)
                optimizer.step()

            cur_drag_per_track += 1
            need_track = (cur_drag_per_track == self.max_drag_per_track) or (
                    drag_loss <= self.drag_loss_threshold) or self.once_drag
            if need_track:
                track_num += 1
                handle_points_cur = copy.deepcopy(handle_points)
                if self.compare_mode:
                    handle_points = point_tracking_start(features[t.item()].cuda(),
                                F1, handle_points, handle_points_init, target_points, self.r_2)
                    #handle_points = point_tracking(features[t.item()].cuda(),
                    #                               F1, handle_points, handle_points_init, self.r_2)
                    #handle_points = point_tracking_clipdrag_v1(features[t.item()].cuda(),
                    #            F1, handle_points, handle_points_init, target_points, 3)
                else:
                    handle_points = point_tracking_start(features[t_.item()].cuda(),
                                F1_, handle_points, handle_points_init, target_points, self.r_2)
                    #handle_points = point_tracking(features[t_.item()].cuda(),
                    #                               F1_, handle_points, handle_points_init, self.r_2)
                    #handle_points = point_tracking_clipdrag_v1(features[t_.item()].cuda(),
                    #            F1_, handle_points, handle_points_init, target_points, 3)

                if self.compare_tensor_lists(handle_points, handle_points_cur):
                    self.no_change_track_num += 1
                    print(f'{self.no_change_track_num} times handle points no changes.')
                else:
                    self.no_change_track_num = 0

                self.drag_count += 1
                cur_drag_per_track = 0
                print(f'New handle points: {handle_points}, drag count: {self.drag_count}')

        init_code = drag_latents.clone().detach()
        init_code.requires_grad_(False)
        del optimizer, drag_latents
        torch.cuda.empty_cache()

        return init_code, handle_points, text_embeddings, target_embeddings.detach().requires_grad_(False)

    def prepare_mask(self, mask):
        mask = torch.from_numpy(mask).float() / 255.
        mask[mask > 0.0] = 1.0
        mask = rearrange(mask, "h w -> 1 1 h w").cuda()
        mask = F.interpolate(mask, (self.sup_res_h, self.sup_res_w), mode="nearest")
        return mask
    def prepare_soft_mask(self, handle_points, target_points, shape):
        self.sigma = 20.0
        # 建立空白地圖
        soft_mask = torch.zeros(shape).to(self.device)

        for pt in handle_points + target_points:
            x, y = int(pt[0]), int(pt[1])
            for i in range(shape[2]):
                for j in range(shape[3]):
                    dist_sq = (i - x)**2 + (j - y)**2
                    soft_mask[0, 0, i, j] += math.exp(-dist_sq / (2 * self.sigma ** 2))

        # 正規化到 [0, 1]
        soft_mask = soft_mask / soft_mask.max()
        return soft_mask

    def prepare_soft_mask_v3(self, handle_points, target_points, shape):
        """
        沿 handle → target 線上插值點，畫到 map 上，經過 GaussianBlur 得到 soft mask。
        """
        self.sigma = 30.0
        H, W = shape[2], shape[3]

        

        # 建立空白 2D numpy map
        point_map = np.zeros((H, W), dtype=np.float32)

        for h_pt, t_pt in zip(handle_points, target_points):
            hx, hy = h_pt[0], h_pt[1]
            tx, ty = t_pt[0], t_pt[1]

            # 👉 動態決定插值數量
            dist = math.sqrt((tx - hx)**2 + (ty - hy)**2)
            num_steps = int(dist / 5)
            num_steps = max(1, num_steps)

            for step in range(num_steps):
                alpha = step / num_steps
                x = int((1 - alpha) * hx + alpha * tx)
                y = int((1 - alpha) * hy + alpha * ty)

                if 0 <= x < W and 0 <= y < H:
                    point_map[y, x] = 1.0  # 注意是 (y, x)

        # 模糊權重圖：讓線上的點「光暈」化
        kernel_size = int(self.sigma * 4) | 1  # 取奇數
        blurred_map = cv2.GaussianBlur(point_map, (kernel_size, kernel_size), sigmaX=self.sigma)

        # 正規化到 [0, 1]
        blurred_map /= (blurred_map.max() + 1e-6)

        # 回轉成 torch tensor（[1, 1, H, W]）
        soft_mask = torch.from_numpy(blurred_map).unsqueeze(0).unsqueeze(0).to(self.device)

        return soft_mask

    def prepare_soft_mask_v4(self, handle_points, target_points, shape):
        """
        不插值點，而是沿 handle→target 線段掃描所有經過的 pixel，建立 binary mask，再 GaussianBlur。
        """
        self.sigma = soft_mask_sigma
        #self.sigma = 50.0
        #self.sigma = 10.0
        #self.sigma = 15.0
        H, W = shape[2], shape[3]

        point_map = np.zeros((H, W), dtype=np.float32)

        for h_pt, t_pt in zip(handle_points, target_points):
            hx, hy = [int(round(v.item())) for v in h_pt]
            tx, ty = [int(round(v.item())) for v in t_pt]

            # 使用 Bresenham-like 直線掃描（或 numpy 版本的 line iterator）
            num_steps = max(abs(tx - hx), abs(ty - hy)) + 1
            for step in range(num_steps):
                alpha = step / (num_steps - 1) if num_steps > 1 else 0
                x = int(round((1 - alpha) * hx + alpha * tx))
                y = int(round((1 - alpha) * hy + alpha * ty))

                if 0 <= x < W and 0 <= y < H:
                    point_map[y, x] = 1.0  # 注意是 y, x（行, 列）

        # 模糊處理
        kernel_size = int(self.sigma * 4) | 1
        blurred_map = cv2.GaussianBlur(point_map, (kernel_size, kernel_size), sigmaX=self.sigma)

        # 正規化
        blurred_map /= (blurred_map.max() + 1e-6)

        # 轉成 tensor
        return torch.from_numpy(blurred_map).unsqueeze(0).unsqueeze(0).to(self.device)

    def prepare_soft_mask_v5(self, handle_points, target_points, shape):
        """
        每條 handle→target 線段根據距離自適應調整 sigma，生成多條模糊區域合成的 soft mask。
        """
        H, W = shape[2], shape[3]
        blurred_total = np.zeros((H, W), dtype=np.float32)

        # 可自訂的 sigma 範圍
        min_sigma = 5.0
        max_sigma = 40.0
        max_dist = math.hypot(H, W)

        for h_pt, t_pt in zip(handle_points, target_points):
            hx, hy = [int(round(v.item())) for v in h_pt]
            tx, ty = [int(round(v.item())) for v in t_pt]

            # === 1. 自適應 sigma ===
            dist = math.hypot(tx - hx, ty - hy)
            sigma = min_sigma + (max_sigma - min_sigma) * (dist / max_dist)
            sigma = max(min_sigma, min(sigma, max_sigma))

            # === 2. 掃描中間線段點 ===
            point_map = np.zeros((H, W), dtype=np.float32)
            num_steps = max(abs(tx - hx), abs(ty - hy)) + 1

            for step in range(num_steps):
                alpha = step / (num_steps - 1) if num_steps > 1 else 0
                x = int(round((1 - alpha) * hx + alpha * tx))
                y = int(round((1 - alpha) * hy + alpha * ty))

                if 0 <= x < W and 0 <= y < H:
                    point_map[y, x] = 1.0

            # === 3. 套用高斯模糊 ===
            kernel_size = int(sigma * 4) | 1
            blurred = cv2.GaussianBlur(point_map, (kernel_size, kernel_size), sigmaX=sigma)

            blurred_total += blurred

        # === 4. 正規化合併結果 ===
        blurred_total /= (blurred_total.max() + 1e-6)

        return torch.from_numpy(blurred_total).unsqueeze(0).unsqueeze(0).to(self.device)

    def prepare_soft_mask_v1(self, handle_points, target_points, shape):
        self.sigma = 20.0  # 可調參數
        H, W = shape[2], shape[3]

        # 1. 建立空白點圖
        point_map = np.zeros((H, W), dtype=np.float32)

        # 2. 在每個 handle/target 點位置設為 1.0
        for pt in handle_points + target_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                point_map[y, x] = 1.0  # 注意 numpy 是 [y, x]

        # 3. 高斯模糊，產生 soft 權重圖
        ksize = int(self.sigma * 4) | 1  # 保證為奇數
        decay_map = cv2.GaussianBlur(point_map, (ksize, ksize), sigmaX=self.sigma)

        # 4. 正規化到 [0, 1]
        decay_map /= (decay_map.max() + 1e-6)

        # 5. 轉換為 torch tensor [1, 1, H, W]
        soft_mask = torch.from_numpy(decay_map).unsqueeze(0).unsqueeze(0).to(self.device)

        return soft_mask

    def prepare_soft_mask_v2(self, mask, handle_points, target_points, shape, sigma=20):
        """
        結合 mask 與 point-based 距離衰減，只在 mask 中進行高斯衰減。
        - mask: 原始 binary mask (1 表示拖曳區域)
        - handle_points, target_points: 拖曳點 (tensor 或 list)
        - shape: (1, 1, H, W) — 輸出的 soft mask shape
        - sigma: 高斯模糊半徑
        """
        H, W = shape[2], shape[3]

        # 建立初始空白圖
        decay_map = np.zeros((H, W), dtype=np.float32)

        # 畫出所有點的位置（設為 1）
        for pt in handle_points + target_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                decay_map[y, x] = 1.0  # 注意 numpy 座標是 y, x

        # 套用高斯模糊產生距離衰減場
        blur_radius = int(sigma * 4) | 1  # kernel size 要是奇數
        decay_map = cv2.GaussianBlur(decay_map, (blur_radius, blur_radius), sigmaX=sigma)

        # Normalize to [0, 1]
        decay_map = decay_map / (decay_map.max() + 1e-6)

        # 將原始二值 mask 應用：只在 mask 區域保留權重
        mask_np = mask.squeeze().cpu().numpy().astype(np.float32) / 255.0  # 假設是 255 二值圖
        soft_mask_np = decay_map * mask_np  # 遮罩外變 0，遮罩內根據點距離衰減

        # 轉回 tensor
        soft_mask = torch.from_numpy(soft_mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return soft_mask

    def set_latent_masactrl(self):
        editor = MutualSelfAttentionControl(start_step=0,
                                            start_layer=10,
                                            total_steps=self.n_inference_step,
                                            guidance_scale=self.guidance_scale)
        if self.lora_path == "":
            register_attention_editor_diffusers(self.model, editor, attn_processor='attn_proc')
        else:
            register_attention_editor_diffusers(self.model, editor, attn_processor='lora_attn_proc')

    def get_intermediate_images(self, intermediate_images, intermediate_images_original, intermediate_images_t_idx,
                                valid_timestep, text_embeddings):
        for i in range(len(intermediate_images)-1):
            current_original_code = intermediate_images_original[i].to(self.device)
            current_init_code = intermediate_images[i].to(self.device)

            self.set_latent_masactrl()

            for inter_i, inter_t in enumerate(valid_timestep[intermediate_images_t_idx[i] + 1:]):
                with torch.no_grad():
                    noise_pred_all = self.model.unet(torch.cat([current_original_code, current_init_code]), inter_t,
                                                     encoder_hidden_states=torch.cat(
                                                         [text_embeddings, text_embeddings]),iter_cur=inter_i, save_kv=use_fastdrag_kv_copy)
                    noise_pred = noise_pred_all[1]
                    noise_pred_original = noise_pred_all[0]
                    current_init_code = \
                        self.model.scheduler.step(noise_pred, inter_t, current_init_code, return_dict=False)[0]
                    current_original_code = \
                        self.model.scheduler.step(noise_pred_original, inter_t, current_original_code,
                                                  return_dict=False)[0]
            intermediate_images[i] = self.latent2image(current_init_code, return_type="pt").cpu()
        intermediate_images.pop()
        return intermediate_images

    def refine_points_to_feature_peaks(self, features: torch.Tensor, points: List[torch.Tensor], window_size: int = 5) -> List[torch.Tensor]:
        """
        將每個點向 feature map 上的局部最大值靠攏。

        Args:
            features (torch.Tensor): [B, C, H, W] 的 feature tensor
            points (List[Tensor]): 每個點是 [y, x] 的 tensor（座標已符合 feature 尺寸）
            window_size (int): 搜尋區域半徑（總區域大小為 (2*window+1)^2）

        Returns:
            List[Tensor]: 更新後的 refined 點
        """
        _, _, H, W = features.shape
        refined_points = []

        for pt in points:
            y, x = int(pt[0]), int(pt[1])

            y0 = max(0, y - window_size)
            y1 = min(H, y + window_size + 1)
            x0 = max(0, x - window_size)
            x1 = min(W, x + window_size + 1)

            patch = features[0, :, y0:y1, x0:x1]  # [C, h, w]
            score_map = patch.norm(dim=0)        # [h, w] 特徵強度

            max_pos = torch.nonzero(score_map == score_map.max(), as_tuple=False)[0]
            dy, dx = max_pos.tolist()
            new_y = y0 + dy
            new_x = x0 + dx

            refined_points.append(torch.tensor([new_y, new_x], device=features.device, dtype=torch.float))

        return refined_points

    def good_drag(self,
                  source_image,
                  points,
                  mask,
                  return_intermediate_images=False,
                  return_intermediate_features=False
                  ):
        
        
        # readout guidance
        # Create root save folder
        config_path = "./rg_config.yaml"
        rg_config = OmegaConf.load(config_path)
        save_folder = rg_config["output_dir"]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        #OmegaConf.save(rg_config, f"{save_folder}/config.yaml")

        handle_points, target_points = self.get_handle_target_points(points)

        # Create edit config and load aggregation network
        num_frames = rg_config.get("num_frames", 2)
        device="cuda"
        dtype=torch.float16
        rg_edits = rg_helpers.get_edits(rg_config, device, dtype)

        latent_height = latent_width = self.model.unet.config.sample_size
        height = width = latent_height * self.model.vae_scale_factor
        image_dim = (width, height)
        latent_dim = (latent_height, latent_width)
        batch_size = 1
        #args.negative_prompt = rg_config.get("negative_prompt", "")

        rg_prompts, rg_latents = rg_helpers.get_prompts_latents(
            self.model,
            self.prompt,
            batch_size, 
            self.seed,
            latent_dim,
            device,
            dtype,
        )

        rg_controller = rg_pipeline.ReadoutGuidance(
            self.model,
            edits=rg_edits,
            latent_dim=latent_dim
        )


        init_code = self.invert(source_image, self.prompt)

        rg_feat_f0 = rg_controller.collect_and_resize_feats()




        original_init = init_code.detach().clone()
        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(self.prompt)
            text_embeddings = text_embeddings.detach()
        else:
            text_embeddings = self.get_text_embeddings(self.prompt).detach()

        self.model.text_encoder.to('cpu')
        self.model.vae.encoder.to('cpu')

        timesteps = self.model.scheduler.timesteps
        start_time_step_idx = self.n_inference_step - self.n_actual_inference_step

        
        original_step_output, features = self.get_original_features(init_code, text_embeddings)
        #print(features.keys())
        #t0 = sorted(features.keys())[-1]
        #initial_feature = features[t0]
        #handle_points = self.refine_points_to_feature_peaks(initial_feature, handle_points)
        #target_points = self.refine_points_to_feature_peaks(initial_feature, target_points)

        handle_points_init = copy.deepcopy(handle_points)
        mask = np.ones((source_image.shape[-2], source_image.shape[-1]), dtype=np.uint8)
        mask = self.prepare_mask(mask)

        
        if use_drag_stretch:
            ratio = drag_stretch_ratio
            new_target_points = [
                h + (t - h) * ratio for h, t in zip(handle_points, target_points)
            ]
            # add
            shift_yx = new_target_points[0]-handle_points[0] # only one point
            shift_yx = shift_yx.to(device=source_image.device)

            # mask_cp_handle = Fu.resize(mask, (64, 64))
            mask_cp_handle = Fu.resize(mask, (int(mask.shape[-2]/4), int(mask.shape[-1]/4))) # some image are not h==w
            shift_y,shift_x= int(shift_yx[0]/4),int(shift_yx[1]/4)
            mask_cp_target = shift_matrix(mask_cp_handle, shift_x, shift_y)

            init_code = drag_stretch_multipoint_ratio_interp(invert_code=init_code,
                                        handle_points=handle_points,
                                        target_points=new_target_points,
                                        mask_cp_handle=mask_cp_handle,
                                        fill_mode='interpolation')
                
        if use_soft_mask:
            #mask = self.prepare_soft_mask(handle_points, target_points, (1,1,self.sup_res_h, self.sup_res_w))
            #mask = self.prepare_soft_mask_v3(handle_points, target_points, (1,1,self.sup_res_h, self.sup_res_w))
            mask = self.prepare_soft_mask_v4(handle_points, target_points, (1,1,self.sup_res_h, self.sup_res_w))
            #mask = self.prepare_soft_mask_v5(handle_points, target_points, (1,1,self.sup_res_h, self.sup_res_w))
            #mask = self.prepare_soft_mask_v1(handle_points, target_points, (1,1,self.sup_res_h, self.sup_res_w))
            #mask = self.prepare_soft_mask_v2(mask, handle_points, target_points, mask.shape)

        interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')
        #interp_mask = None
        intermediate_features = [init_code.detach().clone().cpu()] if return_intermediate_features else []
        valid_timestep = timesteps[start_time_step_idx:]
        set_mutual = True

        intermediate_images, intermediate_images_original, intermediate_images_t_idx = [], [], []

        did_drag = False
        step_idx = 0
        for i, t in enumerate(tqdm(valid_timestep,
                                   desc="Drag and Denoise")):
            if i < self.t2 and self.do_drag and (self.no_change_track_num != self.max_no_change_track_num):
                t_ = valid_timestep[i + 1]
                init_code, handle_points , text_embeddings, target_embeddings \
                    = self.gooddrag_step(init_code, t, t_, text_embeddings, handle_points,
                                                              target_points, features, handle_points_init,
                                                              original_step_output, interp_mask,
                                                              rg_controller,rg_latents,rg_feat_f0,step_idx)
                step_idx += 1
                did_drag = True
            else:
                if set_mutual:
                    set_mutual = False
                    self.set_latent_masactrl()

            with torch.no_grad():
                noise_pred_all = self.model.unet(torch.cat([original_init, init_code]), t,
                                                 #encoder_hidden_states=torch.cat([text_embeddings, text_embeddings]),
                                                 encoder_hidden_states=torch.cat([text_embeddings, target_embeddings]),
                                                 iter_cur=i, save_kv=use_fastdrag_kv_copy)
                noise_pred = noise_pred_all[1]
                noise_pred_original = noise_pred_all[0]
                init_code = self.model.scheduler.step(noise_pred, t, init_code, return_dict=False)[0]
                original_init = self.model.scheduler.step(noise_pred_original, t, original_init, return_dict=False)[0]
                #if did_drag:
                #    original_step_output, features = self.get_original_features(init_code, text_embeddings)

            if did_drag and return_intermediate_images:
                current_init_code = init_code.detach().clone()
                current_original_code = original_init.detach().clone()

                intermediate_images.append(current_init_code.cpu())
                intermediate_images_original.append(current_original_code.cpu())
                intermediate_images_t_idx.append(i)
            did_drag = False
            if return_intermediate_features:
                intermediate_features.append(init_code.detach().clone().cpu())

        if return_intermediate_images:
            intermediate_images = self.get_intermediate_images(intermediate_images, intermediate_images_original,
                                                               #intermediate_images_t_idx, valid_timestep, text_embeddings)
                                                               intermediate_images_t_idx, valid_timestep, target_embeddings)

        image = self.latent2image(init_code, return_type="pt")
        print(f'points={points}')
        return image, intermediate_features, handle_points, intermediate_images
