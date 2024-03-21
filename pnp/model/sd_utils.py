import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.utils.torch_utils import randn_tensor

from model.register import register_attention_control_efficient,register_conv_control_efficient,register_time,unregister_attention_control_efficient,unregister_conv_control_efficient
from model.register import register_attention_control_efficient_xl,register_conv_control_efficient_xl,register_time_xl,unregister_conv_control_efficient_xl,unregister_attention_control_efficient_xl

class PnP():
    def __init__(self,pipeline,num_inference_steps) -> None:
        self.model = pipeline
        self.model.scheduler.set_timesteps(num_inference_steps)

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.model.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.model.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self.model, self.qk_injection_timesteps)
        register_conv_control_efficient(self.model, self.conv_injection_timesteps)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        # for pnp
        pnp_attn_t = 0.5,
        pnp_f_t = 0.8
    ):
        device = self.model._execution_device
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        timesteps = self.model.scheduler.timesteps
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents = latents.expand(batch_size,  self.model.unet.config.in_channels, height // 8, width // 8)
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)
        # for editing part
        pnp_f_t = int(num_inference_steps * pnp_f_t)
        pnp_attn_t = int(num_inference_steps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                register_time(self.model, t.item())

                # predict the noise residual
                noise_pred = self.model.unet(latent_model_input,t,
                                        encoder_hidden_states=prompt_embeds,
                                        cross_attention_kwargs=cross_attention_kwargs,).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
        # 8. Post-processing
        images = self.latent2image(latents)
        unregister_attention_control_efficient(self.model)
        unregister_conv_control_efficient(self.model)
        return images

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

class PnP_XL(PnP):
    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.model.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.model.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient_xl(self.model, self.qk_injection_timesteps)
        register_conv_control_efficient_xl(self.model, self.conv_injection_timesteps)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        # for pnp
        pnp_attn_t = 0.5,
        pnp_f_t = 0.8
    ):
        device = self.model._execution_device
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt,device,do_classifier_free_guidance,height,width,batch_size)

        timesteps = self.model.scheduler.timesteps
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents = latents.expand(batch_size,  self.model.unet.config.in_channels, height // 8, width // 8)
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)
        # for editing part
        pnp_f_t = int(num_inference_steps * pnp_f_t)
        pnp_attn_t = int(num_inference_steps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                register_time_xl(self.model, t.item())

                # predict the noise residual
                noise_pred = self.model.unet(latent_model_input,t,
                                        encoder_hidden_states=prompt_embeds,
                                        cross_attention_kwargs=cross_attention_kwargs,
                                        added_cond_kwargs=added_cond_kwargs).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
        # 8. Post-processing
        images = self.latent2image(latents)
        unregister_attention_control_efficient_xl(self.model)
        unregister_conv_control_efficient_xl(self.model)
        return images

    def encode_prompt_xl(self,prompt,device,do_classifier_free_guidance,height,width,batch_size):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )
        original_size = target_size = (height, width)
        add_time_ids = self.model._get_add_time_ids(
            original_size, (0,0), target_size, dtype=prompt_embeds.dtype
        )
        negative_add_time_ids = add_time_ids
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * 1, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return prompt_embeds,added_cond_kwargs

class PnP_NTI(PnP):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        # for pnp
        pnp_attn_t = 0.5,
        pnp_f_t = 0.8,
        # 
        uncond_embeddings_list = None,
    ):
        device = self.model._execution_device
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        timesteps = self.model.scheduler.timesteps
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents = latents.expand(batch_size,  self.model.unet.config.in_channels, height // 8, width // 8)
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)
        # for editing part
        pnp_f_t = int(num_inference_steps * pnp_f_t)
        pnp_attn_t = int(num_inference_steps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                register_time(self.model, t.item())

                # predict the noise residual
                prompt_embeds[0:2] = uncond_embeddings_list[i].expand(*prompt_embeds[2:4].shape)
                noise_pred = self.model.unet(latent_model_input,t,
                                        encoder_hidden_states=prompt_embeds,
                                        cross_attention_kwargs=cross_attention_kwargs,).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
        # 8. Post-processing
        images = self.latent2image(latents)
        unregister_attention_control_efficient(self.model)
        unregister_conv_control_efficient(self.model)
        return images

class PnP_XL_NTI(PnP_XL):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        # for pnp
        pnp_attn_t = 0.5,
        pnp_f_t = 0.8,
        # for NTI
        uncond_embeddings_list = None,
    ):
        device = self.model._execution_device
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt,device,do_classifier_free_guidance,height,width,batch_size)

        timesteps = self.model.scheduler.timesteps
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents = latents.expand(batch_size,  self.model.unet.config.in_channels, height // 8, width // 8)
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)
        # for editing part
        pnp_f_t = int(num_inference_steps * pnp_f_t)
        pnp_attn_t = int(num_inference_steps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                register_time_xl(self.model, t.item())

                # predict the noise residual
                prompt_embeds[0:2] = uncond_embeddings_list[i].expand(*prompt_embeds[2:4].shape)
                noise_pred = self.model.unet(latent_model_input,t,
                                        encoder_hidden_states=prompt_embeds,
                                        cross_attention_kwargs=cross_attention_kwargs,
                                        added_cond_kwargs=added_cond_kwargs).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
        # 8. Post-processing
        images = self.latent2image(latents)
        unregister_attention_control_efficient_xl(self.model)
        unregister_conv_control_efficient_xl(self.model)
        return images