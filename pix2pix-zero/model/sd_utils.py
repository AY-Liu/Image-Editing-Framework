import torch
from typing import Any, Dict, List, Optional, Union
from model.attention_control import prep_unet
import numpy as np

class P2P_Zero():
    def __init__(self,pipeline,num_inference_steps):
        self.model = pipeline
        self.model.scheduler.set_timesteps(num_inference_steps)
    
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        edit_dir=None,
        only_sample=False, # only perform sampling, and no editing

    ):


        # 0. modify the unet to be useful :D
        self.model.unet,self.original_processors = prep_unet(self.model.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.model._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt = 2x77x1024
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt[0],
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
        # 4. Prepare timesteps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with torch.no_grad():
            with self.model.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.model.unet(latent_model_input,t,
                                           encoder_hidden_states=prompt_embeds,
                                           cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in self.model.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        image_rec = self.latent2image(latents)

        if only_sample:
            return image_rec

        # for editing part
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt[1],
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt = None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        if edit_dir is not None:
            prompt_embeds_edit += edit_dir
        
        latents = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual
                noise_pred = self.model.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),
                                       cross_attention_kwargs=cross_attention_kwargs,).sample

                loss = 0.0
                for name, module in self.model.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention" and 'attn2' in name:
                        curr = module.attn_probs # size is num_channel,s*s,77
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr-ref)**2).sum((1,2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.model.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample
                
                latents = x_in.detach().chunk(2)[0]

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
        image_edit = self.latent2image(latents)
        return image_rec, image_edit

    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        with torch.no_grad():
            image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

class P2P_Zero_XL(P2P_Zero):
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        edit_dir=None,
        only_sample=False, # only perform sampling, and no editing

    ):
        # 0. modify the unet to be useful :D
        self.model.unet,self.original_processors = prep_unet(self.model.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.model._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt = 2x77x1024
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt[0],device,do_classifier_free_guidance,height,width,1)
        # 4. Prepare timesteps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with torch.no_grad():
            with self.model.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.model.unet(latent_model_input,t,
                                           encoder_hidden_states=prompt_embeds,
                                           cross_attention_kwargs=cross_attention_kwargs,added_cond_kwargs=added_cond_kwargs).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in self.model.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        image_rec = self.latent2image(latents)

        if only_sample:
            return image_rec

        # for editing part
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt[1],device,do_classifier_free_guidance,height,width,1)
        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        if edit_dir is not None:
            prompt_embeds_edit += edit_dir
        
        latents = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual
                noise_pred = self.model.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),
                                       cross_attention_kwargs=cross_attention_kwargs,added_cond_kwargs=added_cond_kwargs).sample

                loss = 0.0
                for name, module in self.model.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention" and 'attn2' in name:
                        curr = module.attn_probs # size is num_channel,s*s,77
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr-ref)**2).sum((1,2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.model.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,
                                                 added_cond_kwargs=added_cond_kwargs).sample
                
                latents = x_in.detach().chunk(2)[0]

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
        image_edit = self.latent2image(latents)
        return image_rec, image_edit

    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        with torch.no_grad():
            image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image
    
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
        added_cond_kwargs = {"text_embeds": add_text_embeds.detach(), "time_ids": add_time_ids.detach()}

        return prompt_embeds,added_cond_kwargs


class P2P_Zero_NTI(P2P_Zero):
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        edit_dir=None,
        only_sample=False, # only perform sampling, and no editing
        #
        uncond_embeddings_list = None,

    ):


        # 0. modify the unet to be useful :D
        self.model.unet,self.original_processors = prep_unet(self.model.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.model._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt = 2x77x1024
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt[0],
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
        # 4. Prepare timesteps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with torch.no_grad():
            with self.model.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    prompt_embeds[0] = uncond_embeddings_list[i]
                    noise_pred = self.model.unet(latent_model_input,t,
                                           encoder_hidden_states=prompt_embeds,
                                           cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in self.model.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        image_rec = self.latent2image(latents)

        if only_sample:
            return image_rec

        # for editing part
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt[1],
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt = None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        if edit_dir is not None:
            prompt_embeds_edit += edit_dir
        
        latents = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual
                prompt_embeds_edit[0] = uncond_embeddings_list[i]
                noise_pred = self.model.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),
                                       cross_attention_kwargs=cross_attention_kwargs,).sample

                loss = 0.0
                for name, module in self.model.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention" and 'attn2' in name:
                        curr = module.attn_probs # size is num_channel,s*s,77
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr-ref)**2).sum((1,2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.model.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample
                
                latents = x_in.detach().chunk(2)[0]

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
        image_edit = self.latent2image(latents)
        return image_rec, image_edit

class P2P_Zero_XL_NTI(P2P_Zero_XL):
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        edit_dir=None,
        only_sample=False, # only perform sampling, and no editing
        
        #
        uncond_embeddings_list = None,

    ):
        # 0. modify the unet to be useful :D
        self.model.unet,self.original_processors = prep_unet(self.model.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.model._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt = 2x77x1024
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt[0],device,do_classifier_free_guidance,height,width,1)
        # 4. Prepare timesteps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.model.prepare_latents(1 * num_images_per_prompt, 
                                       num_channels_latents, 
                                       height, width, 
                                       prompt_embeds.dtype, 
                                       device, 
                                       generator, 
                                       latents,)
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with torch.no_grad():
            with self.model.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    prompt_embeds[0] = uncond_embeddings_list[i]
                    noise_pred = self.model.unet(latent_model_input,t,
                                           encoder_hidden_states=prompt_embeds,
                                           cross_attention_kwargs=cross_attention_kwargs,added_cond_kwargs=added_cond_kwargs).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in self.model.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        image_rec = self.latent2image(latents)

        if only_sample:
            return image_rec

        # for editing part
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt[1],device,do_classifier_free_guidance,height,width,1)
        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        if edit_dir is not None:
            prompt_embeds_edit += edit_dir
        
        latents = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual
                prompt_embeds_edit[0] = uncond_embeddings_list[i]
                noise_pred = self.model.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),
                                       cross_attention_kwargs=cross_attention_kwargs,
                                       added_cond_kwargs=added_cond_kwargs).sample

                loss = 0.0
                for name, module in self.model.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention" and 'attn2' in name:
                        curr = module.attn_probs # size is num_channel,s*s,77
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr-ref)**2).sum((1,2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.model.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,
                                                 added_cond_kwargs=added_cond_kwargs).sample
                
                latents = x_in.detach().chunk(2)[0]

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
        image_edit = self.latent2image(latents)
        return image_rec, image_edit