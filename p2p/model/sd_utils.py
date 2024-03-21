from typing import Optional,List
import torch
from model.register import register_attention_control
from tqdm import tqdm
import numpy as np
from diffusers.utils.torch_utils import randn_tensor


class P2P():
    def __init__(self,model,num_inference_steps) -> None:
        model.scheduler.set_timesteps(num_inference_steps)

    def init_latent(self,latent, model, height, width, generator, batch_size):
        if latent is None:
            latent = randn_tensor(
                (1, model.unet.config.in_channels, height // 8, width // 8),
                generator=generator,device=model.device, dtype=model.unet.dtype
            )
        latent = latent * model.scheduler.init_noise_sigma
        latents = latent.expand(batch_size,  model.unet.config.in_channels, height // 8, width // 8)
        return latent, latents

    @torch.no_grad()
    def text2image_ldm_stable(
        self,
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
    ):
        # register the model
        if controller is not None:
            register_attention_control(model, controller)
        height = width = 512
        batch_size = len(prompt)
        
        # get text embeddings
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        
        # get latent
        latent, latents = self.init_latent(latent, model, height, width, generator, batch_size)
        
        model.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(model.scheduler.timesteps,desc="Now doing P2P editing"):
            latents = self.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
        
        image = self.latent2image(model.vae, latents)
        return image, latent

    def diffusion_step(self,model, controller, latents, context, t, guidance_scale, low_resource=False):
        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        if controller is not None:
            latents = controller.step_callback(latents)
        return latents
    
    @torch.no_grad()
    def latent2image(self,vae, latents):
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

class P2P_NTI(P2P):
    @torch.no_grad()
    def text2image_ldm_stable(
        self,
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
        # for NTI
        uncond_embeddings_list = None,
    ):
        # register the model
        if controller is not None:
            register_attention_control(model, controller)
        height = width = 512
        batch_size = len(prompt)
        
        # get text embeddings
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        max_length = text_input.input_ids.shape[-1]
        if uncond_embeddings_list == None:
            uncond_input = model.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        
        # get latent
        latent, latents = self.init_latent(latent, model, height, width, generator, batch_size)
        
        # set timesteps
    #     extra_set_kwargs = {"offset": 1}
        model.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(model.scheduler.timesteps,desc="Now doing P2P editing")):
            if uncond_embeddings_list == None:
                context = torch.cat([uncond_embeddings, text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_list[i].expand(*text_embeddings.shape), text_embeddings])
            latents = self.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
        image = self.latent2image(model.vae, latents)
        return image, latent
    
class P2P_XL(P2P):

    @torch.no_grad()
    def text2image_ldm_stable(
        self,
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
    ):
        # register the model
        if controller is not None:
            register_attention_control(model, controller)
        height = width = 1024
        batch_size = len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0
        device = model._execution_device
        # get text embeddings
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(model,prompt,device,do_classifier_free_guidance,height,width,batch_size)
        # get latent
        latent, latents = self.init_latent(latent, model, height, width, generator, batch_size)
        
        model.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(model.scheduler.timesteps,desc="Now doing P2P editing"):
            latents = self.diffusion_step(model, controller, latents, prompt_embeds, t, guidance_scale, added_cond_kwargs ,low_resource)
        
        image = self.latent2image(model.vae, latents)
        return image, latent

    def diffusion_step(self,model, controller, latents, context, t, guidance_scale, added_cond_kwargs, low_resource=False):
        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0],added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1],added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context,added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        if controller is not None:
            latents = controller.step_callback(latents)
        return latents
    
    def encode_prompt_xl(self,model,prompt,device,do_classifier_free_guidance,height,width,batch_size):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = model.encode_prompt(
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
        add_time_ids = model._get_add_time_ids(
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

class P2P_XL_NTI(P2P):
    @torch.no_grad()
    def text2image_ldm_stable(
        self,
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
        # for NTI
        uncond_embeddings_list = None,
    ):
        # register the model
        if controller is not None:
            register_attention_control(model, controller)
        height = width = 1024
        batch_size = len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0
        device = model._execution_device
        # get text embeddings
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(model,prompt,device,do_classifier_free_guidance,height,width,batch_size)
        # get latent
        latent, latents = self.init_latent(latent, model, height, width, generator, batch_size)
        
        model.scheduler.set_timesteps(num_inference_steps)
        for i,t in enumerate(tqdm(model.scheduler.timesteps,desc="Now doing P2P editing")):
            prompt_embeds[0:2] = uncond_embeddings_list[i].expand(*prompt_embeds[2:4].shape)
            latents = self.diffusion_step(model, controller, latents, prompt_embeds, t, guidance_scale, added_cond_kwargs ,low_resource)
        image = self.latent2image(model.vae, latents)
        return image, latent

    def diffusion_step(self,model, controller, latents, context, t, guidance_scale, added_cond_kwargs, low_resource=False):
        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0],added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1],added_cond_kwargs=added_cond_kwargs)["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context,added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        if controller is not None:
            latents = controller.step_callback(latents)
        return latents
    
    def encode_prompt_xl(self,model,prompt,device,do_classifier_free_guidance,height,width,batch_size):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = model.encode_prompt(
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
        add_time_ids = model._get_add_time_ids(
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