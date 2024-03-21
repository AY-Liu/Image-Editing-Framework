import torch
import numpy as np
from typing import  Union
from PIL import Image
from tqdm import tqdm

class ddim_inversion():

    def ddim_reverse(self,model, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        next_timestep = timestep.item()
        timestep = min(model.scheduler.config.num_train_timesteps - 1, next_timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * pred_original_sample + next_sample_direction
        return next_sample
    
    @torch.no_grad()
    def ddim_inversion_loop(self,model,latent,prompt,cross_attention_kwargs=None):
        context = self.get_context(model,prompt)
        uncond_embeddings, cond_embeddings = context.chunk(2)
        # inversion
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(model.scheduler.num_inference_steps),desc="Now doing inversion"):
            t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
            noise_pred = model.unet(latent, t, encoder_hidden_states=cond_embeddings,cross_attention_kwargs=cross_attention_kwargs).sample
            latent = self.ddim_reverse(model, noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent,context
    
    @torch.no_grad()
    def image2latent(self, model, image,device,dtype):
        image = np.array(image)
        image = torch.from_numpy(image).to(dtype) / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        latents = model.vae.encode(image)['latent_dist'].mean
        latents = latents * model.vae.config.scaling_factor
        return latents
    
    def get_context(self,model,prompt):
        batch_size = len(prompt)
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",)
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        return context

class ddim_inversion_xl(ddim_inversion):

    @torch.no_grad()
    def ddim_inversion_loop(self,model, latent, prompt,cross_attention_kwargs=None
                            ,height=1024,width=1024):
        # get context
        context = self.get_context(model,prompt)
        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds = context
        original_size = target_size = (height, width)
        add_time_ids = model._get_add_time_ids(
            original_size, (0,0), target_size, dtype=prompt_embeds.dtype
        )
        device = model._execution_device
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = pooled_prompt_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1 * 1, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        
        # inversion
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(model.scheduler.num_inference_steps),desc="Now doing inversion"):
            t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
            noise_pred = model.unet(latent, t, encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,
                                    added_cond_kwargs = added_cond_kwargs
                                    ).sample
            latent = self.ddim_reverse(model, noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent,context

    def get_context(self,model,prompt,):
        (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = model.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=model.unet.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )
        return (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds)
