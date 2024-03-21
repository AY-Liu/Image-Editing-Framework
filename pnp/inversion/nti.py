from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.optim.adam import Adam
from inversion.ddim import ddim_inversion,ddim_inversion_xl

class NTI(ddim_inversion):

    def null_optimization(self, model, latents,context, num_inner_steps, epsilon,guidance_scale):  # num_inner_steps  = optimization epoch
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * model.scheduler.num_inference_steps,desc="Now doing null-text optimization")
        for i in range(model.scheduler.num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2] # = latents[- i - 2]
            t = model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = model.unet(latent_cur, t, cond_embeddings).sample
            for j in range(num_inner_steps):
                noise_pred_uncond = model.unet(latent_cur, t, uncond_embeddings).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = model.scheduler.step(noise_pred, t, latent_cur).prev_sample
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latents_input = torch.cat([latent_cur] * 2)
                noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                latent_cur = model.scheduler.step(noise_pred, t, latent_cur).prev_sample
        bar.close()
        return uncond_embeddings_list

class NTI_XL(ddim_inversion_xl):

    def null_optimization(self,model, latents, context, num_inner_steps, epsilon,guidance_scale
                          ,height=1024,width=1024):  # num_inner_steps  = optimization epoch
        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds = context
        original_size = target_size = (height, width)
        add_time_ids = model._get_add_time_ids(
            original_size, (0,0), target_size, dtype=prompt_embeds.dtype
        )
        add_time_ids = add_time_ids.to(model.unet.device)
        negative_add_time_ids = add_time_ids
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        added_uncond_kwargs = {"text_embeds": negative_pooled_prompt_embeds.detach(), "time_ids": negative_add_time_ids.detach()}
        added_kwargs = {"text_embeds": torch.cat([negative_pooled_prompt_embeds,pooled_prompt_embeds]), 
                        "time_ids": torch.cat([negative_add_time_ids,add_time_ids])}
        # null optimization
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * model.scheduler.num_inference_steps,desc="Now doing null-text optimization")
        for i in range(model.scheduler.num_inference_steps):
            uncond_embeddings = negative_prompt_embeds.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=5e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2] # = latents[- i - 2]
            t = model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = model.unet(latent_cur, t, prompt_embeds,added_cond_kwargs = added_cond_kwargs).sample
            for j in range(num_inner_steps):
                noise_pred_uncond = model.unet(latent_cur, t, uncond_embeddings,added_cond_kwargs= added_uncond_kwargs).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = model.scheduler.step(noise_pred, t, latent_cur).prev_sample
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, prompt_embeds])
                latents_input = torch.cat([latent_cur] * 2)
                noise_pred = model.unet(latents_input, t, encoder_hidden_states=context,added_cond_kwargs=added_kwargs)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                latent_cur = model.scheduler.step(noise_pred, t, latent_cur).prev_sample
        bar.close()
        return uncond_embeddings_list