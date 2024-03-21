import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

class MasaCtrl():
    def __init__(self,pipeline,num_inference_steps) -> None:
        self.model = pipeline
        self.model.scheduler.set_timesteps(num_inference_steps)

    @torch.no_grad()
    def latent2image(self , latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        **kwds):
        DEVICE = self.model.unet.device
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )

        text_embeddings = self.model.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.model.unet.config.in_channels, height//8, width//8)
        if latents is None:
            latents = randn_tensor(latents_shape,device=DEVICE,dtype=self.model.unet.dtype,generator=None)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."
        init_latent = latents.clone()
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.model.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.model.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.model.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.model.scheduler.timesteps))
        # latents_list = [latents]
        # pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict tghe noise
            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            model_out = self.model.scheduler.step(noise_pred, t, latents,return_dict=True)
            latents, pred_x0 = model_out["prev_sample"],model_out['pred_original_sample']
            # latents_list.append(latents)
            # pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="np")
        # if return_intermediates:
        #     pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
        #     latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
        #     return image, pred_x0_list, latents_list
        return image,init_latent
    

class MasaCtrl_XL(MasaCtrl):
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        ):
        DEVICE = self.model.unet.device
        do_classifier_free_guidance = guidance_scale>1.
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        # get text embeddings
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt,DEVICE,do_classifier_free_guidance,height,width,batch_size)
        # define initial latents
        latents_shape = (batch_size, self.model.unet.config.in_channels, height//8, width//8)
        if latents is None:
            latents = randn_tensor(latents_shape,device=DEVICE,dtype=self.model.unet.dtype,generator=None)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."
        init_latent = latents.clone()
        print("latents shape: ", latents.shape)
        # iterative sampling
        self.model.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.model.scheduler.timesteps))
        # latents_list = [latents]
        # pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            # predict tghe noise
            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=prompt_embeds,
                                         added_cond_kwargs=added_cond_kwargs).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            model_out = self.model.scheduler.step(noise_pred, t, latents,return_dict=True)
            latents, pred_x0 = model_out["prev_sample"],model_out['pred_original_sample']
            # latents_list.append(latents)
            # pred_x0_list.append(pred_x0)
        image = self.latent2image(latents, return_type="np")
        # if return_intermediates:
        #     pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
        #     latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
        #     return image, pred_x0_list, latents_list
        return image,init_latent
    
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

class MasaCtrl_NTI(MasaCtrl):
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        #
        uncond_embeddings_list = None,
        **kwds):
        DEVICE = self.model.unet.device
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )

        text_embeddings = self.model.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.model.unet.config.in_channels, height//8, width//8)
        if latents is None:
            latents = randn_tensor(latents_shape,device=DEVICE,dtype=self.model.unet.dtype,generator=None)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."
        init_latent = latents.clone()
        # unconditional embedding for classifier free guidance
                
        print("latents shape: ", latents.shape)
        # iterative sampling
        self.model.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.model.scheduler.timesteps))
        # latents_list = [latents]
        # pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            context = torch.cat([uncond_embeddings_list[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict tghe noise
            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=context).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            model_out = self.model.scheduler.step(noise_pred, t, latents,return_dict=True)
            latents, pred_x0 = model_out["prev_sample"],model_out['pred_original_sample']
            # latents_list.append(latents)
            # pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="np")
        # if return_intermediates:
        #     pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
        #     latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
        #     return image, pred_x0_list, latents_list
        return image,init_latent

class MasaCtrl_XL_NTI(MasaCtrl_XL):
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        #
        uncond_embeddings_list = None,
        ):
        DEVICE = self.model.unet.device
        do_classifier_free_guidance = guidance_scale>1.
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        # get text embeddings
        prompt_embeds,added_cond_kwargs = self.encode_prompt_xl(prompt,DEVICE,do_classifier_free_guidance,height,width,batch_size)
        # define initial latents
        latents_shape = (batch_size, self.model.unet.config.in_channels, height//8, width//8)
        if latents is None:
            latents = randn_tensor(latents_shape,device=DEVICE,dtype=self.model.unet.dtype,generator=None)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."
        init_latent = latents.clone()
        print("latents shape: ", latents.shape)
        # iterative sampling
        self.model.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.model.scheduler.timesteps))
        # latents_list = [latents]
        # pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            # predict tghe noise
            prompt_embeds[0:2] = uncond_embeddings_list[i].expand(*prompt_embeds[2:4].shape)
            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=prompt_embeds,
                                         added_cond_kwargs=added_cond_kwargs).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            model_out = self.model.scheduler.step(noise_pred, t, latents,return_dict=True)
            latents, pred_x0 = model_out["prev_sample"],model_out['pred_original_sample']
            # latents_list.append(latents)
            # pred_x0_list.append(pred_x0)
        image = self.latent2image(latents, return_type="np")
        # if return_intermediates:
        #     pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
        #     latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
        #     return image, pred_x0_list, latents_list
        return image,init_latent