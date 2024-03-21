# import package
import torch
from diffusers import StableDiffusionPipeline,DDIMScheduler,StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
from lightning.pytorch import seed_everything
import os
import argparse
import random
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.attention_base import AttentionStore,EmptyControl
from model.attention_control import AttentionReplace,AttentionRefine
from model.register import unregister_attention_control
from utils.save_image import save_images,save_img
from model.sd_utils import P2P,P2P_NTI,P2P_XL, P2P_XL_NTI
from inversion.ddim import ddim_inversion,ddim_inversion_xl
from inversion.nti import NTI,NTI_XL
from dataset.pie import PIE
from sd_mapping import sd_maps

# SD version
sd_version = "1.5"
model_key = sd_maps[sd_version]
refiner_key = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Basic param
device = torch.device("cuda:1")
seed = 42
seed_everything(seed)
guidance_scale = 7.5
dtype = torch.float32
num_inference_steps = 50
inversion_type = "ddim"

# path param
exp_path = "./test_exp"
dataset_path = "./PIE"
inversion_path = None

# p2p param
LOW_RESOURCE = False
cross_replace_steps = 0.8
self_replace_steps = 0.6

# for nti param
num_inner_steps = 10
early_stop_epsilon = 1e-5

# load model
scheduler_config = {
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": False,
  "num_train_timesteps": 1000,
  "set_alpha_to_one": False,
  "skip_prk_steps": True,
  "steps_offset": 1,
  "trained_betas": None,
  "use_karras_sigmas": False
}
scheduler = DDIMScheduler.from_config(scheduler_config)
if sd_version in ["2.1","1.5","1.4"]: # non-xl version
    pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=dtype,scheduler=scheduler)
    pipe = pipe.to(device)
elif sd_version in ["xl-base"]: # xl version
    pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype, use_safetensors=True, variant="fp16",scheduler=scheduler)
    pipe = pipe.to(device)
elif sd_version in ["xl-refiner"]: # refine version
    pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype, use_safetensors=True, variant="fp16",scheduler=scheduler)
    pipe = pipe.to(device)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_key,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
        force_zeros_for_empty_prompt=False
        )
    refiner = refiner.to(device)
elif sd_version in ["animagineXL"]: # one safetensors based on XL
    pipe = StableDiffusionXLPipeline.from_single_file(model_key, torch_dtype=dtype, use_safetensors=True, variant="fp16",scheduler=scheduler)
    pipe = pipe.to(device)
elif sd_version in ["ghostv2","cf","anythingv4-5"]:# one safetensors based on non-xl 
    pipe = StableDiffusionPipeline.from_single_file(model_key, torch_dtype=dtype,load_safety_checker=False)
    pipe.scheduler = scheduler
    pipe = pipe.to(device)
else:
    raise ValueError("please use the right sd_version")

# editor
if pipe.__class__.__name__ == "StableDiffusionPipeline":
    if inversion_type == "ddim":
        editor = P2P(model=pipe,num_inference_steps=num_inference_steps)
        invertor = ddim_inversion()
    elif inversion_type == "null-text":
        editor = P2P_NTI(model=pipe,num_inference_steps=num_inference_steps)
        invertor = NTI()
    height = 512
    width = 512

elif pipe.__class__.__name__ == "StableDiffusionXLPipeline":
    if inversion_type == "ddim":
        editor = P2P_XL(model=pipe,num_inference_steps=num_inference_steps)
        invertor = ddim_inversion_xl()
    elif inversion_type == "null-text":
        editor = P2P_XL_NTI(model=pipe,num_inference_steps=num_inference_steps)
        invertor = NTI_XL()
    height = 1024
    width = 1024

for category in [0,1,2,3,4,6,7,8,9]:
    dataset = PIE(dataset_path,inversion_path,category=category)
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False,pin_memory=True,num_workers=8)
    for _ , batch in enumerate(tqdm(dataloader)):
        image_path,source_prompt,target_prompt = batch
        image_path = image_path[0]
        if len(source_prompt[0].split(" ")) == len(target_prompt[0].split(" ")):
            edit_type = "replace"
        else:
            edit_type = "refine"
        original_image = Image.open(image_path).convert("RGB").resize((height,width))
        out_path = os.path.join(exp_path,os.path.relpath(image_path.split(".")[0], os.path.join(dataset_path,"annotation_images")))
        os.makedirs(out_path,exist_ok=True)
        original_image.save(os.path.join(out_path,"source.png"))
        latent = invertor.image2latent(model=pipe,image=original_image,device=device,dtype=dtype)
        if inversion_type == "ddim":
            latents,_ = invertor.ddim_inversion_loop(pipe,latent,source_prompt)
            latent = latents[-1]
            # edit
            if edit_type == "replace":
                controller = AttentionReplace(prompts = source_prompt+target_prompt, tokenizer =pipe.tokenizer ,
                                            num_steps= num_inference_steps, cross_replace_steps=cross_replace_steps, 
                                            self_replace_steps=self_replace_steps,dtype=dtype,device=device)
                image, latent = editor.text2image_ldm_stable(pipe, source_prompt+target_prompt, controller, latent=latent, 
                                    num_inference_steps=num_inference_steps, 
                                    guidance_scale=guidance_scale, low_resource=LOW_RESOURCE)
                save_img(image[0],os.path.join(out_path,"inversion.png"))
                save_img(image[1],os.path.join(out_path,"edit.png"))
            elif edit_type == "refine":
                controller = AttentionRefine(prompts = source_prompt+target_prompt, tokenizer =pipe.tokenizer ,
                                            num_steps= num_inference_steps, cross_replace_steps=cross_replace_steps, 
                                            self_replace_steps=self_replace_steps,device=device)
                image, latent = editor.text2image_ldm_stable(pipe, source_prompt+target_prompt, controller, latent=latent, 
                                    num_inference_steps=num_inference_steps, 
                                    guidance_scale=guidance_scale, low_resource=LOW_RESOURCE)
                save_img(image[0],os.path.join(out_path,"inversion.png"))
                save_img(image[1],os.path.join(out_path,"edit.png"))
            else:
                raise ValueError("Please choose right eidt type")
        elif inversion_type == "null-text":
            latents,context = invertor.ddim_inversion_loop(pipe,latent,source_prompt)
            latent = latents[-1]
            uncond_embeddings_list = invertor.null_optimization(pipe,latents,context,num_inner_steps,early_stop_epsilon,guidance_scale)
            # edit
            if edit_type == "replace":
                controller = AttentionReplace(prompts = source_prompt+target_prompt, tokenizer =pipe.tokenizer ,
                                            num_steps= num_inference_steps, cross_replace_steps=cross_replace_steps, 
                                            self_replace_steps=self_replace_steps,dtype=dtype,device=device)
                image, latent = editor.text2image_ldm_stable(pipe, source_prompt+target_prompt, controller, latent=latent, 
                                    num_inference_steps=num_inference_steps, 
                                    guidance_scale=guidance_scale, low_resource=LOW_RESOURCE,uncond_embeddings_list=uncond_embeddings_list)
                save_img(image[0],os.path.join(out_path,"inversion.png"))
                save_img(image[1],os.path.join(out_path,"edit.png"))
            elif edit_type == "refine":
                controller = AttentionRefine(prompts = source_prompt+target_prompt, tokenizer =pipe.tokenizer ,
                                            num_steps= num_inference_steps, cross_replace_steps=cross_replace_steps, 
                                            self_replace_steps=self_replace_steps,device=device)
                image, latent = editor.text2image_ldm_stable(pipe, source_prompt+target_prompt, controller, latent=latent, 
                                    num_inference_steps=num_inference_steps, 
                                    guidance_scale=guidance_scale, low_resource=LOW_RESOURCE,uncond_embeddings_list=uncond_embeddings_list)
                save_img(image[0],os.path.join(out_path,"inversion.png"))
                save_img(image[1],os.path.join(out_path,"edit.png"))
            else:
                raise ValueError("Please choose right eidt type")
        else:
            raise ValueError("Please choose right inversion type")
        controller.reset()
        unregister_attention_control(pipe,controller)
