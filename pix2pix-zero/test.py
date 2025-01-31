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

from model.attention_control import restore_original_processors
from model.sd_utils import P2P_Zero,P2P_Zero_XL,P2P_Zero_NTI,P2P_Zero_XL_NTI
from utils.save_image import save_images,save_img
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

# for pix2pix-zero parameter
only_sample = False

# path param
exp_path = "./test_exp"
dataset_path = "./PIE"
inversion_path = None

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
    height=512
    width=512
    if inversion_type == "ddim":
        invertor = ddim_inversion()
        editor = P2P_Zero(pipe,num_inference_steps)
    elif inversion_type == "null-text":
        invertor = NTI()
        editor = P2P_Zero_NTI(pipe,num_inference_steps)
elif pipe.__class__.__name__ == "StableDiffusionXLPipeline":
    height=1024
    width=1024
    if inversion_type == "ddim":
        invertor = ddim_inversion_xl()
        editor = P2P_Zero_XL(pipe,num_inference_steps)
    elif inversion_type == "null-text":
        invertor = NTI_XL()
        editor = P2P_Zero_XL_NTI(pipe,num_inference_steps)

for category in [0,1,2,3,4,6,7,8,9]:
    dataset = PIE(dataset_path,inversion_path,category=category)
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False,pin_memory=True,num_workers=8)
    for _ , batch in enumerate(tqdm(dataloader)):
        image_path,source_prompt,target_prompt = batch
        image_path = image_path[0]
        out_path = os.path.join(exp_path,os.path.relpath(image_path.split(".")[0], os.path.join(dataset_path,"annotation_images")))
        os.makedirs(out_path,exist_ok=True)
        original_image = Image.open(image_path).convert("RGB").resize((height,width))
        original_image.save(os.path.join(out_path,"source.png"))
        latent = invertor.image2latent(model=pipe,image=original_image,device=device,dtype=dtype)
        if inversion_type == "ddim":
            latents,_ = invertor.ddim_inversion_loop(pipe,latent,source_prompt)
            init_latent = latents[-1]
            image_source,image_edit = editor(prompt = source_prompt+target_prompt,num_inference_steps=num_inference_steps,
                                            guidance_scale=guidance_scale,only_sample=only_sample,edit_dir=None,
                                            latents = init_latent)
            save_img(image_source,os.path.join(out_path,"inversion.png"))
            save_img(image_edit,os.path.join(out_path,"edit.png"))
        elif inversion_type == "null-text":
            latents, context = invertor.ddim_inversion_loop(pipe,latent,source_prompt)
            uncond_embeddings_list = invertor.null_optimization(pipe,latents,context,num_inner_steps,early_stop_epsilon,guidance_scale)
            init_latent = latents[-1]
            image_source,image_edit = editor(prompt = source_prompt+target_prompt,num_inference_steps=num_inference_steps,
                                            guidance_scale=guidance_scale,only_sample=only_sample,edit_dir=None,
                                            latents = init_latent,
                                            uncond_embeddings_list=uncond_embeddings_list)
            save_img(image_source,os.path.join(out_path,"inversion.png"))
            save_img(image_edit,os.path.join(out_path,"edit.png"))
        restore_original_processors(editor.model.unet, editor.original_processors)