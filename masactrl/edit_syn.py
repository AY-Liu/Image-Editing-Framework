# import package
import torch
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline,DDIMScheduler,StableDiffusionXLImg2ImgPipeline
from lightning.pytorch import seed_everything
import os 
from model.sd_utils import MasaCtrl,MasaCtrl_XL
from model.attention_base import AttentionStore,AttentionBase
from model.attention_control import MutualSelfAttentionControl
from utils.save_image import save_images,save_img
from model.register import regiter_attention_editor_diffusers
import argparse
import random
from sd_mapping import sd_maps

# set general config
parser  = argparse.ArgumentParser("General config")
parser.add_argument("--sd_version",type=str,default="1.5")
parser.add_argument("--device",type=int,default=0)
parser.add_argument("--seed",type=int,default=8888) # random.randint(0,1e8)
parser.add_argument("--source_prompt",type=str,default="A standing dog on the grass field")
parser.add_argument("--target_prompt",type=str,default="A running dog on the grass field")
args = parser.parse_args()

# SD version
sd_version = args.sd_version
model_key = sd_maps[sd_version]
refiner_key = "stabilityai/stable-diffusion-xl-refiner-1.0"

# set up
device = torch.device("cuda:{}".format(args.device))
seed = args.seed
seed_everything(seed)
source_prompt = [args.source_prompt]
target_prompt =  [args.target_prompt]
num_inference_steps = 50
GUIDANCE_SCALE = 7.5
LOW_RESOURCE = False
dtype = torch.float32
out_path = "./exp"
# for masactrl parameter
STEP = 4
# LAYPER = 10 modify at following codes

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

if pipe.__class__.__name__ == "StableDiffusionPipeline":
    model_type = "SD"
    height=512
    width=512
    editor = MasaCtrl(pipe,num_inference_steps)
    LAYPER = 10
elif pipe.__class__.__name__ == "StableDiffusionXLPipeline":
    model_type = "SDXL"
    height=1024
    width=1024
    editor = MasaCtrl_XL(pipe,num_inference_steps)
    LAYPER = 54

# synthesis
latent = None
controller = AttentionBase()
regiter_attention_editor_diffusers(editor.model, controller)
image,init_latent = editor(prompt = source_prompt,guidance_scale=GUIDANCE_SCALE,num_inference_steps=num_inference_steps,height=height,width=width)
save_img(image,os.path.join(out_path,"source.png"))
# edit
init_latent = torch.cat([init_latent,init_latent])
controller = MutualSelfAttentionControl(STEP, LAYPER, model_type=model_type)
regiter_attention_editor_diffusers(editor.model, controller)
image_masactrl , _ = editor(prompt = source_prompt+target_prompt, latents=init_latent, guidance_scale=GUIDANCE_SCALE,num_inference_steps=num_inference_steps,height=height,width=width)
save_img(image_masactrl[1],os.path.join(out_path,"edit.png"))
