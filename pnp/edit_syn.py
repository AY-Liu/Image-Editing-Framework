# import package
import torch
from diffusers import StableDiffusionPipeline,DDIMScheduler,StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
from lightning.pytorch import seed_everything
import os 
import argparse
from utils.save_image import save_images,save_img
from model.sd_utils import PnP,PnP_XL
import random
from sd_mapping import sd_maps

# set general config
parser  = argparse.ArgumentParser("General config")
parser.add_argument("--sd_version",type=str,default="1.5")
parser.add_argument("--device",type=int,default=0)
parser.add_argument("--seed",type=int,default=74089447)
parser.add_argument("--source_prompt",type=str,default="A crisp, juicy green apple sits perched on a wooden table, its smooth surface glistening in the light")
parser.add_argument("--target_prompt",type=str,default="A crisp, juicy red apple sits perched on a wooden table, its smooth surface glistening in the light")
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

# for pnp parameter
pnp_attn_t = 1.
pnp_f_t = 1.
only_sample = False

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

# synthesis and edit
if pipe.__class__.__name__ == "StableDiffusionPipeline":
    editor = PnP(pipe,num_inference_steps)
elif pipe.__class__.__name__ == "StableDiffusionXLPipeline":
    editor = PnP_XL(pipe,num_inference_steps)

os.makedirs(out_path,exist_ok=True)
images = editor(prompt = source_prompt+target_prompt,num_inference_steps=num_inference_steps,guidance_scale=GUIDANCE_SCALE,
                            pnp_attn_t = pnp_attn_t,
                            pnp_f_t = pnp_f_t
                            )
save_img(images[0],os.path.join(out_path,"source.png"))
save_img(images[1],os.path.join(out_path,"edit.png"))
