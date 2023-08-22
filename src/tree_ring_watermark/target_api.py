#!/usr/bin/env python3
import torch

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler

from _get_noise import get_noise
from _detect import detect

model_id = 'runwayml/stable-diffusion-v1-5'

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
pipe = pipe.to(device)

# initialize
shape = (1, 4, 64, 64)
init_latents = torch.randn(shape)

wm_latents, w_key, w_mask = get_noise(shape, init_latents=init_latents)

nonwatermarked_image = pipe(prompt="an astronaut", latents=init_latents).images[0]
watermarked_image = pipe(prompt="an astronaut", latents=wm_latents).images[0]

is_watermarked = detect(watermarked_image, pipe, [w_key], [w_mask])
print(f'is_watermarked: {is_watermarked}')
is_watermarked = detect(nonwatermarked_image, pipe, [w_key], [w_mask])
print(f'is_watermarked: {is_watermarked}')
