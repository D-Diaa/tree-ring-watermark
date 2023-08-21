#!/usr/bin/env python3
import torch

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, DDIMScheduler

from _get_noise import get_noise
from _detect import detect

model_id = 'stabilityai/stable-diffusion-2-1'

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# IMPORTANT: We need to make sure to be able to use a normal diffusion pipeline so that people see 
# the tree-ring-watermark method as general enough
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
# or
# scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
pipe = pipe.to(device)

shape = (1, 4, 64, 64)
latents, w_key, w_mask = get_noise(shape)

watermarked_image = pipe(prompt="an astronaut", latents=latents).images[0]

is_watermarked = detect(watermarked_image, pipe, [w_key], [w_mask])
print(f'is_watermarked: {is_watermarked}')
