import tomesd
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
import time

pipeline = StableDiffusionXLPipeline.from_pretrained("/workspace/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")

batch_size = 4
resolution = 896
trials = 2

tt = 0
for _ in range(trials):
    st = time.time()
    pipeline(prompt="Laundromat Stories: Inside a laundromat on a rainy day. People load clothes into washing machines and read magazines while waiting. Charcoal drawing, chiaroscuro, dramatic \
 lighting from overhead fluorescents.", num_inference_steps=20, num_images_per_prompt=batch_size, width = resolution, height=resolution)
    tt += time.time() - st
print("SDXL no tomesd: avg time", tt/trials)

pipeline = tomesd.apply_patch(pipeline, ratio=0.75, max_downsample = 4)

tt = 0
for _ in range(trials):
    st = time.time()
    pipeline(prompt="Laundromat Stories: Inside a laundromat on a rainy day. People load clothes into washing machines and read magazines while waiting. Charcoal drawing, chiaroscuro, dramatic \
 lighting from overhead fluorescents.", num_inference_steps=20, num_images_per_prompt=batch_size, width = resolution, height=resolution)
    tt += time.time() - st
print("SDXL w/ tomesd: avg time", tt/trials)