from tomesd.patch import apply_patch 
from diffusers import DiffusionPipeline 
import torch 



# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "/workspace/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

apply_patch(base, ratio=0.5)


print(base)


# refiner = DiffusionPipeline.from_pretrained(
#     "/workspace/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# refiner.to("cuda")

# # Define how many steps and what % of steps to be run on each experts (80/20) here
# n_steps = 39
# high_noise_frac = 0.7

# prompt = "A majestic lion jumping from a big stone at night"
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
# torch.cuda.synchronize()
# curr_time = starter.elapsed_time(ender)
# print("Elapsed time:{:>9.2f} ms".format(curr_time))

