"""
inpainting_utils.py

Handles:
- LaMa inpainting
- Stable Diffusion inpainting
"""

import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from diffusers import StableDiffusionInpaintPipeline


# ------------------------------
# LaMa Inpainting
# ------------------------------

def lama_inpaint(image_path: str, mask_path: str, output_path: str):
    """Run LaMa inpainting."""
    lama = SimpleLama()
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert("L")

    result = lama(image, mask)
    result.save(output_path)


# ------------------------------
# Stable Diffusion Inpainting
# ------------------------------

def sd_inpaint(image_path: str, mask_path: str, prompt: str, output_path: str):
    """Run Stable Diffusion inpainting."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    result.save(output_path)
