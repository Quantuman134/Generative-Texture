import os
import torch
import random
import numpy as np
from diffusers import AutoencoderKL

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").to(device)

# output imgs size [B, C, H, W]
def decode_latents(latents):

    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
    return imgs