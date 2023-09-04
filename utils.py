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

# output imgs size [B, C, 8H, 8W], output range [0, 1]
def decode_latents(latents):
    B, C, H, W = latents.size()
    rows = int(H/64)
    cols = int(W/64)

    latents = latents.chunk(rows, dim=2)
    latents = torch.cat(latents, dim=3)
    latents = latents.chunk(rows*cols, dim=3)
    latents = torch.cat(latents, dim=0)

    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.chunk(rows*cols, dim=0)
    imgs = torch.cat(imgs, dim=3)
    imgs = imgs.chunk(rows, dim=3)
    imgs = torch.cat(imgs, dim=2)
        
    return imgs

# input range [0, 1], output latent size [B, C, H/8, W/8], output range [-1, 1]
def encode_latents(imgs):
    B, C, H, W = imgs.size()
    rows = int(H/512)
    cols = int(W/512)
    imgs = 2 * imgs - 1

    imgs = imgs.chunk(rows, dim=2)
    imgs = torch.cat(imgs, dim=3)
    imgs = imgs.chunk(rows*cols, dim=3)
    imgs = torch.cat(imgs, dim=0)

    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215

    latents = latents.chunk(rows*cols, dim=0)
    latents = torch.cat(latents, dim=3)
    latents = latents.chunk(rows, dim=3)
    latents = torch.cat(latents, dim=2)

    return latents

def read_obj(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex)
    return torch.tensor(vertices, dtype=torch.float32)

def write_obj(file_path, vertices):
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

def vert_normalize(in_path, out_path):
    vert_tensor = read_obj(in_path)
    vert_max, _ = vert_tensor.max(dim=0)
    vert_min, _ = vert_tensor.min(dim=0)
    max_length = torch.max(vert_max - vert_min)
    center = (vert_max + vert_min) * 0.5
    vert_tensor = (vert_tensor - center)/max_length * 2
    write_obj(out_path, vert_tensor)
    print(f'Converted and saved {len(vert_tensor)} vertices to {output_file}')

if __name__ == '__main__':
    input_file = './Assets/3D_Model/Table/mesh.obj'
    output_file = './Assets/3D_Model/Table/new_mesh.obj'

    vert_normalize(input_file, output_file)
