import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp
import argparse
import utils
from Neural_Texture_Field import NeuralTextureField
from Texture_Generator import TextureGenerator

def setup(rank, world_size):
# initial distributed group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    print(f"[INFO] Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # device set
    device = utils.cuda_set_device(rank)

    # seed set
    utils.seed_everything(0 + rank)

    # network configuration
    input_dim = 3
    brdf = True

    field_sample = False
    if input_dim == 3:
        field_sample = True

    # input configuration
    mesh_path = "./Assets/3D_Model/Pineapple/mesh.obj"
    text_prompt = "a golden metal pineapple"
    save_path = "./Experiments/Generative_Texture_MLP/Pineapple/test8"

    # diffusion model
    guidance_scale = 100

    # other configuration
    img_size = 512
    tex_size = 512

    diff_tex = NeuralTextureField(width=32, depth=2, pe_enable=True, input_dim=input_dim, brdf=brdf, device=device)
    ddp_diff_tex = ddp(diff_tex, device_ids=[device])

    texture_generator = TextureGenerator(mesh_path=mesh_path, diff_tex=ddp_diff_tex, is_latent=False, device=device, rank=rank)
    texture_generator.texture_train(text_prompt=text_prompt, guidance_scale=guidance_scale, lr=0.01, epochs=10, save_path=save_path,
                                    dist_range=[1.2, 1.2], elev_range=[0.0, 0.0], azim_range=[0.0, 0.0],
                                    info_update_period=1, render_light_enable=True, tex_size=tex_size, 
                                    rendered_img_size=img_size, annealation=True, field_sample=field_sample, brdf=brdf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', default=1, type=int)
    opt = parser.parse_args()

    world_size = opt.gpu_num

    mp.spawn(main, args=(world_size, ))


