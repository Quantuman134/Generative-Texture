import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

proxy = 'http://127.0.0.1:7890'

os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

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
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank=0, world_size=1):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    print(f"world_size:{world_size}")
    print(f"rank:{rank}")

    print(f"[INFO] Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # device set
    device = utils.cuda_set_device(rank)

    # seed set
    utils.seed_everything(42)

    # network configuration
    input_dim = 3
    brdf = True

    field_sample = False
    if input_dim == 3:
        field_sample = True

    # input configuration
    mesh_path = "./Assets/3D_Model/Pineapple/Pineapple.obj"
    text_prompt = "a pineapple"
    save_path = "./Experiments/Generative_Texture_MLP/Pineapple/test9"

    # diffusion model
    guidance_scale = 50

    # other configuration
    img_size = 512
    tex_size = 512

    diff_tex = NeuralTextureField(width=32, depth=2, pe_enable=True, input_dim=input_dim, brdf=brdf, device=device)
    ddp_diff_tex = ddp(diff_tex, device_ids=[device])

    texture_generator = TextureGenerator(mesh_path=mesh_path, diff_tex=ddp_diff_tex, is_latent=False, device=device, rank=rank)
    texture_generator.texture_train(text_prompt=text_prompt, guidance_scale=guidance_scale, lr=0.01, epochs=4000, save_path=save_path,
                                    dist_range=[1.2, 1.2], elev_range=[-10.0, 45.0], azim_range=[0.0, 360.0],
                                    info_update_period=200, render_light_enable=True, tex_size=tex_size, 
                                    rendered_img_size=img_size, annealation=False, field_sample=field_sample, brdf=brdf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument("--local-rank", type=int, default=0)
    opt = parser.parse_args()

    world_size = opt.gpu_num

    main()
