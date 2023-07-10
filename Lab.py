import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import device
import matplotlib.pyplot as plt
import numpy as np
from Neural_Texture_Field import NeuralTextureField
import Neural_Texture_Field 
from Img_Asset import PixelDataSet
import Img_Asset
import utils

def main():
    latent = torch.ones((1, 4, 64, 64), device=device)
    latent[:, :, 0:31, 0:31] = 0
    image_tensor = utils.decode_latents(latent)
    image_array = image_tensor[0, :, :, :].permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_array)
    plt.show()
def main_2():
    #mlp_path =  "./Assets/Image_MLP/nascar2/nth.pth"
    mlp_path =  "./nth.pth"
    tex_net = torch.load(mlp_path)


if __name__ == "__main__":
    main()