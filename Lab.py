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
    latent = torch.ones((1, 4, 128, 128), device=device)
    latent[:, :, 0:64, 0:64] = 0
    image_tensor = utils.decode_latents(latent)
    image_array = image_tensor[0, :, :, :].permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_array)
    plt.show()
def main_2():
    a = torch.tensor([1, 2, 3, 4, 5, 6])
    b = a
    print(a)
    print(b)
    b[3] = 0
    print(a)
    print(b)

if __name__ == "__main__":
    main_2()