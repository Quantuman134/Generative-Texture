import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import device
import matplotlib.pyplot as plt
import numpy as np
from Neural_Texture_Field import NeuralTextureField
from Img_Asset import PixelDataSet
import Img_Asset
def main():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.zeros((2, 1))
    print(a)
    a = torch.cat((a, b), 1)
    print(a)
    

def render_img(mlp, width, height):
    img_tensor = torch.empty((width, height, 3), dtype=torch.float32, device=device)
    for y in range(height):
        for x in range(width):
            x_temp = x / width
            y_temp = y / height
            coo = torch.tensor([x_temp, y_temp], dtype=torch.float32, device=device)
            coo = Img_Asset.tensor_transform(coo, mean=[0.5, 0.5], std=[0.5, 0.5])
            pixel = mlp(coo)
            img_tensor[x, y, :] = pixel

    return img_tensor

if __name__ == "__main__":
    main()