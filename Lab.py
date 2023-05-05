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
    x = torch.tensor([[1, 2, 3], [1, 2, 3]], device=device)
    y = torch.tensor([[4], [5]], device=device)
    index = (x == 2).nonzero(as_tuple=False)
    print(x[1, :] == 2)
    x[index[:, 0], index[:, 1]] = y[0, :]
    print(x)

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