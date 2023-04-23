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
    mlp = NeuralTextureField(512, 3, pe_enable=False)
    mlp.load_state_dict(torch.load("./ntf.pth"))
    img_tensor = render_img(mlp, 16, 16)
    print(f"img_size: {img_tensor.size()}")
    img_tensor = img_tensor.reshape(1, 16, 16, 3).permute(0, 3, 1, 2).contiguous()
    print(f"img_size: {img_tensor.size()}")


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