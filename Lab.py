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
    mlp = NeuralTextureField(width=512, depth=3, pe_enable=False)
    mlp.load_state_dict(torch.load("./ntf.pth"))
    #rendering
    width = 512
    height = 512
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(width):
        for i in range(height):
            x = j / width
            y = i / height
            coo = torch.tensor([x, y], dtype=torch.float32, device=device)
            coo = Img_Asset.tensor_transform(coo, mean=[0.5, 0.5], std=[0.5, 0.5])
            pixel = ((mlp(coo) + 1) * 255/2).cpu().data.numpy()
            img_array[i, j, :] = pixel

    plt.imshow(img_array)
    plt.show()

if __name__ == "__main__":
    main()