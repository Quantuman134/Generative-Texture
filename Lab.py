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
def main():
    a = torch.tensor([0, 1, 2], dtype=torch.float32, requires_grad=True)
    loss = a.sum()
    loss.backward()
    print(a.grad)

def main_2():
    #mlp_path =  "./Assets/Image_MLP/nascar2/nth.pth"
    mlp_path =  "./nth.pth"
    tex_net = torch.load(mlp_path)

if __name__ == "__main__":
    main()