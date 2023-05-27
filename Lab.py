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
    mlp_path =  "./nth.pth"
    save_path = "./Experiments/Generative_Texture_1/test_experiment3"
    tex_net = NeuralTextureField(width=512, depth=3, pe_enable=True)
    tex_net.to(device)
    a = torch.sum(tex_net.render_img())
    a.backward()
    optimizer = torch.optim.Adam(tex_net.parameters(), lr = 0.001)
    optimizer.step()
    tex_net.net_save(save_path=mlp_path)
    tex_net_2 = torch.load(mlp_path)
    #tex_net.img_save(save_path=save_path)
    #tex_net.img_show()
    #tex_net.reset_weights()
    print(tex_net_2.pe_enable)

    #tex_net_2.reset_weights()

def main_2():
    #mlp_path =  "./Assets/Image_MLP/nascar2/nth.pth"
    mlp_path =  "./nth.pth"
    tex_net = torch.load(mlp_path)

if __name__ == "__main__":
    main_2()