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
    mlp_path =  "./nth.pth"
    save_path = "./Experiments/Generative_Texture_1/test_experiment/tex.png"
    tex_net = NeuralTextureField()
    torch.save(tex_net, mlp_path)
    tex_net_2 = torch.load(mlp_path)
    #tex_net.img_save(save_path=save_path)
    #tex_net.img_show()
    #tex_net.reset_weights()
    print(tex_net_2.pe_enable)

    tex_net_2.reset_weights()

if __name__ == "__main__":
    main()