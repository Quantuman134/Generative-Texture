from typing import Iterator
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import device
import matplotlib.pyplot as plt
import numpy as np

#size: width x height
# output: color, ideal range is [-1, 1] 
# input coordinate: the dimension of input coordinate is 2, and range of value is [-1, 1]. The positive direction
# of x and y are right and up respectively
# current texture sample is bilinear 

class DiffTexture(nn.Module):
    def __init__(self, size=(512, 512)) -> None:
        super().__init__()
        self.width = size[0]
        self.height = size[1]
        self.set_gaussian()

    def set_gaussian(self, mean=0, sig=1):
        self.texture = torch.nn.Parameter(torch.randn((self.width, self.height, 3), dtype=torch.float32, device=device, requires_grad=True) * sig + mean)

    def set_image(self, img_tensor):
        #img_tensor size: [W, H, 3], color value [0, 1]
        W, H, C = img_tensor.size()
        img_tensor = img_tensor.permute(2, 0, 1).reshape(1, C, W, H) * 2 - 1.0
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(1, C, self.width, self.height))
        self.texture = torch.nn.Parameter(img_tensor.sequeeze().permute(1, 2, 0))
        self.texture.device = device
        self.texture.requires_grad = True
        

    def forward(self, uvs):
        if uvs.dim() == 1:
            color = self.texture_sample(uvs)
            return color
        else:
            colors = self.texture_batch_sample(uvs)
            return colors
    
    def texture_sample(self, uv):
        uv = (uv + 1)/2
        uv[0] *= (self.width - 1)
        uv[1] *= (self.height - 1)
        u_0 = uv[0].floor().type(torch.int32)
        u_1 = uv[0].ceil().type(torch.int32)
        v_0 = uv[1].floor().type(torch.int32)
        v_1 = uv[1].ceil().type(torch.int32)
        a = uv[0] - u_0
        b = uv[1] - v_0
        color = (self.texture[u_0, v_0] * a + self.texture[u_1, v_0] * (1 - a)) * b \
        + (self.texture[u_0, v_1] * a + self.texture[u_1, v_1] * (1 - a)) * (1 - b)
        return color
    
    def texture_batch_sample(self, uvs):
        uvs = (uvs + 1)/2
        uvs[:, 0] *= (self.width - 1)
        uvs[:, 1] *= (self.height - 1)
        us_0 = uvs[:, 0].floor().type(torch.int32)
        us_1 = uvs[:, 0].ceil().type(torch.int32)
        vs_0 = uvs[:, 1].floor().type(torch.int32)
        vs_1 = uvs[:, 1].ceil().type(torch.int32)
        a = (uvs[:, 0] - us_0).reshape(-1, 1)
        b = (uvs[:, 1] - vs_0).reshape(-1, 1)
        colors = (self.texture[us_0, vs_0] * a + self.texture[us_1, vs_0] * (1 - a)) * b \
        + (self.texture[us_0, vs_1] * a + self.texture[us_1, vs_1] * (1 - a)) * (1 - b)
        return colors
    
    def render_img(self, width=512, height=512):
        coo_tensor = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
        j = torch.arange(start=0, end=height, device=device).unsqueeze(0).transpose(0, 1).repeat(1, width)
        i = torch.arange(start=0, end=height, device=device).unsqueeze(0).repeat(height, 1)
        x = (j * 2 + 1.0) / height - 1.0
        y = (i * 2 + 1.0) / width - 1.0
        coo_tensor[:, :, 0] = x
        coo_tensor[:, :, 1] = y
        coo_tensor = coo_tensor.reshape(-1, 2) # [H*W, 2]
        img_tensor = (self(coo_tensor) + 1) / 2 # color rgb [0, 1]
        img_tensor = img_tensor.reshape(width, height, 3)
        #img_tensor: [1, 3, H, W]
        img_tensor = img_tensor.reshape(1, height, width, 3).permute(0, 3, 1, 2).contiguous()
        return img_tensor
    
    def img_show(self, width=512, height=512):
        img_tensor = self.render_img(width, height)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imshow(img_array)
        plt.show()
    
    def img_save(self, save_path, width=512, height=512):
        img_tensor = self.render_img(width, height)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imsave(save_path, img_array)


def main():
    texture = DiffTexture()
    texture.img_show(512, 512)
    

if __name__ == '__main__':
    main()
