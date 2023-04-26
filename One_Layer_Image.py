import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import device 
from PIL import Image
import numpy as np

# one layer represents an image, use neural layer to be compatible with learning pipeline
class OneLayerImage(nn.Module):
    def __init__(self, img=None, width=512, height=512, color_dim=3) -> None:
        super().__init__()
        if img is not None:
            self.img = img.resize((width, height))
        else:
            self.img = Image.effect_noise((width * 3, height), 255/2)
        self.width = width
        self.height = height
        self.color_dim = color_dim
        layers = []
        output_size = width * height * color_dim
        linear = nn.Linear(1, output_size)
        linear.bias.data = torch.zeros(output_size)

        #initialize weight of each linear neural as color value of the image
        img_tensor = transforms.ToTensor()(self.img)
        weight = torch.reshape(img_tensor, linear.weight.data.size())
        linear.weight.data = weight * 2 - 1 # remap values to [-1, 1con]
        layers.append(linear)

        self.layers = nn.ModuleList(layers)
        print(self.layers)
        self.to(device=device)
    
    #if you want to output the img, input the x=1
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        #x = (x + 1) / 2
        img_tensor = torch.reshape(x, (1, self.color_dim, self.height, self.width))
        return img_tensor
    
    # return a rendered img with numpy format (use matplotlib can show the image)
    def render_img(self):
        img_tensor = self(torch.tensor([1], dtype=torch.float32, device=device))
        img_tensor = img_tensor.reshape(self.color_dim, self.height, self.width).permute(1, 2, 0).contiguous()
        img_tensor = (img_tensor + 1) / 2
        img = img_tensor.cpu().data.numpy()
        return img

def main():
    import matplotlib.pyplot as plt
    img_path = "./Assets/Images/test_image_216_233.jpeg"
    img = Image.open(img_path)
    #oli = OneLayerImage(img=img)
    oli = OneLayerImage()
    img_test = oli.render_img()
    plt.imshow(img_test)
    plt.show()

if __name__ == "__main__":
    main()