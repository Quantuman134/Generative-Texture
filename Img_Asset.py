from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import device
import random

#Load an image and convert it to a tensor for torch
class PixelDataSet(Dataset):
    def __init__(self, image_path) -> None:
        super().__init__()
        self.img = Image.open(image_path)
        self.width, self.height = self.img.size

    def __getitem__(self, index):
        x = index % self.width
        y = index // self.width
        #x += random.random()
        #y += random.random()
        if x > self.width - 1:
            x = self.width - 1
        if y > self.height - 1:
            y = self.height - 1
        pixel = self.img.getpixel((x, y))
        pixel = torch.tensor(pixel, dtype=torch.float32, device=device)
        coo = torch.tensor([x/self.width, y/self.height], dtype=torch.float32, device=device)
        '''
        if x == self.width - 1 or y == self.height - 1 :
            pixel = self.img.getpixel((x, y))
            pixel = torch.tensor(pixel, dtype=torch.float32, device=device)
            coo = torch.tensor([x/self.width, y/self.height], dtype=torch.float32, device=device)
        else:
            pixel1 = self.img.getpixel((x, y))
            pixel1 = torch.tensor(pixel1, dtype=torch.float32, device=device)
            pixel2 = self.img.getpixel((x+1, y))
            pixel2 = torch.tensor(pixel2, dtype=torch.float32, device=device)
            pixel3 = self.img.getpixel((x, y+1))
            pixel3 = torch.tensor(pixel3, dtype=torch.float32, device=device)
            pixel4 = self.img.getpixel((x+1, y+1))
            pixel4 = torch.tensor(pixel4, dtype=torch.float32, device=device)
            rand_cor = torch.rand(2, device=device)
            pixel = (pixel1 * rand_cor[0] + pixel2 * (1 - rand_cor[0])) * rand_cor[1] + (pixel3 * rand_cor[0] + pixel4 * (1 - rand_cor[0])) * (1 - rand_cor[1]) 
            coo = torch.tensor([(x+rand_cor[0])/self.width, (y+rand_cor[1])/self.height], dtype=torch.float32, device=device)
        '''
        return tensor_transform(coo, mean=[0.5, 0.5], std=[0.5, 0.5]), \
        tensor_transform(pixel, mean=[255/2, 255/2, 255/2], std=[255/2, 255/2, 255/2])
    
    def __len__(self):
        return self.width * self.height

def tensor_transform(ts, mean, std):
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    std = torch.tensor(std, dtype=torch.float32, device=device)
    ts_trans = (ts - mean) / std
    return ts_trans

def main():
    PD = PixelDataSet("./Assets/Images/test_image_216_233.jpeg")
    imgplot = plt.imshow(np.asarray(PD.img))
    plt.show()

if __name__ == "__main__":
    main()