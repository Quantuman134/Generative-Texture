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
    img_path = "./Assets/Images/test_image_16_16.png"
    pd = PixelDataSet(image_path=img_path)
    #rendering
    '''
    width = 512
    height = 512
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(width):
        for i in range(height):
            if j/32 >= 15 or i/32 >= 15:
                x = j//32
                y = i//32
                pixel = pd.img.getpixel((x,y))
                img_array[i, j, :] = np.asarray(pixel)
            else:
                x = j//32
                y = i//32
                x_f = j/32 - j//32
                y_f = i/32 - i//32
                pixel1 = pd.img.getpixel((x, y))
                pixel1 = np.asarray(pixel1)
                pixel2 = pd.img.getpixel((x+1, y))
                pixel2 = np.asarray(pixel2)
                pixel3 = pd.img.getpixel((x, y+1))
                pixel3 = np.asarray(pixel3)
                pixel4 = pd.img.getpixel((x+1, y+1))
                pixel4 = np.asarray(pixel4)
                pixel = (pixel1 * x_f + pixel2 * (1 - x_f)) * y_f + (pixel3 * x_f + pixel4 * (1 - x_f)) * (1 - y_f) 
                img_array[i, j, :] = pixel

    '''
    width = 512
    height = 512
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(width):
        for i in range(height):
            x = j/32
            y = i/32
            pixel = np.asarray(pd.img.getpixel((x,y)))
            img_array[i, j, :] = pixel

    plt.imshow(img_array)
    plt.show()

if __name__ == "__main__":
    main()