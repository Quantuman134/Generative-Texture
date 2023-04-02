from PIL import Image
import torch
from torch.utils.data import Dataset

#Load an image and convert it to a tensor for torch
class PixelDataSet(Dataset):
    def __init__(self, image_path) -> None:
        super().__init__()
        self.img = Image.open(image_path)
        self.width, self.height = self.img.size

    

    def __getitem__(self, index):
        x = (index % self.width) / self.width
        y = (index // self.width) / self.height
        pixel = self.img.getpixel((index % self.width, index // self.width))
        pixel = torch.Tensor(pixel)
        coo = torch.Tensor([x, y])

        return coo_transform(coo), pixel
    
    def __len__(self):
        return self.width * self.height

def coo_transform(coo):
    mean = torch.Tensor([0.5, 0.5])
    std = torch.Tensor([0.5, 0.5])
    coo_trans = (coo - mean) / std
    return coo_trans

def main():
    PD = PixelDataSet("./Assets/Images/test_image_16_16.png")
    PD.img.show()

if __name__ == "__main__":
    main()