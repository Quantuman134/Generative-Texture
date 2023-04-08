from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import device

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
        pixel = torch.tensor(pixel, dtype=torch.float32, device=device)
        
        coo = torch.tensor([x, y], dtype=torch.float32, device=device)

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
    PD.img.show()

if __name__ == "__main__":
    main()