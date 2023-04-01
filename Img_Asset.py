from PIL import Image
from torchvision import transforms

class ImgTensorLoader():
    def __init__(self, image_path) -> None:
        self.img = Image.open(image_path)
        self.img_tensor = transforms.ToTensor(self.img)