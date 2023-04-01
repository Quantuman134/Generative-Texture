from PIL import Image
from torchvision import transforms

#Load an image and convert it to a tensor for torch
class ImgTensorLoader():
    def __init__(self, image_path) -> None:
        self.img = Image.open(image_path)
        self.img_tensor = transforms.ToTensor()(self.img)

def main():
    ITL = ImgTensorLoader("./Assets/Images/test_image_16_16.png")
    ITL.img.show()

if __name__ == "__main__":
    main()