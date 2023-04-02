from Neural_Texture_Field import NeuralTextureField
from Img_Asset import PixelDataSet
import Img_Asset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from PIL import Image

def train_mlp(mlp):
    img_path = "./Assets/Images/test_image_16_16.png"
    pd = PixelDataSet(img_path)
    pd.img.show()
    dataloader = DataLoader(pd, batch_size=64, shuffle=True)
    
    learning_rate = 0.001
    epochs = 1000
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (coos_gt, pixels_gt) in enumerate(dataloader):
            optimizer.zero_grad()
            pixels_pred = mlp(coos_gt)
            loss = criterion(pixels_pred, pixels_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/(batch_idx+1)}")
    
    torch.save(mlp.state_dict(), "ntf.pth")

def render_img(mlp) -> Image:
    width, height = 16, 16
    img = Image.new("RGB", (width, height))
    for y in range(img.height):
        for x in range(img.width):
            x_temp = x / img.width
            y_temp = y / img.height
            coo = torch.Tensor([x_temp, y_temp])
            coo = Img_Asset.coo_transform(coo)
            pixel = mlp(coo) * 255
            img.putpixel((x, y), tuple(map(int, pixel)))

    return img




def main():
    mlp = NeuralTextureField(width=256, depth=4)
    train_mlp(mlp)
    img = render_img(mlp)
    img.show()


if __name__ == "__main__":
    main()


