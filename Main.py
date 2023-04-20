from Neural_Texture_Field import NeuralTextureField
from Img_Asset import PixelDataSet
import Img_Asset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from PIL import Image
from utils import device

def train_mlp(mlp, img_path):
    pd = PixelDataSet(img_path)
    pd.img.show()
    dataloader = DataLoader(pd, batch_size=64, shuffle=True)
    
    learning_rate = 0.001
    epochs = 400
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        if epoch > 0.5 * epochs:
            learning_rate = 0.001
        for batch_idx, (coos_gt, pixels_gt) in enumerate(dataloader):
            optimizer.zero_grad()
            pixels_pred = mlp(coos_gt)
            loss = criterion(pixels_pred, pixels_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss
        
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/(batch_idx+1)}")
    
    torch.save(mlp.state_dict(), "ntf.pth")

def render_img(mlp, width, height) -> Image:
    img = Image.new("RGB", (width, height))
    for y in range(img.height):
        for x in range(img.width):
            x_temp = x / img.width
            y_temp = y / img.height
            coo = torch.tensor([x_temp, y_temp], dtype=torch.float32, device=device)
            coo = Img_Asset.tensor_transform(coo, mean=[0.5, 0.5], std=[0.5, 0.5])
            pixel = (mlp(coo) + 1) * 255/2
            img.putpixel((x, y), tuple(map(int, pixel)))

    return img




def main():
    img_path = "./Assets/Images/test_image_16_16.png"
    mlp = NeuralTextureField(width=128, depth=2, pe_enable=True).cuda()
    mlp.to(device)
    mlp.reset_weights()
    
    train_mlp(mlp, img_path=img_path)
    img = render_img(mlp, width=16, height=16)
    img.show()


if __name__ == "__main__":
    main()


