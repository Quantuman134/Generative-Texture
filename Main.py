from Neural_Texture_Field import NeuralTextureField
from Img_Asset import PixelDataSet
import Img_Asset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from Stable_Diffusion import StableDiffusion
from PIL import Image
from utils import device
import utils
import time
import matplotlib.pyplot as plt

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

def render_img(mlp, width, height):
    img_tensor = torch.empty((width, height, 3), dtype=torch.float32, device=device)
    for i in range(height):
        for j in range(width):
            x = j / width
            y = i / height
            coo = torch.tensor([x, y], dtype=torch.float32, device=device)
            coo = Img_Asset.tensor_transform(coo, mean=[0.5, 0.5], std=[0.5, 0.5])
            pixel = mlp(coo)
            img_tensor[i, j, :] = pixel

    img_tensor = img_tensor.reshape(1, height, width, 3).permute(0, 3, 1, 2).contiguous()
    return img_tensor

def train_mlp_sd(mlp, epochs, lr, text_prompt):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    guidance = StableDiffusion(device=device)
    text_embeddings = guidance.get_text_embeds(text_prompt, '')
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    print(f"[INFO] traning starts")
    total_loss = 0
    for epoch in range(epochs):
        start_t = time.time()

        #render current image (test resolution: 16x16)
        img_pred = render_img(mlp, 16, 16)
        optimizer.zero_grad()
        loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings)
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        end_t = time.time()
        if True:
            print(f"[INFO] epoch {epoch} takes {(end_t - start_t):.4f} seconds.")




def main():
    seed = 0
    utils.seed_everything(seed)
    mlp = NeuralTextureField(width=512, depth=3, pe_enable=False)
    mlp.load_state_dict(torch.load("./ntf.pth"))
    mlp.to(device)

    epochs = 100
    lr = 0.001
    text_prompt = "a pixel style image of an orange cat head."
    #text_prompt = "an apple."
    train_mlp_sd(mlp, epochs, lr, text_prompt)
    
    img = render_img(mlp, width=16, height=16)
    img = img.reshape(3, 16, 16).permute(1, 2, 0).contiguous()
    img = (img + 1)/2
    img_array = img.cpu().data.numpy()
    plt.imshow(img_array)
    plt.show()


if __name__ == "__main__":
    main()


