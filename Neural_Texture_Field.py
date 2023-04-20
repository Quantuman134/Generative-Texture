import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device
import utils

################################
#the position encoding layer
class PositionEncoding(nn.Module):
    def __init__(self, input_dim=2, upper_freq_index=10) -> None:
        super().__init__()
        self.upper_freq_index = upper_freq_index
        self.freq_indices = torch.tensor([i for i in range(upper_freq_index)], device=device).repeat(input_dim)
        self.mapping_size = input_dim*2*upper_freq_index + input_dim

    def forward(self, x):
        x_input = x
        if x.dim() == 1:
            x_input = x.unsqueeze(0)
        x = x.repeat(1, self.upper_freq_index)
        x = torch.mul(x, pow(2, self.freq_indices)) * torch.pi
        return torch.cat((x_input, torch.sin(x), torch.cos(x)), dim=1).squeeze()
        
################################
# mlp representing a texture image
# output: color clamped within [-1, 1]    
class NeuralTextureField(nn.Module):
    def __init__(self, width, depth, input_dim=2, pixel_dim=3, pe_enable=True) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.pe_enable = pe_enable
        layers = []
        
        if pe_enable:
            pe = PositionEncoding(input_dim=input_dim)
            layers.append(pe)
            layers.append(nn.ReLU())
            layers.append(nn.Linear(pe.mapping_size, width))
        else:
            layers.append(nn.Linear(input_dim, width))  
        layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, pixel_dim))
        self.base = nn.ModuleList(layers)

        print(self.base)
        self.to(device)
    
    def reset_weights(self):
        self.base[-1].weight.data.zero_()
        self.base[-1].bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = x

        #tanh clamp
        #colors = F.tanh(colors)
        return colors

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import Img_Asset
    from Img_Asset import PixelDataSet
    from torch.utils.data import DataLoader

    img_path = "./Assets/Images/test_image_16_16.png"
    pd = PixelDataSet(image_path=img_path)
    test_mlp = NeuralTextureField(width=512, depth=3, pe_enable=False)
    test_mlp.reset_weights()

    #training
    dataloader = DataLoader(pd, batch_size=16, shuffle=True)
    learning_rate = 0.0001
    epochs = 5000
    optimizer = torch.optim.Adam(test_mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (coos_gt, pixels_gt) in enumerate(dataloader):
            optimizer.zero_grad()
            pixels_pred = test_mlp(coos_gt)
            loss = criterion(pixels_pred, pixels_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/(batch_idx+1)}")

    torch.save(test_mlp.state_dict(), "ntf.pth")

    #rendering
    width = 16
    height = 16
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(width):
        for i in range(height):
            x = j / width
            y = i / height
            coo = torch.tensor([x, y], dtype=torch.float32, device=device)
            coo = Img_Asset.tensor_transform(coo, mean=[0.5, 0.5], std=[0.5, 0.5])
            pixel = ((test_mlp(coo) + 1) * 255/2).cpu().detach().numpy()
            img_array[i, j, :] = pixel

    plt.imshow(img_array)
    plt.show()



if __name__ == "__main__":
    main()