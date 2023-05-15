import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device
import utils
import matplotlib.pyplot as plt

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
            #layers.append(nn.ReLU())
            layers.append(nn.Linear(pe.mapping_size, width))
        else:
            layers.append(nn.Linear(input_dim, width))  
        layers.append(nn.ReLU())
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, pixel_dim))
        self.base = nn.ModuleList(layers)
        self.reset_weights()

        print(self.base)
        self.to(device)
    
    def reset_weights(self):
        #self.base[-1].weight.data.zero_()
        #self.base[-1].bias.data.zero_()
        self.base[-1].weight.data = torch.randn_like(self.base[-1].weight.data)
        self.base[-1].bias.data = torch.randn_like(self.base[-1].bias.data)

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = x

        #tanh clamp
        colors = F.tanh(colors)
        return colors
    
    def render_img(self, width, height, validate=False):
        coo_tensor = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
        j = ((torch.arange(start=0, end=height, device=device) * 2 + 1.0) / height - 1.0).unsqueeze(0).transpose(0, 1).repeat(1, width)
        i = ((torch.arange(start=0, end=width, device=device) * 2 + 1.0) / width - 1.0).unsqueeze(0).repeat(height, 1)
        coo_tensor[:, :, 0] = j
        coo_tensor[:, :, 1] = i
        if validate:
            coo_tensor[:, :, 0] = j + 0.005 * torch.rand_like(j, device=device)
            coo_tensor[:, :, 1] = i + 0.005 * torch.rand_like(i, device=device)
        coo_tensor = coo_tensor.reshape(-1, 2)
        img_tensor = (self(coo_tensor) + 1) / 2 # color rgb [0, 1]
        #img_tensor: [1, 3, H, W]
        img_tensor = img_tensor.reshape(1, height, width, 3).permute(0, 3, 1, 2).contiguous()
        return img_tensor
    
    def img_show(self, width, height, validate=False):
        img_tensor = self.render_img(width, height, validate)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        plt.imshow(img_array)
        plt.show()
    
    def img_save(self, width, height, save_path, validate=False):
        img_tensor = self.render_img(width, height, validate)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        plt.imsave(save_path, img_array)
        


# test main function: train a mlp and render it
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
    learning_rate = 0.001
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

# test main function: train a mlp with entire image and render it
def main_3():
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    width = 512
    height = 512
    img_path = "./Assets/Images/Gaussian_Noise.png"
    save_path = "./Experiments/mlp_represented_image_training _entire_image/gaussian_noise/"
    img_train = Image.open(img_path).resize((width, height))
    img_train.save(save_path + "train.png")
    img_train = transforms.ToTensor()(img_train).to(device).unsqueeze(0)[:,0:3,:,:]
    print(img_train.size())

    test_mlp = NeuralTextureField(width=512, depth=3, pe_enable=True)
    learning_rate = 0.00001
    epochs = 10000
    optimizer = torch.optim.Adam(test_mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    total_loss = 0
    for epoch in range(epochs): 
        optimizer.zero_grad()     
        img_tensor = test_mlp.render_img(width, height, validate=True)
        loss = criterion(img_tensor, img_train)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss}")
            total_loss = 0
        if (epoch+1) % 500 == 0:
            test_mlp.img_save(width, height, save_path=save_path+f"ep{epoch+1}.png")


    torch.save(test_mlp.state_dict(), save_path+"nth.pth")
    test_mlp.img_show(width, height)
    test_mlp.img_save(width, height, save_path=(save_path + f"{width}_{height}.png"))





# test main function: use pretrained mlp and directly render the image of mlp
def main_2():
    import torch
    from Neural_Texture_Field import NeuralTextureField

    mlp = NeuralTextureField(width=512, depth=3, pe_enable=True)
    load_path = "./Experiments/mlp_represented_image_training _entire_image/test5_validate/nth.pth"
    mlp.load_state_dict(torch.load(load_path))
    #validation rendering
    mlp.img_show(512, 512, validate=True)


if __name__ == "__main__":
    main_3()