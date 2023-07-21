import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device
import matplotlib.pyplot as plt
import numpy as np

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
# output: color, ideal range is [0, 1] 
# input coordinate: the dimension of input coordinate is 2, and range of value is [-1, 1]. The positive direction
# of x and y are right and up respectively  
class NeuralTextureField(nn.Module):
    def __init__(self, width=512, depth=3, input_dim=2, pixel_dim=3, pe_enable=True) -> None:
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
        self.to(device)
        print(self.base)
        
    def reset_weights(self):
        self.base[-1].weight.data = torch.randn_like(self.base[-1].weight.data)
        self.base[-1].bias.data = torch.randn_like(self.base[-1].bias.data)

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = x

        colors = torch.clamp(colors, 0, 1)
        return colors
    
    def render_img(self, width=512, height=512, disturb=False):
        #only support one RGB image rendering now 
        coo_tensor = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
        j = torch.arange(start=0, end=height, device=device).unsqueeze(0).transpose(0, 1).repeat(1, width)
        i = torch.arange(start=0, end=height, device=device).unsqueeze(0).repeat(height, 1)
        if disturb:
            x = ((j + torch.rand_like(j, dtype=torch.float32) - 0.5) * 2 + 1.0) / height - 1.0
            y = ((i + torch.rand_like(i, dtype=torch.float32) - 0.5) * 2 + 1.0) / width - 1.0
        else:
            x = (j * 2 + 1.0) / height - 1.0
            y = (i * 2 + 1.0) / width - 1.0
        coo_tensor[:, :, 0] = x
        coo_tensor[:, :, 1] = y
        coo_tensor = coo_tensor.reshape(-1, 2) # [H*W, 2]
        img_tensor = (self(coo_tensor) + 1) / 2 # color rgb [0, 1]
        #img_tensor: [1, 3, H, W]
        img_tensor = img_tensor.reshape(1, height, width, 3).permute(0, 3, 1, 2).contiguous()
        return img_tensor
    
    def img_show(self, width=512, height=512):
        img_tensor = self.render_img(width, height)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        plt.imshow(img_array)
        plt.show()
    
    def img_save(self, save_path, width=512, height=512):
        img_tensor = self.render_img(width, height)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imsave(save_path, img_array)

    def net_save(self, save_path):
        #format of saved object is .pth
        torch.save(self, save_path)

# test main function: train a mlp with entire image and render it, and save the mlp
def main():
    from torchvision import transforms
    from PIL import Image
    import argparse
    from Neural_Texture_Field import NeuralTextureField

    parse = argparse.ArgumentParser()
    parse.add_argument('--lr', type=float, help='learning rate')
    parse.add_argument('--ep', type=int, help='epochs')
    parse.add_argument( '--ns', action='store_true', help='do not save the network and images')
    arg = parse.parse_args()

    width = 512
    height = 512
    img_path = "./Assets/Images/car_texture.png"
    save_path = "./Assets/Image_MLP/nascar/"
    img_train = Image.open(img_path).resize((width, height))
    img_train.save(save_path + "train.png")
    img_train = transforms.ToTensor()(img_train).to(device).unsqueeze(0)[:,0:3,:,:] # [1, 3, H, W]

    test_mlp = NeuralTextureField(width=512, depth=3, pe_enable=True)
    learning_rate = 0.00001
    if arg.lr:
        learning_rate = arg.lr
    epochs = 10000
    if arg.ep:
        epochs = arg.ep
    optimizer = torch.optim.Adam(test_mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    total_loss = 0
    
    for epoch in range(epochs): 
        optimizer.zero_grad()     
        img_tensor = test_mlp.render_img(width, height, disturb=True)
        loss = criterion(img_tensor, img_train)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss}")
            total_loss = 0
        if (epoch+1) % 2000 == 0:
            test_mlp.img_save(width=width, height=height, save_path=save_path+f"ep{epoch+1}.png")
    
    if not arg.ns:
        test_mlp.net_save(save_path=save_path+"nth.pth")
        test_mlp.img_show(width, height)
        test_mlp.img_save(width=width, height=height, save_path=(save_path + f"{width}_{height}.png"))

# test main function: use pretrained mlp and directly render the image of mlp
def main_2():
    import torch
    from Neural_Texture_Field import NeuralTextureField

    mlp = NeuralTextureField(width=512, depth=3, pe_enable=True)
    load_path = "./Experiments/mlp_represented_image_training _entire_image/test5_validate/nth.pth"
    mlp.load_state_dict(torch.load(load_path))
    #validation rendering
    mlp.img_show(512, 512)

if __name__ == "__main__":
    main()