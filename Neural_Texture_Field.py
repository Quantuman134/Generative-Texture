import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device
import utils
import matplotlib.pyplot as plt
import numpy as np
import tinycudann as tcnn

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
# output: color, ideal range is [-1, 1], latent, [-1, 1] 
# input coordinate: the dimension of input coordinate is 2, and range of value is [-1, 1]. The positive direction
# of x and y are right and up respectively  
# sampling_disturb: is the uv coordinate disturbe subtly during forward texture sampling
# expected_tex_size: expected texture size represented by mlp, used in the forward disturb function
class NeuralTextureField(nn.Module):
    def __init__(self, width=512, depth=3, input_dim=2, expected_tex_size = 512, 
                 pe_enable=True, is_latent=False, sampling_disturb=False) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.pe_enable = pe_enable
        self.is_latent = is_latent
        self.sampling_disturb = sampling_disturb
        self.output_dim = 3
        if is_latent:
            self.output_dim = 4
        layers = []
        
        if pe_enable:
            #pe = PositionEncoding(input_dim=input_dim) #fraquency encoding
            desired_resolution = 4096
            num_levels = 16
            base_grid_resolution = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale                
            }
            pe = tcnn.Encoding(input_dim, encoding_config, dtype=torch.float32)
            layers.append(pe)
            #layers.append(nn.Linear(pe.mapping_size, width, bias=False))
            layers.append(nn.Linear(pe.n_output_dims, width, bias=False))
        else:
            layers.append(nn.Linear(input_dim, width, bias=False))  

        for i in range(depth - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(width, width, bias=False))

        layers.append(nn.ReLU())  
        layers.append(nn.Linear(width, self.output_dim, bias=False))
        self.base = nn.ModuleList(layers)

        self.reset_weights()
        self.to(device)
        print("[NEURAL TEXTURE INFO:]")
        print(self.base)
        
    def reset_weights(self):
        self.base[-1].weight.data = torch.ones_like(self.base[-1].weight.data) * 0.5
        #self.base[-1].bias.data = torch.randn_like(self.base[-1].bias.data)

    def forward(self, x):
        if self.sampling_disturb:
            x += (torch.rand_like(x) - 0.5) * (2/512) #the 512 of 2/512: a expected size of neural texture image.

        for layer in self.base:
            x = layer(x)
        colors = x

        colors = torch.clamp(colors, -1, 1)
        return colors
    

    def render_img(self, height=512, width=512):
        #only support one RGB image rendering now 
        #output: 'latent' or 'rgb', latent: expected output range [-1, 1], rgb: expected output range [0, 1]
        #output dim: tensor: [B, C, H, W]

        coo_tensor = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
        j = torch.arange(start=0, end=height, device=device).unsqueeze(0).transpose(0, 1).repeat(1, width)
        i = torch.arange(start=0, end=height, device=device).unsqueeze(0).repeat(height, 1)
        x = (j * 2 + 1.0) / height - 1.0
        y = (i * 2 + 1.0) / width - 1.0
        coo_tensor[:, :, 0] = x
        coo_tensor[:, :, 1] = y
        img_tensor = coo_tensor.reshape(-1, 2) # [H*W, 2] # color latent [-1, 1]
        if not self.is_latent:
            img_tensor = (self(img_tensor) + 1) / 2 # color rgb [0, 1]
        
        img_tensor = img_tensor.reshape(1, height, width, self.output_dim).permute(0, 3, 1, 2).contiguous()
        return img_tensor
    
    def img_show(self, height=512, width=512):
        img_tensor = self.render_img(height, width)[:, 0:3, :, :]
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imshow(img_array)
        plt.show()
    
    def img_save(self, save_path, height=512, width=512, rgb=True):
        img_tensor = self.render_img(height, width)
        if self.is_latent and rgb:
            img_tensor = utils.decode_latents(img_tensor)
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imsave(save_path, img_array)

    def tex_save(self, save_path):
        #format of saved object is .pth
        torch.save(self.state_dict(), save_path)

    def tex_load(self, tex_path):
        self.load_state_dict(torch.load(tex_path))

# test main function: train a mlp with entire image and render it, and save the mlp
def main():
    from torchvision import transforms
    from PIL import Image
    import argparse
    from Neural_Texture_Field import NeuralTextureField
    import time

    parse = argparse.ArgumentParser()
    parse.add_argument('--lr', type=float, help='learning rate')
    parse.add_argument('--ep', type=int, help='epochs')
    parse.add_argument( '--ns', action='store_true', help='do not save the network and images')
    arg = parse.parse_args()

    height = 64
    width = 64
    
    img_path = "./Assets/Images/Gaussian_Noise_Latent.png"
    save_path = "./Experiments/MLP_Image_Tracking/Gaussian_noise_64"
    #img_path = "./Assets/Images/cat.jpg"
    #save_path = "./Experiments/MLP_Image_Tracking/cat_64_64"
    img_train = Image.open(img_path).resize((width, height))
    img_train.save(save_path + "/train.png")
    img_train = transforms.ToTensor()(img_train).to(device).unsqueeze(0)[:,0:3,:,:] # [1, 3, H, W]

    test_mlp = NeuralTextureField(width=32, depth=6, pe_enable=True)

    #load existing model
    #load_path = "./Experiments/MLP_Image_Tracking/cat_256_256/nth.pth"
    #test_mlp.tex_load(load_path)
    
    #recommanded lr: mlp 256x6 --- 0.0001, mlp 32x4 ---0.001
    learning_rate = 0.001
    if arg.lr:
        learning_rate = arg.lr
    epochs = 10000
    if arg.ep:
        epochs = arg.ep
    
    info_update_period = 5000
    optimizer = torch.optim.AdamW(test_mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    total_loss = 0
    
    start_t = time.time()

    test_mlp.sampling_disturb = True
    for epoch in range(epochs): 
        optimizer.zero_grad()     
        img_tensor = test_mlp.render_img(height, width)
        loss = criterion(img_tensor, img_train)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if (epoch+1) % info_update_period == 0:
            end_t = time.time()
            print(f"Epoch {epoch+1}, takes {(end_t - start_t):.4f} s. Loss: {total_loss/info_update_period}")
            start_t = end_t
            total_loss = 0
            test_mlp.img_save(width=width, height=height, save_path=save_path+f"/ep_{epoch+1}.png")
    
    if not arg.ns:
        test_mlp.tex_save(save_path=save_path+"/nth.pth")
        test_mlp.img_show(height, width)
        test_mlp.img_save(width=width, height=height, save_path=(save_path + f"/{width}_{height}.png"))

# test main function: use pretrained mlp and directly render the image of mlp
def main_2():
    import torch
    from Neural_Texture_Field import NeuralTextureField

    mlp = NeuralTextureField(width=32, depth=6, pe_enable=True)
    load_path = "./Assets/Image_MLP/Gaussian_noise_latent_64/nth.pth"
    mlp.tex_load(load_path)
    mlp.sampling_disturb=True
    img_tensor = mlp.render_img(64, 64)
    #validation rendering
    img_array = img_tensor[0, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
    img_array = np.clip(img_array, 0, 1)
    #mlp.img_show(512, 512)
    plt.imshow(img_array)
    plt.show()

# According to the principle of NeRF, we rendered the neural texture field as an image, and optimize the image 
# using several one rendered images in fixed view and a exsiting square mesh 
def main_3():
    from Neural_Texture_Renderer import NeuralTextureRenderer
    from Neural_Texture_Field import NeuralTextureField
    from PIL import Image
    from pytorch3d import io
    from torchvision import transforms
    import time

    renderer = NeuralTextureRenderer()
    diff_tex = NeuralTextureField(width=32, depth=6, pe_enable=True, sampling_disturb=True)

    # asset loading
    mesh_path = "./Assets/3D_Model/Square/square.obj"
    image_path = "./Assets/Images/cat_512_512.png"
    save_path = "./Experiments/MLP_3D_Appearance_Tracking/Square/cat_32"

    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    _, faces, aux = io.load_obj(mesh_path, device=device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    img = Image.open(image_path).resize((64, 64))
    img.save(save_path + "/train.png")
    img_trained = transforms.ToTensor()(img).unsqueeze(0).permute(0, 2, 3, 1).to(device)
    
    # optimization in the fixed view
    # renderer setting
    offset = torch.tensor([[0, 0, 0]])
    renderer.camera_setting(dist=1.6, elev=0, azim=0, offset=offset)
    renderer.rasterization_setting(image_size=64)

    # optimization parameters
    epochs = 10000
    lr = 0.001
    info_update_period = 1000
    optimizer = torch.optim.Adam(diff_tex.parameters(), lr=lr)
    criterion = nn.MSELoss()

    #training starts
    print("[INFO] training starts...")
    total_loss = 0
    start_t = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        img_rendered = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=False)[:, :, :, 0:3]

        loss = criterion(img_rendered, img_trained)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if (epoch+1) % info_update_period == 0:
            end_t = time.time()
            print(f"[INFO] Epoch {epoch+1}, takes {(end_t - start_t):.4f} s. Loss: {total_loss/info_update_period}")
            total_loss = 0
            diff_tex.img_save(save_path=save_path+f"/tex_ep_{epoch+1}.png")
            img_rendered = img_rendered[0, :, :, 0:3].detach().cpu().numpy()
            img_rendered = np.clip(img_rendered, 0, 1)
            plt.imsave(save_path+f"/rendered_ep_{epoch+1}.png", img_rendered)

            start_t = end_t

def main_4():
    from Neural_Texture_Renderer import NeuralTextureRenderer
    from Neural_Texture_Field import NeuralTextureField
    from Differentiable_Texture import DiffTexture
    from PIL import Image
    from pytorch3d import io
    from torchvision import transforms
    import time

    # asset loading
    mesh_path = "./Assets/3D_Model/Orange_Car/source/Orange_Car.obj"
    texture_path = "./Assets/3D_Model/Orange_Car/textures/Body_dDo_d_orange.jpeg"
    save_path = "./Experiments/MLP_3D_Appearance_Tracking/Model_Fixed_View"

    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)

    # normalize model, fit it into a bounding box with [-0.5, 0.5] range
    verts_packed = mesh_obj.verts_packed()
    verts_max = verts_packed.max(dim=0).values
    verts_min = verts_packed.min(dim=0).values
    max_length = (verts_max - verts_min).max().item()
    center = (verts_max + verts_min)/2

    verts_list = mesh_obj.verts_list()
    verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list]

    _, faces, aux = io.load_obj(mesh_path, device=device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    renderer = NeuralTextureRenderer()
    diff_tex = NeuralTextureField(width=256, depth=6, pe_enable=True, sampling_disturb=True)
    gt_tex = DiffTexture(size=(1024, 1024), is_latent=False)

    image = Image.open(texture_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    gt_tex.set_image(image_tensor)

    # optimization in the fixed view
    # renderer setting
    offset = torch.tensor([[0, 0, 0]])
    renderer.camera_setting(dist=1.0, elev=0, azim=45, offset=offset)
    renderer.rasterization_setting(image_size=512)
    renderer.light_setting([[-1, 1, -1]])

    with torch.no_grad():
        img_trained = renderer.rendering(mesh_data=mesh_data, diff_tex=gt_tex, light_enable=False)[:, :, :, 0:3]
        img_array = img_trained[0, :, :, 0:3].detach().cpu().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imsave(save_path+f"/gt.png", img_array)
        gt_tex.img_save(save_path=save_path+"/gt_tex.png", height=1024, width=1024)


    # optimization parameters
    epochs = 10000
    lr = 0.00001
    info_update_period = 100
    optimizer = torch.optim.Adam(diff_tex.parameters(), lr=lr)
    criterion = nn.MSELoss()

    #training starts
    print("[INFO] training starts...")
    total_loss = 0
    start_t = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        img_rendered = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=False)[:, :, :, 0:3]

        loss = criterion(img_rendered, img_trained)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if (epoch+1) % info_update_period == 0:
            end_t = time.time()
            print(f"[INFO] Epoch {epoch+1}, takes {(end_t - start_t):.4f} s. Loss: {total_loss/info_update_period}")
            total_loss = 0
            diff_tex.img_save(save_path=save_path+f"/tex_ep_{epoch+1}.png")
            img_rendered = img_rendered[0, :, :, 0:3].detach().cpu().numpy()
            img_rendered = np.clip(img_rendered, 0, 1)
            plt.imsave(save_path+f"/rendered_ep_{epoch+1}.png", img_rendered)

            start_t = end_t

def main_5():
    from Neural_Texture_Renderer import NeuralTextureRenderer
    from Neural_Texture_Field import NeuralTextureField
    from Differentiable_Texture import DiffTexture
    from PIL import Image
    from pytorch3d import io
    from torchvision import transforms
    import time

    # asset loading
    mesh_path = "./Assets/3D_Model/Orange_Car/source/Orange_Car.obj"
    texture_path = "./Assets/3D_Model/Orange_Car/textures/Body_dDo_d_orange.jpeg"
    save_path = "./Experiments/MLP_3D_Appearance_Tracking/Model"

    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)

    # normalize model, fit it into a bounding box with [-0.5, 0.5] range
    verts_packed = mesh_obj.verts_packed()
    verts_max = verts_packed.max(dim=0).values
    verts_min = verts_packed.min(dim=0).values
    max_length = (verts_max - verts_min).max().item()
    center = (verts_max + verts_min)/2

    verts_list = mesh_obj.verts_list()
    verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list]

    _, faces, aux = io.load_obj(mesh_path, device=device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    renderer = NeuralTextureRenderer()
    diff_tex = NeuralTextureField(width=256, depth=6, pe_enable=True, sampling_disturb=True)
    gt_tex = DiffTexture(size=(1024, 1024), is_latent=False)

    image = Image.open(texture_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    gt_tex.set_image(image_tensor)

    # optimization in the fixed view
    # renderer setting
    offset = torch.tensor([[0, 0, 0]])
    renderer.camera_setting(dist=1.0, elev=0, azim=45, offset=offset)
    renderer.rasterization_setting(image_size=512)
    renderer.light_setting([[-1, 1, -1]])

    with torch.no_grad():
        img_list_trained = renderer.render_around(mesh_data=mesh_data, diff_tex=gt_tex, dist=1.0, elev=0, light_enable=False)
        for count, img_tensor in enumerate(img_list_trained):
            img_array = img_tensor[0, :, :, 0:3].detach().cpu().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path+f"/gt_{count}.png", img_array)
        gt_tex.img_save(save_path=save_path+f"/gt_tex.png", height=1024, width=1024)


    # optimization parameters
    epochs = 10000
    lr = 0.00001
    info_update_period = 500
    optimizer = torch.optim.Adam(diff_tex.parameters(), lr=lr)
    criterion = nn.MSELoss()

    #training starts
    print("[INFO] training starts...")
    total_loss = 0
    start_t = time.time()
    for epoch in range(epochs):
        for count, img_trained in enumerate(img_list_trained):
            optimizer.zero_grad()

            renderer.camera_setting(dist=1.0, elev=0, azim=count*45, offset=offset)
            img_rendered = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=False)[:, :, :, 0:3]
            loss = criterion(img_rendered, img_trained[:, :, :, 0:3])
            total_loss += loss
            loss.backward()
            optimizer.step()

        if (epoch+1) % info_update_period == 0:
            end_t = time.time()
            print(f"[INFO] Epoch {epoch+1}, takes {(end_t - start_t):.4f} s. Loss: {total_loss/info_update_period}")
            total_loss = 0
            diff_tex.img_save(save_path=save_path+f"/tex_ep_{epoch+1}.png")
            image_tensors = renderer.render_around(mesh_data=mesh_data, diff_tex=diff_tex, dist=1.0, offset=offset, elev=0)
            for count, image_tensor in enumerate(image_tensors):
                image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
                image_array = np.clip(image_array, 0, 1)
                plt.imsave(save_path+f"/rendered_ep_{epoch+1}_{count}.png", image_array)
            
            del image_tensors

            start_t = end_t

if __name__ == "__main__":
    main_3()