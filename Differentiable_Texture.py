from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import device
import utils
import matplotlib.pyplot as plt
import numpy as np

#size: width x height
# output: color, ideal range is [-1, 1] 
# input coordinate: the dimension of input coordinate is 2, and range of value is [-1, 1]. The positive direction
# of x and y are right and up respectively
# current texture sample is bilinear 
# is_latent: flag of latent format. (latent size are as [1, 4, width, height])

class DiffTexture(nn.Module):
    def __init__(self, size=(512, 512), is_latent=False) -> None:
        super().__init__()
        self.width = size[0]
        self.height = size[1]
        self.is_latent = is_latent
        self.set_gaussian()

    def set_gaussian(self, mean=0, sig=1):
        if self.is_latent:
            self.texture = torch.nn.Parameter(torch.randn((self.height, self.width, 4), dtype=torch.float32, device=device, requires_grad=True) * sig + mean)
        else:
            self.texture = torch.nn.Parameter(torch.randn((self.height, self.width, 3), dtype=torch.float32, device=device, requires_grad=True) * sig + mean)

    # temporarily disable
    def set_image(self, img_tensor):
        #img_tensor size: [B, C, H, W], color value [0, 1]
        B, C, H, W = img_tensor.size()
        img_tensor = img_tensor * 2 - 1.0
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(self.height, self.width))
        img_tensor = img_tensor.to(device)
        self.texture = torch.nn.Parameter(img_tensor.squeeze().permute(1, 2, 0))
        

    def forward(self, uvs):
        if uvs.dim() == 1:
            color = self.texture_sample(uvs)
            return color
        else:
            colors = self.texture_batch_sample(uvs)
            return colors
    
    def texture_sample(self, uv):
        uv = (uv + 1)/2
        uv[0] *= (self.width - 1)
        uv[1] *= (self.height - 1)
        u_0 = uv[0].floor().type(torch.int32)
        u_1 = uv[0].ceil().type(torch.int32)
        v_0 = uv[1].floor().type(torch.int32)
        v_1 = uv[1].ceil().type(torch.int32)
        a = uv[0] - u_0
        b = uv[1] - v_0
        color = (self.texture[u_0, v_0] * a + self.texture[u_1, v_0] * (1 - a)) * b \
        + (self.texture[u_0, v_1] * a + self.texture[u_1, v_1] * (1 - a)) * (1 - b)

        color = F.tanh(color)
        return color
    
    def texture_batch_sample(self, uvs):
        uvs = (uvs + 1)/2
        uvs[:, 0] *= (self.width - 1)
        uvs[:, 1] *= (self.height - 1)
        us_0 = uvs[:, 0].floor().type(torch.int32)
        us_1 = uvs[:, 0].ceil().type(torch.int32)
        vs_0 = uvs[:, 1].floor().type(torch.int32)
        vs_1 = uvs[:, 1].ceil().type(torch.int32)
        a = (uvs[:, 0] - us_0).reshape(-1, 1)
        b = (uvs[:, 1] - vs_0).reshape(-1, 1)
        colors = (self.texture[us_0, vs_0] * a + self.texture[us_1, vs_0] * (1 - a)) * b \
        + (self.texture[us_0, vs_1] * a + self.texture[us_1, vs_1] * (1 - a)) * (1 - b)

        colors = F.tanh(colors)
        return colors
    
    def render_img(self, width=512, height=512):
        #output: 'torch' or 'rgb', torch: expected output range [-1, 1], rgb: expected output range [0, 1]
        coo_tensor = torch.zeros((height, width, 2), dtype=torch.float32, device=device)
        j = torch.arange(start=0, end=height, device=device).unsqueeze(0).transpose(0, 1).repeat(1, width)
        i = torch.arange(start=0, end=height, device=device).unsqueeze(0).repeat(height, 1)
        x = (j * 2 + 1.0) / height - 1.0
        y = (i * 2 + 1.0) / width - 1.0
        coo_tensor[:, :, 0] = x
        coo_tensor[:, :, 1] = y
        coo_tensor = coo_tensor.reshape(-1, 2) # [H*W, 2]
        img_tensor = (self(coo_tensor) + 1) / 2 # color rgb [0, 1]
        img_tensor = img_tensor.reshape(width, height, img_tensor.size()[1]) # depth 3 or 4
        #img_tensor: [1, 3 or 4, H, W]
        img_tensor = img_tensor.reshape(1, height, width, img_tensor.size()[2]).permute(0, 3, 1, 2).contiguous()
        return img_tensor
    
    def img_show(self, width=512, height=512):
        img_tensor = self.render_img(width, height)[:, 0:3, :, :] #if latents, maintain the first 3 components
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imshow(img_array)
        plt.show()
    
    def img_save(self, save_path, width=512, height=512):
        img_tensor = self.render_img(width, height)[:, 0:3, :, :] #if latents, maintain the first 3 components
        img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img_array = np.clip(img_array, 0, 1)
        plt.imsave(save_path, img_array)
    
    def latent2rgb(self):
        self.is_latent = False
        img_tensor = utils.decode_latents(self.texture.reshape(1, self.height, self.width, 4).permute(0, 3, 1, 2))
        self.texture = img_tensor.unsqueeze(0).permute(1, 2, 0)
    


def main():
    from torch.utils.tensorboard import SummaryWriter
    import utils
    from Stable_Diffusion import StableDiffusion
    import time
    from PIL import Image
    from torchvision import transforms
    texture = DiffTexture()
    import_img_path = "./test_1.png"
    img = Image.open(import_img_path)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)[:, 0:3, :, :]

    texture.set_image(img_tensor=img_tensor)
    texture.img_show(512, 512)

# training a differentiable texture with stable-diffusion guidance in rgb space
def main_2():
    from torch.utils.tensorboard import SummaryWriter
    import utils
    from Stable_Diffusion import StableDiffusion
    import time
    from PIL import Image
    from torchvision import transforms

    seed = 0
    utils.seed_everything(seed)
    diff_tex = DiffTexture(size=(512, 512))
    guidance = StableDiffusion(device=device)
    #scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    #configuring
    text_prompt = "an orange cat head"
    text_embeddings = guidance.get_text_embeds(text_prompt, '')
    min_t = 0.02
    max_t = 0.98
    epochs = 1000
    lr = 0.1

    #import_img_path = "./Assets/Images/Gaussian_Noise_Latent.png"
    #import_img_path = "./test.png"
    #img = Image.open(import_img_path)
    #img_tensor = transforms.ToTensor()(img).unsqueeze(0)[:, 0:3, :, :]
    #diff_tex.set_image(img_tensor=img_tensor)

    optimizer = torch.optim.Adam(diff_tex.parameters(), lr=lr)
    info_period: int = 50
    image_save = True
    save_path = "./Experiments/Differentiable_Image_Generation/structure_noise_comparison/latent_image/"
    save_period: int = 50

    #tensorboard
    writer = SummaryWriter()

    #training
    start_t = time.time()
    total_loss = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        img_pred = diff_tex.render_img()
        img_pred.retain_grad()
        if image_save and (epoch+1)%save_period == 0:
            tensor_for_backward, p_loss, latents, noise_rgb, img_noisy_rgb, noise_pred_rgb, img_denoised_rgb, pred_diff_rgb, t\
              = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings, min_t=min_t, max_t=max_t, detailed=True)
        else:
            tensor_for_backward, p_loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings, min_t=min_t, max_t=max_t)

        total_loss += p_loss
        tensor_for_backward.backward()
        optimizer.step()
        #scaler.scale(tensor_for_backward).backward()
        #scaler.step(optimizer)
        #custom_lr_adjust(optimizer, epoch, lr)
        #scaler.update()

        if (epoch+1)%info_period == 0:
            end_t = time.time()
            print(f"[INFO] epoch {epoch+1} takes {(end_t - start_t):.4f} seconds. loss = {total_loss / info_period}")
            writer.add_scalar("Loss/train", total_loss / info_period, epoch)
            total_loss = 0
            start_t = end_t
        if image_save and (epoch+1)%save_period == 0:
            diff_tex.img_save(save_path=save_path + f"ep_{epoch+1}.png")
            
            #latents: latents is a tensor with size of [1, 4, 64, 64], save it as image of [64, 64, 3], which only reserve first 3 dimension of depth.
            latents = latents[:, 0:3, :, :]
            img_array = latents.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_latent.png", img_array)

            #image_grad
            grad_tensor = (diff_tex.texture.grad * 200 + 1)/2 
            img_array = grad_tensor.cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_image_grad.png", img_array)

            #added_noise
            img_array = noise_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_added_noise.png", img_array)

            #noisy img
            img_array = img_noisy_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_noisy_img.png", img_array)

            #pred_noise
            img_array = noise_pred_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_pred_noise.png", img_array)

            #denoised img
            img_array = img_denoised_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_denoised_img.png", img_array)

            #noise difference
            img_array = pred_diff_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_noise_diff.png", img_array)

        torch.cuda.empty_cache()
            
    
    writer.flush()
    writer.close()

    #rendering
    img_tensor = diff_tex.render_img()
    img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)
    img_array = img_tensor.cpu().detach().numpy()
    img_array = np.clip(img_array, 0, 1)
    plt.imshow(img_array)
    plt.show()

# training a differentiable texture with stable-diffusion guidance in latent space
def main_3():
    from torch.utils.tensorboard import SummaryWriter
    import utils
    from Stable_Diffusion import StableDiffusion
    import time
    from PIL import Image
    from torchvision import transforms

    seed = 0
    utils.seed_everything(seed)
    diff_tex = DiffTexture(size=(512, 512), is_latent=True)
    guidance = StableDiffusion(device=device)
    #scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    #configuring
    text_prompt = "an orange cat head"
    text_embeddings = guidance.get_text_embeds(text_prompt, '')
    min_t = 0.02
    max_t = 0.98
    epochs = 1000
    lr = 0.1

    #import_img_path = "./Assets/Images/Gaussian_Noise_Latent.png"
    #import_img_path = "./test.png"
    #img = Image.open(import_img_path)
    #img_tensor = transforms.ToTensor()(img).unsqueeze(0)[:, 0:3, :, :]
    #diff_tex.set_image(img_tensor=img_tensor)

    optimizer = torch.optim.Adam(diff_tex.parameters(), lr=lr)
    info_period: int = 50
    image_save = True
    save_path = "./Experiments/Differentiable_Image_Generation/structure_noise_comparison/latent_image/"
    save_period: int = 50

    #tensorboard
    writer = SummaryWriter()

    #training
    start_t = time.time()
    total_loss = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        img_pred = diff_tex.render_img()
        img_pred.retain_grad()
        if image_save and (epoch+1)%save_period == 0:
            tensor_for_backward, p_loss, latents, noise_rgb, img_noisy_rgb, noise_pred_rgb, img_denoised_rgb, pred_diff_rgb, t\
              = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings, min_t=min_t, max_t=max_t, detailed=True)
        else:
            tensor_for_backward, p_loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings, min_t=min_t, max_t=max_t)

        total_loss += p_loss
        tensor_for_backward.backward()
        optimizer.step()
        #scaler.scale(tensor_for_backward).backward()
        #scaler.step(optimizer)
        #custom_lr_adjust(optimizer, epoch, lr)
        #scaler.update()

        if (epoch+1)%info_period == 0:
            end_t = time.time()
            print(f"[INFO] epoch {epoch+1} takes {(end_t - start_t):.4f} seconds. loss = {total_loss / info_period}")
            print(f"learning rate = {optimizer.param_groups[0]['lr']}")
            writer.add_scalar("Loss/train", total_loss / info_period, epoch)
            total_loss = 0
            start_t = end_t
        if image_save and (epoch+1)%save_period == 0:
            diff_tex.img_save(save_path=save_path + f"ep_{epoch+1}.png")
            
            #latents: latents is a tensor with size of [1, 4, 64, 64], save it as image of [64, 64, 3], which only reserve first 3 dimension of depth.
            latents = latents[:, 0:3, :, :]
            img_array = latents.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_latent.png", img_array)

            #image_grad
            grad_tensor = (diff_tex.texture.grad * 200 + 1)/2 
            img_array = grad_tensor.cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_image_grad.png", img_array)

            #added_noise
            img_array = noise_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_added_noise.png", img_array)

            #noisy img
            img_array = img_noisy_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_noisy_img.png", img_array)

            #pred_noise
            img_array = noise_pred_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_pred_noise.png", img_array)

            #denoised img
            img_array = img_denoised_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_denoised_img.png", img_array)

            #noise difference
            img_array = pred_diff_rgb.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"ep_{epoch+1}_noise_diff.png", img_array)

        torch.cuda.empty_cache()
            
    
    writer.flush()
    writer.close()

    #rendering
    img_tensor = diff_tex.render_img()
    img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)
    img_array = img_tensor.cpu().detach().numpy()
    img_array = np.clip(img_array, 0, 1)
    plt.imshow(img_array)
    plt.show()

if __name__ == '__main__':
    main_2()
