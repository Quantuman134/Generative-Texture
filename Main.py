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
from One_Layer_Image import OneLayerImage
import numpy as np
from Neural_Texture_Shader import NeuralTextureShader
from pytorch3d import renderer
from pytorch3d import io
from Texture_Generator import TextureGenerator

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

def train_mlp_sd(mlp, epochs, lr, text_prompt, save_path):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    guidance = StableDiffusion(device=device)
    text_embeddings = guidance.get_text_embeds(text_prompt, '')
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    print(f"[INFO] traning starts")
    total_loss = 0
    img_array = mlp.render_img(512, 512).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img_array = np.clip(img_array, 0, 1)
    plt.imsave(save_path + "/initial.png", img_array)
    for epoch in range(epochs):
        start_t = time.time()

        #render current image (test resolution: 512x512)
        img_pred = mlp.render_img(512, 512)
        
        if (epoch+1)%1000 == 0:
            img_array = mlp.render_img(512, 512).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"/ep{epoch+1}.png", img_array)

        optimizer.zero_grad()
        loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings)
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end_t = time.time()
        if True:
            print(f"[INFO] epoch {epoch} takes {(end_t - start_t):.4f} seconds.")

def train_oli_sd(mlp, epochs, lr, text_prompt):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    guidance = StableDiffusion(device=device)
    text_embeddings = guidance.get_text_embeds(text_prompt, '')
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    print(f"[INFO] traning starts")
    total_loss = 0
    for epoch in range(epochs):
        start_t = time.time()

        img_pred = mlp(torch.tensor([1], dtype=torch.float32, device=device))
        optimizer.zero_grad()
        loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end_t = time.time()
        if True:
            print(f"[INFO] epoch {epoch} takes {(end_t - start_t):.4f} seconds.")

def train_tex_mlp_sd(mlp, mesh_obj, faces, aux, epochs, lr, text_prompt, save_path=None):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    guidance = StableDiffusion(device=device)
    text_embeddings = guidance.get_text_embeds(text_prompt, '')
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    #differentiable rendering
    R, T = renderer.look_at_view_transform(2.3, 0, 135)
    camera = renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    #light
    light = renderer.PointLights(device=device, location=[[0.0, 10.0, 10.0]])

    #renderer
    raster_setting = renderer.RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)

    mesh_renderer = renderer.MeshRenderer(
        rasterizer=renderer.MeshRasterizer(
            cameras=camera,
            raster_settings=raster_setting
        ),
        shader = NeuralTextureShader(
            tex_mlp=mlp,
            device=device,
            cameras=camera,
            light_enable=True,
            lights=light,
            faces=faces,
            aux=aux
        )
    )

    #save initial image
    img_array = mesh_renderer(mesh_obj)[0, :, :, 0:3].cpu().detach().numpy()
    img_array = np.clip(img_array, 0, 1)
    plt.imsave(save_path + "/initial.png", img_array)

    print(f"[INFO] traning starts")
    total_loss = 0
    for epoch in range(epochs):
        if epoch >= 1000:
            lr = 0.000001
            if epoch >=10000:
                lr = 0.0000001
        start_t = time.time()

        #rendering
        img_pred = mesh_renderer(mesh_obj)[:, :, :, 0:3]
        if (epoch+1) % 100 == 0:
            img_array = img_pred[0, :, :, 0:3].cpu().detach().numpy()
            img_array = np.clip(img_array, 0, 1)
            plt.imsave(save_path + f"/ep_{epoch+1}.png", img_array)

        img_pred = img_pred.permute(0, 3, 1, 2)
        optimizer.zero_grad()
        loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end_t = time.time()
        if True:
            print(f"[INFO] epoch {epoch} takes {(end_t - start_t):.4f} seconds.")



    return img_pred

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

# transfer a  One_Layer_Image with stable-diffusion guidance
def main_2_1():
    seed = 551447
    utils.seed_everything(seed)
    save_path = "./Experiments/One_Layer_Image_with_SD_guidance/apple/"
    img_path = "./Assets/Images/test_image_216_233.jpeg"
    img = Image.open(img_path)
    #mlp = OneLayerImage(img=img)
    mlp = OneLayerImage()

    #training
    epochs = 1000
    lr = 0.01
    #text_prompt = "a pixel style image of an orange cat head."
    text_prompt = "a red apple on desk, white background"
    train_oli_sd(mlp=mlp, epochs=epochs, lr=lr, text_prompt=text_prompt)

    img_array = mlp.render_img()
    img_array = np.clip(img_array, 0, 1)
    plt.imshow(img_array)
    plt.show()
    plt.imsave(save_path + f"ep_{epochs}_3.png", img_array)

# transfer a  MLP_Presented_Image with stable-diffusion guidance
def main_2_2():
    seed = 945
    utils.seed_everything(seed)
    save_path = "./Experiments/SDS_in_MLP_Represented_Image/pixel_cat2/test8_lr00001"
    mlp = NeuralTextureField(input_dim=2, width=512, depth=3, pe_enable=True)
    mlp.load_state_dict(torch.load("./Experiments/mlp_represented_image_training _entire_image/gaussian_noise/nth.pth"))

    #training
    epochs = 100000
    lr = 0.00001
    text_prompt = "a photo realistic image of an orange cat head, white background."
    #text_prompt = "a red apple on desk, white background"
    train_mlp_sd(mlp=mlp, epochs=epochs, lr=lr, text_prompt=text_prompt, save_path=save_path)

    img_tensor = mlp.render_img(512, 512)
    img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)
    img_array = img_tensor.cpu().detach().numpy()
    img_array = np.clip(img_array, 0, 1)
    plt.imshow(img_array)
    plt.show()
    #plt.imsave(save_path + f"ep_{epochs}_depth_1.png", img_array)


# train a Neural_Texture_Field under specific view rendering with stable-diffusion guidance
def main_3():
    seed = 84462
    utils.seed_everything(seed)
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    verts, faces, aux = io.load_obj(mesh_path, device=device)
    mlp_path = "./Experiments/mlp_represented_image_training _entire_image/gaussian_noise/nth.pth"
    tex_mlp = NeuralTextureField(width=512, depth=3, input_dim=2, pe_enable=True)
    tex_mlp.load_state_dict(torch.load(mlp_path))

    #training
    epochs = 5000
    lr = 0.00001
    text_prompt = "a photo realistic cow"
    save_path = "./Experiments/SDS_in_MLP_Represented_Texture_Specific_View/Cow_Experiment/cow6"
    #text_prompt = "a cat"
    img_pred = train_tex_mlp_sd(mlp=tex_mlp, mesh_obj=mesh_obj, faces=faces, aux=aux, epochs=epochs, lr=lr, text_prompt=text_prompt, save_path=save_path).permute(0, 2, 3, 1)

    img = img_pred[0, :, :, 0:3].cpu().detach().numpy()
    plt.imshow(img)
    plt.show()

    #plt.imsave(save_path + f"ep_{epochs}_3.png", img_array)    

# train a Neural_Texture_Field view under all around view of model with stable-diffusion guidance
def main_4():
    #configuration
    seed = 0
    utils.seed_everything(seed)
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mlp_path = "./Experiments/mlp_represented_image_training _entire_image/gaussian_noise/nth.pth"
    save_path = "./Experiments/Generative_Texture_1/test_experiment2"
    text_prompt = "a photo realistic cow"
    epochs = 5000
    lr = 0.000001
    tex_net = NeuralTextureField()
    tex_net.load_state_dict(torch.load(mlp_path))
    texture_generator = TextureGenerator(mesh_path=mesh_path, tex_net=tex_net)

    #train step
    texture_generator.texture_train(text_prompt=text_prompt, lr=lr, epochs=epochs, save_path=save_path)



if __name__ == "__main__":
    main_4()


