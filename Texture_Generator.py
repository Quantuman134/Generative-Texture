from utils import device, seed_everything
from pytorch3d import io
import torch
from Neural_Texture_Field import NeuralTextureField
from Differentiable_Texture import DiffTexture
from Stable_Diffusion import StableDiffusion
from Neural_Texture_Renderer import NeuralTextureRenderer
import time
import numpy as np
import matplotlib.pyplot as plt
import utils

class TextureGenerator:
    def __init__(self, mesh_path, diff_tex=None, is_latent=False) -> None:
        self.device = device
        mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
        _, faces, aux = io.load_obj(mesh_path, device=device)
        self.mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}
        self.diff_tex = diff_tex
        if diff_tex is None:
            self.diff_tex = DiffTexture()
        self.renderer = NeuralTextureRenderer()
        self.is_latent=is_latent
    
    def texture_train(self, text_prompt, lr, epochs, save_path=None):
        optimizer = torch.optim.Adam(self.diff_tex.parameters(), lr=lr)
        guidance = StableDiffusion(device=self.device)
        text_embeddings = guidance.get_text_embeds(text_prompt, '')

        if self.is_latent:
            self.renderer.rasterization_setting(image_size=64)
        
        #save initial data
        if save_path is not None:
            self.diff_tex.img_save(save_path=save_path + f"/tex_initial.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, light_enable=False, dist=2.0)
            i = 0
            for img_tensor in img_tensor_list:
                i += 1
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/initial_{i}.png", img_array)
            del img_tensor_list
                
        #range of camera position
        dist_range = [2.0, 2.5]
        elev_range = [0.0, 360.0]
        azim_range = [0.0, 360.0]

        info_update_period = 500

        print(f"[INFO] traning starts")
        start_t = time.time()
        total_loss = 0

        for epoch in range(epochs):

            optimizer.zero_grad()

            #rand = torch.rand(3)
            rand = torch.randint(0, 7, size=(2, ))
            dist = 2.0
            elev = 45 * rand[0]
            azim = 45 * rand[1]
            #dist = rand[0] * (dist_range[1] - dist_range[0]) + dist_range[0]
            #elev = rand[1] * (elev_range[1] - elev_range[0]) + elev_range[0]
            #azim = rand[2] * (azim_range[1] - azim_range[0]) + azim_range[0]
            self.renderer.camera_setting(dist=dist, elev=elev, azim=azim)
            pred_tensor = self.renderer.rendering(self.mesh_data, self.diff_tex, light_enable=False)[:, :, :, 0:-1]

            # save mediate results
            if (save_path is not None) and ((epoch+1) % info_update_period == 0):
                
                img_tensor = pred_tensor
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor.permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, :].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/ep_{epoch+1}.png", img_array)
                self.diff_tex.img_save(save_path=save_path + f"/tex_ep_{epoch+1}.png")

            pred_tensor = pred_tensor.permute(0, 3, 1, 2)
            tensor_for_backward, p_loss = guidance.train_step(pred_tensor=pred_tensor, text_embeddings=text_embeddings, latent_input=self.is_latent)
            tensor_for_backward.backward()
            optimizer.step()
            total_loss += p_loss
            
            if (epoch+1) % info_update_period == 0:
                end_t = time.time()
                print(f"[INFO] epoch {epoch+1} takes {(end_t - start_t):.4f} seconds. loss = {total_loss/info_update_period}")
                total_loss = 0
                start_t = end_t
            
        print(f"[INFO] traning ends")

        del guidance
        
        if save_path is not None:
            self.diff_tex.img_save(save_path=save_path + f"/tex_result.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, light_enable=False, dist=2.0)
            i = 0
            for img_tensor in img_tensor_list:
                i += 1
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/results_{i}.png", img_array)

    #temprarily disable
    def tex_net_save(self, save_path):
        self.tex_net.net_save(save_path)

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    text_prompt = "a black and white cow"
    save_path = "./Experiments/Generative_Texture_2/Diff_Texture_Around/fixed_multi_angles_128"
    #mlp_path = "./Assets/Image_MLP/gaussian_noise/nth.pt"
    #tex_net = torch.jit.load(mlp_path)
    diff_tex = DiffTexture(size=(128, 128), is_latent=True)
    texture_generator = TextureGenerator(mesh_path=mesh_path, diff_tex=diff_tex, is_latent=True)
    texture_generator.texture_train(text_prompt=text_prompt, lr=0.01, epochs=15000, save_path=save_path)


if __name__ == "__main__":
    main()
        





        