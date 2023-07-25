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

        # put mesh in center and [-0.5, 0.5] bounding box
        verts_packed = mesh_obj.verts_packed()
        verts_max = verts_packed.max(dim=0).values
        verts_min = verts_packed.min(dim=0).values
        max_length = (verts_max - verts_min).max().item()
        center = (verts_max + verts_min)/2

        verts_list = mesh_obj.verts_list()
        verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list]

        self.mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}
        self.diff_tex = diff_tex
        if diff_tex is None:
            self.diff_tex = DiffTexture(is_latent=is_latent)
        self.renderer = NeuralTextureRenderer()
        self.is_latent=is_latent
    
    # offset: the camera transition offset that point to center of object
    # dist,elev,azim range: the range of camera configuration, the maximum dist will be implemented
    # in the distance of render_around()
    # info_update_period: the period of saving and printing information of training, whose unit is epoch
     
    def texture_train(self, text_prompt, lr, epochs, save_path=None, 
                      offset=[0.0, 0.0, 0.0], dist_range=[1.0, 2.0], 
                      elev_range=[0.0, 360.0], azim_range=[0.0, 360.0],
                      info_update_period=500):
        if self.is_latent:
            self.renderer.rasterization_setting(image_size=64)
        
        offset = torch.tensor([offset])

        #save initial data
        if save_path is not None:
            self.diff_tex.img_save(save_path=save_path + f"/tex_initial.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, offset=offset, 
                                                          light_enable=False, dist=dist_range[1])
            
            for count, img_tensor in enumerate(img_tensor_list):
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/initial_{count}.png", img_array)
            del img_tensor_list
        
        optimizer = torch.optim.Adam(self.diff_tex.parameters(), lr=lr)
        guidance = StableDiffusion(device=self.device)
        text_embeddings = guidance.get_text_embeds(text_prompt, '')

        print(f"[INFO] traning starts")
        start_t = time.time()
        total_loss = 0

        for epoch in range(epochs):

            optimizer.zero_grad()

            rand = torch.rand(3)
            randint = torch.randint(0, 25, size=(1, ))

            dist = rand[0] * (dist_range[1] - dist_range[0]) + dist_range[0]
            #elev = rand[1] * (elev_range[1] - elev_range[0]) + elev_range[0]
            #azim = rand[2] * (azim_range[1] - azim_range[0]) + azim_range[0]
            elev = 0
            azim = 15 * randint[0]
            #azim = 90.0
            self.renderer.camera_setting(dist=dist, elev=elev, azim=azim, offset=offset)
            pred_tensor = self.renderer.rendering(self.mesh_data, self.diff_tex, light_enable=False, rand_back=True)[:, :, :, 0:-1]

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
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, offset=offset, light_enable=False, dist=1.3)

            for count, img_tensor in enumerate(img_tensor_list):
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/results_{count}.png", img_array)

    #temprarily disable
    def tex_net_save(self, save_path):
        self.diff_tex.net_save(save_path)

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    mesh_path = "./Assets/3D_Model/Orange_Car/source/Orange_Car.obj"
    text_prompt = "a Chevrolet pony car"
    save_path = "./Experiments/Generative_Texture_MLP/Orange_Car"
    mlp_path = "./Assets/Image_MLP/Gaussian_noise_latent/latent_noise.pth"

    #diff_tex = DiffTexture(size=(256, 256), is_latent=True)
    diff_tex = NeuralTextureField(width=256, depth=6)
    diff_tex.tex_load(tex_path=mlp_path)

    texture_generator = TextureGenerator(mesh_path=mesh_path, diff_tex=diff_tex, is_latent=False)
    texture_generator.texture_train(text_prompt=text_prompt, lr=0.00001, epochs=100000, save_path=save_path, 
                                    dist_range=[1.0, 1.0], elev_range=[0.0, 360.0], azim_range=[0.0, 360.0],
                                    info_update_period=100)


if __name__ == "__main__":
    main()
        





        