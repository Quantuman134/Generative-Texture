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
        verts, faces, aux = io.load_obj(mesh_path, device=device)

        # put mesh in center and [-0.5, 0.5] bounding box
        verts_packed = mesh_obj.verts_packed()
        verts_max = verts_packed.max(dim=0).values
        verts_min = verts_packed.min(dim=0).values
        max_length = (verts_max - verts_min).max().item()
        center = (verts_max + verts_min)/2

        verts_list = mesh_obj.verts_list()
        verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list]
        mesh_obj._verts_packed = (verts_packed - center)/max_length
    
        verts = (verts - center)/max_length

        self.mesh_data = {'mesh_obj': mesh_obj,'verts': verts, 'faces': faces, 'aux': aux}

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
                      info_update_period=500, render_light_enable=False,
                      tex_size=512, rendered_img_size=512, annealation=False,
                      field_sample=False):

        # highlight: a temp code line, normal is 512
        self.renderer.rasterization_setting(image_size=rendered_img_size)

        if self.is_latent:
            self.renderer.rasterization_setting(image_size=64)
        
        offset = torch.tensor([offset])

        # light setting
        self.renderer.light_setting(directions=[[1, 1, 1]])

        #save initial data
        if save_path is not None:
            if not field_sample:
                self.diff_tex.img_save(save_path=save_path + f"/tex_initial.png", width=tex_size, height=tex_size)
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, offset=offset, elev=25, 
                                                          light_enable=render_light_enable, dist=dist_range[1],
                                                          field_sample=field_sample)
            
            for count, img_tensor in enumerate(img_tensor_list):
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/initial_{count}.png", img_array)
            del img_tensor_list
        
        #optimizer = torch.optim.Adam(self.diff_tex.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.diff_tex.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-15)
        guidance = StableDiffusion(device=self.device)
        guidance.eval()
        text_embeddings = guidance.get_text_embeds(text_prompt, '')

        print(f"[INFO] traning starts")
        start_t = time.time()
        total_loss = 0

        # annealation
        min_t = 0.02
        max_t = 0.98
        min_t_ann = 0.02
        max_t_ann = 0.50
        ann_threshold = 0.3

        for epoch in range(epochs):

            optimizer.zero_grad()

            rand = torch.rand(3)
            randint = torch.randint(0, 13, size=(2, ))
            rand_render = torch.randint(0, 2, size=(2, ))

            bool_list = [False, False]

            dist = rand[0] * (dist_range[1] - dist_range[0]) + dist_range[0]
            elev = rand[1] * (elev_range[1] - elev_range[0]) + elev_range[0]
            azim = rand[2] * (azim_range[1] - azim_range[0]) + azim_range[0]
            #elev = 30 * randint[0] / 2
            #azim = 30 * randint[1]
            #azim = 90.0
            self.renderer.camera_setting(dist=dist, elev=elev, azim=azim, offset=offset)
            pred_tensor = self.renderer.rendering(self.mesh_data, self.diff_tex, 
                                                  light_enable=render_light_enable, rand_back=False, 
                                                  depth_render=bool_list[rand_render[0]],
                                                  depth_value_inverse=bool_list[rand_render[1]],
                                                  field_sample=field_sample
                                                  )[:, :, :, 0:-1]

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
                if not field_sample:
                    self.diff_tex.img_save(save_path=save_path + f"/tex_ep_{epoch+1}.png", width=tex_size, height=tex_size)

            pred_tensor = pred_tensor.permute(0, 3, 1, 2)

            # SDS with annealation process
            if annealation:
                if epoch <= epochs * ann_threshold:
                    tensor_for_backward, p_loss = guidance.train_step(pred_tensor=pred_tensor, text_embeddings=text_embeddings,
                                                                   latent_input=self.is_latent, min_t=min_t, max_t=max_t)
                else:
                    tensor_for_backward, p_loss = guidance.train_step(pred_tensor=pred_tensor, text_embeddings=text_embeddings,
                                                                   latent_input=self.is_latent, min_t=min_t_ann, max_t=max_t_ann)
            else:
                tensor_for_backward, p_loss = guidance.train_step(pred_tensor=pred_tensor, text_embeddings=text_embeddings,
                                                                   latent_input=self.is_latent, min_t=min_t, max_t=max_t)
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
            if not field_sample:
                self.diff_tex.img_save(save_path=save_path + f"/tex_result.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, offset=offset, elev=25,
                                                           light_enable=render_light_enable, dist=dist_range[1],
                                                           field_sample=field_sample)

            for count, img_tensor in enumerate(img_tensor_list):
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/results_{count}.png", img_array)
            
            self.diff_tex_save(save_path=save_path+"/nth.pth")

    def diff_tex_save(self, save_path):
        self.diff_tex.tex_save(save_path)

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    seed_everything(0)

    mesh_path = "./Assets/3D_Model/Nascar/mesh.obj"
    text_prompt = "a next gen of nascar"
    save_path = "./Experiments/Generative_Texture_MLP/Pineapple/test3"
    mlp_path = "./Assets/Image_MLP/Gaussian_noise_latent/latent_noise.pth"
    #mlp_path = "./Assets/Image_MLP/Gaussian_noise_latent_64/nth.pth"

    #diff_tex = DiffTexture(size=(256, 256), is_latent=True)

    diff_tex = NeuralTextureField(width=32, depth=2, pe_enable=True, input_dim=2)
    #diff_tex.tex_load(tex_path=mlp_path)

    img_size=512
    tex_size=512

    texture_generator = TextureGenerator(mesh_path=mesh_path, diff_tex=diff_tex, is_latent=False)

    #recomanded lr: mlp 256x6 --- 0.0001, 256x2 --- 0.003 mlp 32x6 --- 0.001, 32x2 --- 0.005/0.01
    texture_generator.texture_train(text_prompt=text_prompt, lr=0.01, epochs=4000, save_path=save_path, 
                                    dist_range=[1.1, 1.1], elev_range=[-10.0, 45.0], azim_range=[0.0, 360.0],
                                    info_update_period=100, render_light_enable=True, tex_size=tex_size, 
                                    rendered_img_size=img_size, annealation=True, field_sample=False)

if __name__ == "__main__":
    main()
        





        