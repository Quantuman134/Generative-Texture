from utils import seed_everything, save_img_tensor
from pytorch3d import io
import torch
from Neural_Texture_Field import NeuralTextureField
from Differentiable_Texture import DiffTexture
from Stable_Diffusion import StableDiffusion
from Control_Net import ControlNet
from Neural_Texture_Renderer import NeuralTextureRenderer
import time
import numpy as np
import matplotlib.pyplot as plt
import utils

class TextureGenerator:
    def __init__(self, mesh_path, diff_tex=None, is_latent=False, device=utils.device, rank=0) -> None:
        self.device = device
        self.rank = rank # multi gpu
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
        self.renderer = NeuralTextureRenderer(device=device)
        self.is_latent=is_latent
    
    # offset: the camera transition offset that point to center of object
    # dist,elev,azim range: the range of camera configuration, the maximum dist will be implemented
    # in the distance of render_around()
    # info_update_period: the period of saving and printing information of training, whose unit is epoch
     
    def texture_train(self, text_prompt, guidance_scale, lr, epochs, dm='sd', save_path=None, 
                      offset=[0.0, 0.0, 0.0], dist_range=[1.0, 2.0], 
                      elev_range=[0.0, 360.0], azim_range=[0.0, 360.0],
                      info_update_period=500, render_light_enable=False,
                      tex_size=512, rendered_img_size=512, annealation=False,
                      field_sample=False, brdf=False, num_inference_steps=1000, controlnet_conditioning_scale=1.0,
                      low_threshold=100, high_threshold=200):
        if brdf:
            shading_method = 'brdf'
        else:
            shading_method = 'phong'

        # highlight: a temp code line, normal is 512
        self.renderer.rasterization_setting(image_size=rendered_img_size)

        if self.is_latent:
            self.renderer.rasterization_setting(image_size=64)
        
        offset = torch.tensor([offset])

        # light setting
        if brdf:
            self.renderer.light_setting(directions=[[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], 
                           intensities=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], multi_lights=True)
        else:
            self.renderer.light_setting(directions=[[1, 1, 1]])
        

        #save initial data
        if (save_path is not None) and (self.rank == 0):
            if not (field_sample or brdf):
                self.diff_tex.img_save(save_path=save_path + f"/tex_initial.png", width=tex_size, height=tex_size)
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, offset=offset, elev=25, 
                                                          light_enable=render_light_enable, dist=dist_range[1],
                                                          field_sample=field_sample, shading_method=shading_method)
            
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
        
        if dm == 'sd':
            guidance = StableDiffusion(device=self.device, num_inference_steps=num_inference_steps)
        elif dm == 'cn':
            guidance = ControlNet(controlnet_library='lllyasviel/sd-controlnet-canny', device=self.device, num_inference_steps=num_inference_steps)
        elif dm == 'nm':
            guidance = ControlNet(controlnet_library='lllyasviel/sd-controlnet-normal', device=self.device, num_inference_steps=num_inference_steps)
        
        guidance.eval()
        text_embeddings = guidance.get_text_embeds(text_prompt, '')

        # for ControlNet, we need a white texture for edge detection
        if dm == 'cn':
            white_tex = NeuralTextureField(width=2, depth=1, pe_enable=False, input_dim=self.diff_tex.module.input_dim, brdf=brdf, device=self.device)

        print(f"[INFO{ self.device}] traning starts")
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

            rand = torch.rand(12)
            randint = torch.randint(0, 13, size=(2, ))
            rand_render = torch.randint(0, 2, size=(2, ))

            bool_list = [False, False]

            dist = rand[0 + self.rank * 3] * (dist_range[1] - dist_range[0]) + dist_range[0]
            elev = rand[1 + self.rank * 3] * (elev_range[1] - elev_range[0]) + elev_range[0]
            azim = rand[2 + self.rank * 3] * (azim_range[1] - azim_range[0]) + azim_range[0]
            #elev = 30 * randint[0] / 2
            #azim = 30 * randint[1]
            #azim = 90.0
            self.renderer.camera_setting(dist=dist, elev=elev, azim=azim, offset=offset)
            pred_tensor = self.renderer.rendering(self.mesh_data, self.diff_tex, 
                                                  light_enable=render_light_enable, rand_back=False, 
                                                  depth_render=bool_list[rand_render[0]],
                                                  depth_value_inverse=bool_list[rand_render[1]],
                                                  field_sample=field_sample, shading_method=shading_method
                                                  )[:, :, :, 0:-1]
            
            if dm == 'cn':
                img_for_edge_detection = self.renderer.rendering(self.mesh_data, white_tex, 
                                                  light_enable=render_light_enable, rand_back=False, 
                                                  depth_render=bool_list[rand_render[0]],
                                                  depth_value_inverse=bool_list[rand_render[1]],
                                                  field_sample=field_sample, shading_method=shading_method
                                                  )[:, :, :, 0:-1].permute(0, 3, 1, 2)
                
                edge_map = guidance.edge_detect(imgs=img_for_edge_detection, low_threshold=low_threshold, high_threshold=high_threshold)

            elif dm == 'nm':
                norm_map = self.renderer.rendering(self.mesh_data, self.diff_tex, 
                                                  light_enable=render_light_enable, rand_back=False, 
                                                  depth_render=bool_list[rand_render[0]],
                                                  depth_value_inverse=bool_list[rand_render[1]],
                                                  field_sample=field_sample, shading_method='norm'
                                                  )[:, :, :, 0:-1].permute(0, 3, 1, 2)

            # save mediate results
            if (save_path is not None) and ((epoch+1) % info_update_period == 0) and (self.rank == 0):
                
                img_tensor = pred_tensor.permute(0, 3, 1, 2)
                #latent to RGB
                if self.is_latent:
                    pass
                    #img_tensor = img_tensor.permute(0, 3, 1, 2)
                    #img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                #img_array = img_tensor[0, :, :, :].cpu().detach().numpy()
                #img_array = np.clip(img_array, 0, 1)
                #plt.imsave(save_path + f"/ep_{epoch+1}.png", img_array)
                save_img_tensor(img_tensor, save_dir=save_path + f"/ep_{epoch+1}.png")

                if dm == 'cn':
                    save_img_tensor(edge_map, save_dir=save_path + f"/edge_{epoch+1}.png")
                elif dm == 'nm':
                    save_img_tensor(norm_map, save_dir=save_path + f"/norm_{epoch+1}.png")

                if not (field_sample or brdf):
                    self.diff_tex.img_save(save_path=save_path + f"/tex_ep_{epoch+1}.png", width=tex_size, height=tex_size)

            pred_tensor = pred_tensor.permute(0, 3, 1, 2)

            # SDS with annealation process
            if annealation:
                if epoch >= epochs * ann_threshold:
                    min_t = min_t_ann
                    max_t = max_t_ann
            
            if dm == 'sd':
                tensor_for_backward, p_loss = self.SD_train_step(guidance=guidance, pred_tensor=pred_tensor, text_embeddings=text_embeddings,
                                                                        min_t=min_t, max_t=max_t, guidance_scale=guidance_scale)
            elif dm == 'cn':
                tensor_for_backward, p_loss = self.CN_train_step(guidance=guidance, pred_tensor=pred_tensor, cond_imgs=edge_map, text_embeddings=text_embeddings,
                                                                        min_t=min_t, max_t=max_t, guidance_scale=guidance_scale, controlnet_conditioning_scale=controlnet_conditioning_scale)
            elif dm == 'nm':
                tensor_for_backward, p_loss = self.CN_train_step(guidance=guidance, pred_tensor=pred_tensor, cond_imgs=norm_map, text_embeddings=text_embeddings,
                                                                        min_t=min_t, max_t=max_t, guidance_scale=guidance_scale, controlnet_conditioning_scale=controlnet_conditioning_scale)

            tensor_for_backward.backward()
            optimizer.step()
            total_loss += p_loss
            
            if (epoch+1) % info_update_period == 0  and (self.rank == 0):
                end_t = time.time()
                print(f"[INFO {self.device}] epoch {epoch+1} takes {(end_t - start_t):.4f} seconds. loss = {total_loss/info_update_period}")
                total_loss = 0
                start_t = end_t
            
        print(f"[INFO {self.device}] traning ends")

        del guidance
        
        if (save_path is not None) and (self.rank == 0):
            if not (field_sample or brdf):
                self.diff_tex.img_save(save_path=save_path + f"/tex_result.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.diff_tex, offset=offset, elev=25,
                                                           light_enable=render_light_enable, dist=dist_range[1],
                                                           field_sample=field_sample, shading_method=shading_method)

            for count, img_tensor in enumerate(img_tensor_list):
                #latent to RGB
                if self.is_latent:
                    img_tensor = img_tensor[:, :, :, 0:4].permute(0, 3, 1, 2)
                    img_tensor = utils.decode_latents(img_tensor).permute(0, 2, 3, 1)

                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/results_{count}.png", img_array)
            
            #disable temprarily
            #self.diff_tex_save(save_path=save_path+"/nth.pth")

    def diff_tex_save(self, save_path):
        self.diff_tex.tex_save(save_path)

    def SD_train_step(self, guidance:StableDiffusion, pred_tensor, text_embeddings, min_t, max_t, guidance_scale=7.5):
        tensor_for_backward, p_loss = guidance.train_step(pred_tensor=pred_tensor, text_embeddings=text_embeddings,
                                                        latent_input=self.is_latent, min_t=min_t, max_t=max_t, guidance_scale=guidance_scale)

        return tensor_for_backward, p_loss
    
    def CN_train_step(self, guidance:ControlNet, pred_tensor, cond_imgs, text_embeddings,
                      min_t=0.02, max_t=0.98, guidance_scale=7.5, controlnet_conditioning_scale=1.0):
        # prepare edge map
        #edge_map = guidance.edge_detect(imgs=img_for_edge_detection)
        cond_imgs = torch.cat([cond_imgs] * 2)

        tensor_for_backward, p_loss = guidance.train_step(pred_tensor=pred_tensor, cond_imgs=cond_imgs, text_embeddings=text_embeddings, 
                                                          min_t=min_t, max_t=max_t, guidance_scale=guidance_scale, controlnet_conditioning_scale=controlnet_conditioning_scale)

        return tensor_for_backward, p_loss

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    seed_everything(14551)

    mesh_path = "./Assets/3D_Model/Pineapple/mesh.obj"
    text_prompt = "a pineapple"
    save_path = "./Experiments/Generative_Texture_MLP/Pineapple/test1"
    mlp_path = "./Assets/Image_MLP/Gaussian_noise_latent/latent_noise.pth"
    #mlp_path = "./Assets/Image_MLP/Gaussian_noise_latent_64/nth.pth"
    brdf = True
    input_dim = 3

    field_sample = False
    if input_dim == 3:
        field_sample = True

    #diff_tex = DiffTexture(size=(256, 256), is_latent=True)

    diff_tex = NeuralTextureField(width=32, depth=2, pe_enable=True, input_dim=input_dim, brdf=brdf, device=utils.device)
    #diff_tex.tex_load(tex_path=mlp_path)

    guidance_scale = 100; # 100

    img_size=512
    tex_size=512

    texture_generator = TextureGenerator(mesh_path=mesh_path, diff_tex=diff_tex, is_latent=False)

    #recomanded lr: mlp 256x6 --- 0.0001, 256x2 --- 0.003 mlp 32x6 --- 0.001, 32x2 --- 0.005/0.01
    texture_generator.texture_train(text_prompt=text_prompt, guidance_scale=guidance_scale, lr=0.01, epochs=1000, save_path=save_path, 
                                    dist_range=[1.2, 1.2], elev_range=[-10.0, 45.0], azim_range=[0.0, 360.0],
                                    info_update_period=50, render_light_enable=True, tex_size=tex_size, 
                                    rendered_img_size=img_size, annealation=True, field_sample=field_sample, brdf=brdf)
    
if __name__ == "__main__":
    main()
        





        