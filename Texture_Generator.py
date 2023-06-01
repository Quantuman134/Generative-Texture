from utils import device, seed_everything
from pytorch3d import io
import torch
from Neural_Texture_Field import NeuralTextureField
from Stable_Diffusion import StableDiffusion
from Neural_Texture_Renderer import NeuralTextureRenderer
import time
import numpy as np
import matplotlib.pyplot as plt

class TextureGenerator:
    def __init__(self, mesh_path, tex_net=None) -> None:
        self.device = device
        mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
        _, faces, aux = io.load_obj(mesh_path, device=device)
        self.mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}
        self.tex_net = tex_net
        if tex_net is None:
            self.tex_net = NeuralTextureField()
        self.renderer = NeuralTextureRenderer()
        
    
    def texture_train(self, text_prompt, lr, epochs, save_path=None):
        optimizer = torch.optim.Adam(self.tex_net.parameters(), lr=lr)
        guidance = StableDiffusion(device=self.device)
        text_embeddings = guidance.get_text_embeds(text_prompt, '')
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        
        #save initial data
        if save_path is not None:
            self.tex_net.img_save(save_path=save_path + f"/tex_initial.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.tex_net)
            i = 0
            for img_tensor in img_tensor_list:
                i += 1
                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/initial_{i}.png", img_array)
        del img_tensor_list
                
        #range of camera position
        dist_range = [2.0, 3.0]
        elev_range = [0.0, 360.0]
        azim_range = [0.0, 360.0]

        print(f"[INFO] traning starts")
        for epoch in range(epochs):
            start_t = time.time()

            rand = torch.rand(3)
            dist = rand[0] * (dist_range[1] - dist_range[0]) + dist_range[0]
            elev = rand[1] * (elev_range[1] - elev_range[0]) + elev_range[0]
            azim = rand[2] * (azim_range[1] - azim_range[0]) + azim_range[0]
            self.renderer.camera_setting(dist=dist, elev=elev, azim=azim)
            img_pred = self.renderer.rendering(self.mesh_data, self.tex_net)[:, :, :, 0:3]

            # save mediate results
            if save_path is not None:
                if(epoch+1) % 100 == 0:
                    img_array = img_pred[0, :, :, :].cpu().detach().numpy()
                    img_array = np.clip(img_array, 0, 1)
                    plt.imsave(save_path + f"/ep_{epoch+1}.png", img_array)
                    self.tex_net.img_save(save_path=save_path + f"/tex_ep_{epoch+1}.png")

            img_pred = img_pred.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            loss = guidance.train_step(pred_rgb=img_pred, text_embeddings=text_embeddings)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            end_t = time.time()
            if True:
                print(f"[INFO] epoch {epoch} takes {(end_t - start_t):.4f} seconds.")
            
        print(f"[INFO] traning ends")

        del guidance
        
        if save_path is not None:
            self.tex_net.img_save(save_path=save_path + f"/tex_result.png")
            img_tensor_list = self.renderer.render_around(self.mesh_data, self.tex_net)
            i = 0
            for img_tensor in img_tensor_list:
                i += 1
                img_array = img_tensor[0, :, :, 0:3].cpu().detach().numpy()
                img_array = np.clip(img_array, 0, 1)
                plt.imsave(save_path + f"/results_{i}.png", img_array)

    def tex_net_save(self, save_path):
        self.tex_net.net_save(save_path)

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mlp_path = "./Assets/Image_MLP/gaussian_noise/nth.pt"
    tex_net = torch.jit.load(mlp_path)
    texture_generator = TextureGenerator(mesh_path=mesh_path, tex_net=tex_net)
    texture_generator.texture_train("", lr=0.00001, epochs=10)


if __name__ == "__main__":
    main()
        





        