from utils import device
from pytorch3d import renderer
from Neural_Texture_Shader import NeuralTextureShader

class NeuralTextureRenderer:
    def __init__(self) -> None:
        self.device = device
        self.rasterization_setting()
        self.camera_setting()
        self.light_setting()

    # rendered image: tensor[N, H, W, 4], N: image number, 4: RGBA
    def rendering(self, mesh_data, diff_tex, light_enable=False):
        mesh_renderer = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_setting
            ),
            shader=NeuralTextureShader(
                diff_tex=diff_tex,
                device=self.device,
                cameras=self.cameras,
                light_enable=light_enable,
                lights=self.lights,
                faces=mesh_data['faces'],
                aux=mesh_data['aux']
            )
        )

        return mesh_renderer(mesh_data['mesh_obj'])

    def camera_setting(self, dist=2.0, elev=0, azim=135):
        R, T = renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim)
        self.cameras = renderer.FoVPerspectiveCameras(device=self.device, R=R, T=T)

    def rasterization_setting(self, image_size=512, blur_radius=0.0, face_per_pixel=1):
        self.raster_setting = renderer.RasterizationSettings(image_size=image_size, blur_radius=blur_radius, faces_per_pixel=face_per_pixel)

    def light_setting(self, locations=[[10.0, 10.0, 10.0]]):
        self.lights = renderer.PointLights(location=locations, device=self.device)
    
    def render_around(self, mesh_data, diff_tex, dist=2.5, light_enable=False):
        elev = 45
        image_tensor_list = []
        for azim in range(0, 360, 45):
            self.camera_setting(dist=dist, elev=elev, azim=azim)
            image_tensor = self.rendering(mesh_data, diff_tex, light_enable=light_enable)
            image_tensor_list.append(image_tensor)
        
        return image_tensor_list
            
#RGB space
def main():
    import torch
    from pytorch3d import io
    import matplotlib.pyplot as plt 
    import numpy as np
    from Differentiable_Texture import DiffTexture

    renderer = NeuralTextureRenderer()
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    #mlp_path = "./Assets/Image_MLP/gaussian_noise/nth.pt"
    #diff_tex = torch.jit.load(mlp_path)
    diff_tex = DiffTexture(size=(512, 512), is_latent=False)
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    _, faces, aux = io.load_obj(mesh_path, device=device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    renderer.camera_setting(dist=2.0, elev=0, azim=135)
    renderer.rasterization_setting(image_size=512)
    image_tensor = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=True)
    image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    image_array = np.clip(image_array, 0, 1)
    plt.imshow(image_array)
    plt.show()


#latent space
def main_2():
    import torch
    from pytorch3d import io
    import matplotlib.pyplot as plt 
    import numpy as np
    from Differentiable_Texture import DiffTexture
    import utils

    renderer = NeuralTextureRenderer()
    mesh_path = "./Assets/3D_Model/Square/square.obj"
    tex_path = "./Experiments/Generative_Texture_2/Diff_Texture_Square/latent_space_64_average/tex.pth"
    #diff_tex = torch.jit.load(mlp_path)
    diff_tex = DiffTexture(size=(64, 64), is_latent=True)
    diff_tex.tex_load(tex_path)
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    _, faces, aux = io.load_obj(mesh_path, device=device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    renderer.camera_setting(dist=1.20, elev=0, azim=0)
    renderer.rasterization_setting(image_size=64)

    image_tensor = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=False)

    #latent to rgb
    image_tensor = utils.decode_latents(image_tensor[:, :, :, 0:-1].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    image_array = np.clip(image_array, 0, 1)

    plt.imshow(image_array)
    plt.show()



if __name__ == "__main__":
    main_2()