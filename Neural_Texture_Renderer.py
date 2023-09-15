import utils
from pytorch3d import renderer
from Neural_Texture_Shader import NeuralTextureShader
import torch
from Multi_Directional_Lights import MultiDirectionalLights

class NeuralTextureRenderer:
    def __init__(self, offset=[0.0, 0.0, 0.0], device=utils.device) -> None:
        self.device = device
        self.offset = torch.tensor([offset]) #to center of object
        self.rasterization_setting()
        self.camera_setting(offset=self.offset)
        self.light_setting()

    # rendered image: tensor[N, H, W, 4], N: image number, 4: RGBA
    def rendering(self, mesh_data, diff_tex, light_enable=False, rand_back=False, 
                  depth_render=False, depth_value_inverse=False, field_sample=False,
                  shading_method='phong'):
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
                mesh_data=mesh_data,
                rand_back=rand_back,
                depth_render=depth_render,
                depth_value_inverse=depth_value_inverse,
                field_sample=field_sample,
                shading_method=shading_method
            )
        )

        return mesh_renderer(mesh_data['mesh_obj'])

    def camera_setting(self, dist=2.0, elev=0, azim=135, offset=torch.tensor([[0, 0, 0]])):
        R, T = renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim)
        T += offset
        self.cameras = renderer.FoVPerspectiveCameras(device=self.device, R=R, T=T)

    def rasterization_setting(self, image_size=512, blur_radius=0.0, face_per_pixel=1):
        self.raster_setting = renderer.RasterizationSettings(image_size=image_size, blur_radius=blur_radius, faces_per_pixel=face_per_pixel)

    def light_setting(self, directions=[[1.0, 1.0, 1.0]], intensities = [1.0], multi_lights=False):
        self.lights = renderer.DirectionalLights(direction=directions, device=self.device)
        if multi_lights:
            self.lights = MultiDirectionalLights(directions=directions, intensities=intensities, device=self.device)
    
    def render_around(self, mesh_data, diff_tex, dist=2.5, elev=45, offset=torch.tensor([[0, 0, 0]]), light_enable=False , rand_back=False, depth_render=False, depth_value_inverse=False, field_sample=False, shading_method='phong'):
        image_tensor_list = []
        for azim in range(0, 360, 45):
            self.camera_setting(dist=dist, elev=elev, azim=azim, offset=offset)
            image_tensor = self.rendering(mesh_data, diff_tex, light_enable=light_enable, rand_back=rand_back, depth_render=depth_render, depth_value_inverse=depth_value_inverse, field_sample=field_sample, shading_method=shading_method)
            image_tensor_list.append(image_tensor)
        
        return image_tensor_list
            
#RGB space
def main():
    import torch
    from pytorch3d import io
    import matplotlib.pyplot as plt 
    import numpy as np
    from Differentiable_Texture import DiffTexture
    from PIL import Image
    from torchvision import transforms
    from Neural_Texture_Field import NeuralTextureField

    renderer = NeuralTextureRenderer()
    #mesh_path = "./Assets/3D_Model/Basketball/basketball.obj"
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    image_path = "./Assets/3D_Model/Cow/cow_texture.png"
    save_path = "./Experiments/BRDF_test/test1"
    mlp_path = "./Assets/Image_MLP/Gaussian_noise_latent/latent_noise.pth"

    # differentiable texture
    diff_tex = DiffTexture(size=(512, 512), is_latent=False)
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    diff_tex.set_image(image_tensor)

    # mlp texture
    #diff_tex = NeuralTextureField(width=32, depth=2, input_dim=3, brdf=True)
    #diff_tex.tex_load(mlp_path)

    mesh_obj = io.load_objs_as_meshes([mesh_path], device=utils.device)

    verts_packed = mesh_obj.verts_packed()
    
    verts_max = verts_packed.max(dim=0).values
    verts_min = verts_packed.min(dim=0).values
    max_length = (verts_max - verts_min).max().item()
    center = (verts_max + verts_min)/2

    verts_list = mesh_obj.verts_list()
    verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list] #[-0.5, 0.5]
    mesh_obj._verts_packed = (verts_packed - center)/max_length

    verts, faces, aux = io.load_obj(mesh_path, device=utils.device)
    verts = (verts - center)/max_length

    mesh_data = {'mesh_obj': mesh_obj,'verts': verts, 'faces': faces, 'aux': aux}

    offset = torch.tensor([0, 0, 0])
    renderer.camera_setting(dist=1.1, elev=0, azim=0, offset=offset)
    renderer.rasterization_setting(image_size=512)
    #renderer.light_setting(directions=[[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], 
    #                       intensities=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], multi_lights=True)

    image_tensor = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=True, field_sample=True, shading_method='phong')
    image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    image_array = np.clip(image_array, 0, 1)
    plt.imshow(image_array)
    plt.show()

    image_tensors = renderer.render_around(mesh_data=mesh_data, diff_tex=diff_tex, dist=1.0, offset=offset, elev=0, 
                                           light_enable=True, depth_render=False, depth_value_inverse=True, field_sample=False,
                                           shading_method='phong')
    for count, image_tensor in enumerate(image_tensors):
        image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
        image_array = np.clip(image_array, 0, 1)
        plt.imsave(save_path+f"/brdf_{count}.png", image_array)


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
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=utils.device)
    _, faces, aux = io.load_obj(mesh_path, device=utils.device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    renderer.camera_setting(dist=1.20, elev=0, azim=0)
    renderer.rasterization_setting(image_size=64)
    renderer.light_setting(diffuse_color=[[0.8, 0.8, 0.8]], ambient_color=[[0.2, 0.2, 0.2]])

    image_tensor = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, light_enable=False)

    #latent to rgb
    image_tensor = utils.decode_latents(image_tensor[:, :, :, 0:-1].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    image_array = np.clip(image_array, 0, 1)

    plt.imshow(image_array)
    plt.show()


if __name__ == "__main__":
    main()