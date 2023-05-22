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
    def rendering(self, mesh_data, tex_net):
        mesh_renderer = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_setting
            ),
            shader=NeuralTextureShader(
                tex_net=tex_net,
                device=self.device,
                cameras=self.cameras,
                light_enable=True,
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

    def light_setting(self, locations=[[0.0, 10.0, 10.0]]):
        self.lights = renderer.PointLights(location=locations, device=self.device)
    
    def render_around(self, mesh_data, tex_net):
        dist = 2.7
        elev = 0
        image_tensor_list = []
        for azim in range(0, 360, 45):
            self.camera_setting(dist=dist, elev=elev, azim=azim)
            image_tensor = self.rendering(mesh_data, tex_net)
            image_tensor_list.append(image_tensor)
        
        return image_tensor_list
            

def main():
    import torch
    from pytorch3d import io
    import matplotlib.pyplot as plt 
    import numpy as np

    renderer = NeuralTextureRenderer()
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mlp_path = "./Assets/Image_MLP/gaussian_noise/nth.pt"
    tex_net = torch.jit.load(mlp_path)
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    _, faces, aux = io.load_obj(mesh_path, device=device)
    mesh_data = {'mesh_obj': mesh_obj, 'faces': faces, 'aux': aux}

    image_tensor = renderer.rendering(mesh_data=mesh_data, tex_net=tex_net)
    image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    image_array = np.clip(image_array, 0, 1)
    plt.imshow(image_array)
    plt.show()

if __name__ == "__main__":
    main()