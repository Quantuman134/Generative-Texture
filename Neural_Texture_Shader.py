from typing import Optional
import pytorch3d
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh import shader
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.utils import TensorProperties
import torch
from pytorch3d.structures.meshes import Meshes

class NeuralTextureShader(shader.ShaderBase):
    """
    The texels are retreived from a neural represented texture
    """
    def __init__(self,
        tex_mlp = None,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
        ):
        super().__init__(device, cameras, lights, materials, blend_params)
        self.tex_mlp = tex_mlp

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        #texels = self.texture_sample(fragments)
        #texels = self.pixel_coordinate_color(fragments=fragments)
        #texels = self.z_coordinate_color(fragments=fragments)
        #texels = self.world_coordinate_color(fragments=fragments)
        texels = self.texture_sample(tex_mlp=self.tex_mlp, fragments=fragments)
        #texels = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device, requires_grad=True)
        #texels = texels.repeat(torch.unsqueeze(fragments.pix_to_face, 4).size())
        #print("texels:",texels.size())
        #print("test:",test.size())


        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = shader.phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = shader.softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images
    
    def pixel_coordinate_color(self, fragments: Fragments):
        #r represent the indice i, and g represent the indice j
        N, H, W, K = fragments.pix_to_face.size()
        r = (torch.arange(start=0, end=H, device=self.device)/H).unsqueeze(0).transpose(0, 1).repeat(1, W)
        g = (torch.arange(start=0, end=W, device=self.device)/W).unsqueeze(0).repeat(H, 1)
        grid = torch.ones((H, W, 3), device=self.device)
        grid[:, :, 0] *= r
        grid[:, :, 1] *= g
        grid[:, :, 2] = 0
        texels = grid.unsqueeze(0).unsqueeze(3).repeat(N, 1, 1, K, 1)
        return texels
    
    def z_coordinate_color(self, fragments: Fragments):
        #r represent the indice i, and g represent the indice j
        N, H, W, K = fragments.pix_to_face.size()
        z = fragments.zbuf
        z_near = self.cameras.znear
        z_far = self.cameras.zfar - 96
        print(z.size())
        texels = torch.zeros((N, H, W, K, 3), device=self.device)
        texels[:, :, :, :, 0] = (z - z_near)/(z_far - z_near)
        texels[:, :, :, :, 1] = (z - z_near)/(z_far - z_near)
        texels[:, :, :, :, 2] = (z - z_near)/(z_far - z_near)
        print(texels[:, :, :, :, 2].size())
        #texels[texels<0] == 0

        return texels

    def ndc_coordinate_color(self, fragments: Fragments):
        N, H, W, K = fragments.pix_to_face.size()
        z_near = self.cameras.znear
        z_far = self.cameras.zfar - 94
        depth = (fragments.zbuf - z_near) / (z_far - z_near)
        #ndc coord
        h = ((torch.arange(start=0, end=H, device=self.device) * 2 + 1.0) / H - 1.0).unsqueeze(0).transpose(0, 1).repeat(1, W)
        w = ((torch.arange(start=0, end=W, device=self.device) * 2 + 1.0) / W - 1.0).unsqueeze(0).repeat(H, 1)
        ndc_coordinate = torch.ones((H, W, 3), device=self.device)
        ndc_coordinate[:, :, 0] *= (-w + 1) / 2
        ndc_coordinate[:, :, 1] *= (-h + 1) / 2
        ndc_coordinate[:, :, 2] *= depth.reshape(H, W)
        texels = ndc_coordinate.unsqueeze(0).unsqueeze(3).repeat(N, 1, 1, K, 1)

        return texels

    def world_coordinate_color(self, fragments: Fragments):
        #r represent the indice x, g represent the indice y, b represent the indice z
        N, H, W, K = fragments.pix_to_face.size()
        depth = fragments.zbuf
        #ndc coord
        h = ((torch.arange(start=0, end=H, device=self.device) * 2 + 1.0) / H - 1.0).unsqueeze(0).transpose(0, 1).repeat(1, W)
        w = ((torch.arange(start=0, end=W, device=self.device) * 2 + 1.0) / W - 1.0).unsqueeze(0).repeat(H, 1)
        ndc_coordinate = torch.ones((H, W, 3), device=self.device)
        ndc_coordinate[:, :, 0] *= -w
        ndc_coordinate[:, :, 1] *= -h
        ndc_coordinate[:, :, 2] *= depth.reshape(H, W)
        world_coordinate = self.cameras.unproject_points(ndc_coordinate, world_coordinates=True)
        #color normalize
        world_coordinate[:, :, 0] = (world_coordinate[:, :, 0] + 1) / 2
        world_coordinate[:, :, 1] = (world_coordinate[:, :, 1] + 1) / 2
        world_coordinate[:, :, 2] = (world_coordinate[:, :, 2] + 3) / 4

        texels = world_coordinate.unsqueeze(0).unsqueeze(3).repeat(N, 1, 1, K, 1)
        return texels    

    def texture_sample(self, tex_mlp, fragments: Fragments):
        N, H, W, K = fragments.pix_to_face.size()
        depth = fragments.zbuf.reshape(H, W)
        indices_no_depth = (depth == -1).nonzero(as_tuple=False)
        #ndc coord
        h = ((torch.arange(start=0, end=H, device=self.device) * 2 + 1.0) / H - 1.0).unsqueeze(0).transpose(0, 1).repeat(1, W)
        w = ((torch.arange(start=0, end=W, device=self.device) * 2 + 1.0) / W - 1.0).unsqueeze(0).repeat(H, 1)
        ndc_coordinate = torch.ones((H, W, 3), device=self.device)
        ndc_coordinate[:, :, 0] *= -w
        ndc_coordinate[:, :, 1] *= -h
        ndc_coordinate[:, :, 2] *= depth
        world_coordinate = self.cameras.unproject_points(ndc_coordinate, world_coordinates=True)
        #world_coordinate = world_coordinate.reshape(H * W, 3) * 1000
        world_coordinate = world_coordinate.reshape(-1, 3)
        colors = ((tex_mlp(world_coordinate) + 1) / 2).reshape(H, W, 3)
        colors[indices_no_depth[:, 0], indices_no_depth[:, 1], :] = 0
        print(colors[256, 256, :])
        texels = colors.unsqueeze(0).unsqueeze(3).repeat(N, 1, 1, K, 1)
        #print(texels[0, 0, 2, 0, :])
        return texels 
    
def main():
    from pytorch3d import renderer
    from pytorch3d import io
    import matplotlib.pyplot as plt
    import torch
    from Neural_Texture_Shader import NeuralTextureShader
    from utils import device
    from Neural_Texture_Field import NeuralTextureField
    #mesh
    #mesh_path = "./Assets/3D_Model/Cube/cube.obj"
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    tex_mlp = NeuralTextureField(width=512, depth=3, input_dim=3, pe_enable=True)

    #camera
    R, T = renderer.look_at_view_transform(2.7, 0, 135)
    camera = renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    #light
    light = renderer.PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    #renderer
    raster_setting = renderer.RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)

    mesh_renderer = renderer.MeshRenderer(
        rasterizer=renderer.MeshRasterizer(
            cameras=camera,
            raster_settings=raster_setting
        ),
        shader = NeuralTextureShader(
            tex_mlp=tex_mlp,
            device=device,
            cameras=camera,
            lights=light
        )
    )

    #rendering
    image_tensor = mesh_renderer(mesh_obj)
    #loss = image_tensor.sum()
    #loss.backward()
    img = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()