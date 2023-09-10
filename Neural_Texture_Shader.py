from typing import Optional, Union
import pytorch3d
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.blending import BlendParams, _get_background_color
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh import shader
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.structures.utils import padded_to_list
from pytorch3d.ops import interpolate_face_attributes
import torch
from pytorch3d.structures.meshes import Meshes
from BRDF import brdf_shading

class NeuralTextureShader(shader.ShaderBase):
    """
    The texels are retreived from a neural represented texture
    """
    def __init__(self,
        diff_tex = None,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        light_enable = True,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
        mesh_data = None,
        rand_back = False,
        depth_render = False,
        depth_value_inverse = False,
        field_sample = False,
        shading_method = 'phong' #phong or brdf
        ):
        super().__init__(device, cameras, lights, materials, blend_params)
        self.diff_tex = diff_tex
        self.mesh = mesh_data['mesh_obj']
        self.verts = mesh_data['verts']
        self.aux = mesh_data['aux']
        self.faces = mesh_data['faces']
        self.light_enable = light_enable
        self.rand_back = rand_back
        self.depth_render = depth_render
        self.depth_value_inverse = depth_value_inverse
        self.field_sample = field_sample
        self.shading_method = shading_method
        self.device = device

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        #texels = self.texture_sample(fragments)
        #texels = self.pixel_coordinate_color(fragments=fragments)
        #texels = self.world_coordinate_color(fragments=fragments)
        if self.depth_render:
            texels_z = self.z_coordinate_color(fragments=fragments)
            texels = texels * texels_z
        elif self.field_sample:
            texels = self.field_texture_sample(fragments=fragments)
        else:
            texels = self.texture_sample(fragments=fragments)

        #texels = self.norm_color(fragments=fragments)
        #texels = self.texture_uv_color(fragments=fragments)
        #texels = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device, requires_grad=True)
        #texels = texels.repeat(torch.unsqueeze(fragments.pix_to_face, 4).size())
        #print("texels:",texels.size())
        #print("test:",test.size())

        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        #print(texels.size())
        if self.light_enable:
            if self.shading_method == 'phong':
            #do not support rendering in latent space
                colors = shader.phong_shading(
                    meshes=meshes,
                    fragments=fragments,
                    texels=texels[:, :, :, :, 0:3],
                    lights=lights,
                    cameras=cameras,
                    materials=materials
                )
            elif self.shading_method == 'brdf':
                colors = brdf_shading(
                    meshes=meshes,
                    fragments=fragments,
                    texels=texels[:, :, :, :, 0:8],
                    lights=lights,
                    cameras=cameras,
                    materials=materials
                )
            else:
                colors = texels

        else:
            colors = texels # no lighting

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = self.softmax_rgb_blend_custom(
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
        z_max = z.max().item()
        z_min = z[z>-1].min().item()
        if self.depth_value_inverse:
            z[z>-1] = (z_max + 0.5*(z_max - z_min) - z[z>-1]) / ((z_max - z_min) * 1.5)
        else:
            z[z>-1] = (z[z>-1] - z_min + 0.5*(z_max - z_min)) / ((z_max - z_min) * 1.5)

        texels = torch.zeros((N, H, W, K, 3), device=self.device)
        texels[:, :, :, :, 0] = z
        texels[:, :, :, :, 1] = z
        texels[:, :, :, :, 2] = z

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
        packing_list = [
            i[j] for i, j in zip([self.verts.to(self.device)], [self.faces.verts_idx.to(self.device)])
        ]
        faces_verts = torch.cat(packing_list)
        pixel_verts = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        ).reshape(-1, 3)

        colors = pixel_verts + 0.5
        texels = colors.reshape(N, H, W, K, 3)
        return texels    
    
    def texture_uv_color(self, fragments: Fragments):
        N, H, W, K = fragments.pix_to_face.size()
        packing_list = [
            i[j] for i, j in zip([self.aux.verts_uvs.to(self.device)], [self.faces.textures_idx.to(self.device)])
        ]
        faces_verts_uvs = torch.cat(packing_list)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        ).reshape(-1, 2) #range [0, 1]
        zero_temp = torch.zeros((H*W, 1), device=self.device)
        colors = torch.cat((pixel_uvs, zero_temp), dim=1)
        texels = colors.reshape(N, H, W, K, 3)
        return texels

    def norm_color(self, fragments: Fragments):
        N, H, W, K = fragments.pix_to_face.size()
        packing_list = [
            i[j] for i, j in zip([self.aux.normals.to(self.device)], [self.faces.normals_idx.to(self.device)])
        ]
        faces_normals = torch.cat(packing_list)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        ).reshape(-1, 3)
        colors = pixel_normals
        texels = colors.reshape(N, H, W, K, 3)
        return texels    
    
    def field_texture_sample(self, fragments: Fragments, nearest=False):
        N, H, W, K = fragments.pix_to_face.size()
        depth = fragments.zbuf
        packing_list = [
            i[j] for i, j in zip([self.verts.to(self.device)], [self.faces.verts_idx.to(self.device)])
        ]
        faces_verts = torch.cat(packing_list)
        pixel_verts = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        ).reshape(-1, 3)

        pixel_verts = pixel_verts * 2.0 #range [-1, 1]
        colors = self.diff_tex(pixel_verts)

        if colors.size()[1] == 3:
            colors = (colors + 1) / 2
        elif colors.size()[1] == 8:
            colors[:, 0:5] = (colors[:, 0:5] + 1) / 2

        colors = colors.reshape(N, H, W, colors.size()[1])
        texels = colors.unsqueeze(3).repeat(1, 1, 1, K, 1)
        return texels 

    def texture_sample(self, fragments: Fragments, nearest=False):
        N, H, W, K = fragments.pix_to_face.size()
        packing_list = [
            i[j] for i, j in zip([self.aux.verts_uvs.to(self.device)], [self.faces.textures_idx.to(self.device)])
        ]
        faces_verts_uvs = torch.cat(packing_list)

        pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs).reshape(-1, 2) #range [0, 1]

        pixel_uvs = pixel_uvs * 2.0 - 1.0 #range [-1, 1]
        temps = pixel_uvs[:, 0].clone()
        pixel_uvs[:, 0] = -pixel_uvs[:, 1]
        pixel_uvs[:, 1] = temps
        colors = self.diff_tex(pixel_uvs)
        if colors.size()[1] == 3:
            colors = (colors + 1) / 2
        elif colors.size()[1] == 8:
            colors[:, 0:5] = (colors[:, 0:5] + 1) / 2
        colors = colors.reshape(N, H, W, colors.size()[1])
        texels = colors.unsqueeze(3).repeat(1, 1, 1, K, 1)
        return texels 

    def field_sample(self, fragments: Fragments, nearest=False):
        N, H, W, K = fragments.pix_to_face.size()
        packing_list = [
            i[j] for i, j in zip([self.aux.verts_uvs.to(self.device)], [self.faces.textures_idx.to(self.device)])
        ]
        faces_verts_uvs = torch.cat(packing_list)
        pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs).reshape(-1, 2) #range [0, 1]
    
        pixel_uvs = pixel_uvs * 2.0 - 1.0 #range [-1, 1]
        temps = pixel_uvs[:, 0].clone()
        pixel_uvs[:, 0] = -pixel_uvs[:, 1]
        pixel_uvs[:, 1] = temps
        colors = self.diff_tex(pixel_uvs)
        if colors.size()[1] == 3:
            colors = (colors + 1) / 2
        elif colors.size()[1] == 8:
            colors[:, 0:5, :, :] = (colors[:, 0:5, :, :] + 1) / 2
        colors = colors.reshape(N, H, W, colors.size()[1])
        #print(colors[256, 256, :])
        texels = colors.unsqueeze(3).repeat(1, 1, 1, K, 1)
        return texels 

    def verts_uvs_list(self):
        if self._verts_uvs_list is None:
            if self.isempty():
                self._verts_uvs_list = [
                    torch.empty((0, 2), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                # The number of vertices in the mesh and in verts_uvs can differ
                # e.g. if a vertex is shared between 3 faces, it can
                # have up to 3 different uv coordinates.
                self._verts_uvs_list = list(self._verts_uvs_padded.unbind(0))
        return self._verts_uvs_list
    
    def faces_uvs_list(self):
        if self._faces_uvs_list is None:
            if self.isempty():
                self._faces_uvs_list = [
                    torch.empty((0, 3), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                self._faces_uvs_list = padded_to_list(
                    self._faces_uvs_padded, split_size=self._num_faces_per_mesh
                )
        return self._faces_uvs_list

    def softmax_rgb_blend_custom(
        self,
        colors: torch.Tensor,
        fragments,
        blend_params: BlendParams,
        znear: Union[float, torch.Tensor] = 1.0,
        zfar: Union[float, torch.Tensor] = 100
    ) -> torch.Tensor:
        """
        RGB and alpha channel blending to return an RGBA image based on the method
        proposed in [1]
        - **RGB** - blend the colors based on the 2D distance based probability map and
            relative z distances.
        - **A** - blend based on the 2D distance based probability map.

        Args:
            colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
            fragments: namedtuple with outputs of rasterization. We use properties
                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - dists: FloatTensor of shape (N, H, W, K) specifying
                the 2D euclidean distance from the center of each pixel
                to each of the top K overlapping faces.
                - zbuf: FloatTensor of shape (N, H, W, K) specifying
                the interpolated depth from each pixel to to each of the
                top K overlapping faces.
            blend_params: instance of BlendParams dataclass containing properties
                - sigma: float, parameter which controls the width of the sigmoid
                function used to calculate the 2D distance based probability.
                Sigma controls the sharpness of the edges of the shape.
                - gamma: float, parameter which controls the scaling of the
                exponential function used to control the opacity of the color.
                - background_color: (3) element list/tuple/torch.Tensor specifying
                the RGB values for the background color.
            znear: float, near clipping plane in the z direction
            zfar: float, far clipping plane in the z direction

        Returns:
            RGBA pixel_colors: (N, H, W, 4)

        [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
        Image-based 3D Reasoning'
        """

        N, H, W, K = fragments.pix_to_face.shape
        D = (colors.size()[4] + 1) #pixel buffer depth
        pixel_colors = torch.ones((N, H, W, D), dtype=colors.dtype, device=colors.device)
        #pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
        background_color = torch.ones(D-1, dtype=colors.dtype, device=colors.device) * 0.0
        if self.rand_back:
            background_color =  torch.rand(D-1, dtype=colors.dtype, device=colors.device) * 2 - 1#random background color

        #background_color = _get_background_color(blend_params, fragments.pix_to_face.device)

        # Weight for background color
        eps = 1e-10

        # Mask for padded pixels.
        mask = fragments.pix_to_face >= 0

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

        # The cumulative product ensures that alpha will be 0.0 if at least 1
        # face fully covers the pixel as for that face, prob will be 1.0.
        # This results in a multiplication by 0.0 because of the (1.0 - prob)
        # term. Therefore 1.0 - alpha will be 1.0.
        alpha = torch.prod((1.0 - prob_map), dim=-1)

        # Weights for each face. Adjust the exponential by the max z to prevent
        # overflow. zbuf shape (N, H, W, K), find max over K.
        # TODO: there may still be some instability in the exponent calculation.

        # Reshape to be compatible with (N, H, W, K) values in fragments
        if torch.is_tensor(zfar):
            # pyre-fixme[16]
            zfar = zfar[:, None, None, None]
        if torch.is_tensor(znear):
            # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
            #  `__getitem__`.
            znear = znear[:, None, None, None]

        z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1)[..., None] + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
        weighted_background = delta * background_color
        pixel_colors[..., :(D-1)] = (weighted_colors + weighted_background) / denom
        pixel_colors[..., D-1] = 1.0 - alpha

        return pixel_colors


def main():
    from pytorch3d import renderer
    from pytorch3d import io
    import matplotlib.pyplot as plt
    import torch
    from Neural_Texture_Shader import NeuralTextureShader
    from utils import device
    from Neural_Texture_Field import NeuralTextureField
    #mesh
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    #mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    verts, faces, aux = io.load_obj(mesh_path)
    tex_net = NeuralTextureField(width=512, depth=3, input_dim=2, pe_enable=True)
    tex_net.load_state_dict(torch.load("./Experiments/mlp_represented_image_training _entire_image/test5_validate/nth.pth"))

    #camera
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
            tex_net=tex_net,
            device=device,
            cameras=camera,
            light_enable=True,
            lights=light,
            aux=aux,
            faces=faces
        )
    )

    #rendering
    image_tensor = mesh_renderer(mesh_obj)
    #loss = image_tensor.sum()
    #loss.backward()
    img = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    plt.imshow(img)
    plt.show()

def main_2():
    from pytorch3d import renderer
    from pytorch3d import io
    import matplotlib.pyplot as plt
    import torch
    from Neural_Texture_Shader import NeuralTextureShader
    from utils import device
    from Differentiable_Texture import DiffTexture
    #mesh
    mesh_path = "./Assets/3D_Model/Cow/cow.obj"
    mesh_obj = io.load_objs_as_meshes([mesh_path], device=device)
    verts, faces, aux = io.load_obj(mesh_path)
    diff_tex = DiffTexture(size=(2048, 2048))

    #camera
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
            diff_tex=diff_tex,
            device=device,
            cameras=camera,
            light_enable=True,
            lights=light,
            aux=aux,
            faces=faces
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
    main_2()