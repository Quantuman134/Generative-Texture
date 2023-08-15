import torch
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from utils import device

MIN_DIELECTRICS_F0 = 0.04
PI = 3.141592653589
ONE_OVER_PI = (1.0 / PI)
    
def brdf_shading(
    meshes, fragments, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Input:
        texels: (N, H, W, K, C)

    Returns:
        colors: (N, H, W, K, 3)
    """   

    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    ) # pixel_normals (1, H, W, 1, 3)

    pixel_normals = (pixel_normals + texels[:, :, :, :, 5:8])
    texels[:, :, :, :, 0:5] = (texels[:, :, :, :, 0:5] + 1) * 0.5

    colors = torch.zeros_like(pixel_normals)

    for i in range(lights.num):
        light = lights.lights[i]

        brdf_data = brdf_data_cal(pixel_normals, pixel_coords_in_camera, light, cameras, texels)

        specular = specular_lighting(brdf_data)
        diffuse = diffuse_lighting(brdf_data)

        color_layer = (1 - Fresnel_cal(brdf_data)) * diffuse + specular
        #color_layer = (1 - Fresnel_cal(brdf_data)) * diffuse
        #color_layer = specular

        back_lights = back_light(brdf_data).repeat(1, 1, 1, 1, 3)
        color_layer[back_lights] = 0.0
        colors += color_layer * light.intensity

    return colors

def specular_lighting(brdf_data):
    #specular_alpha = brdf_data['specular_alpha']
    #size = brdf_data['image_size']
    #NdotH = brdf_data['NdotH'].reshape(size+(1, ))
    #NdotL = brdf_data['NdotL'].reshape(size+(1, ))
    #NdotV = brdf_data['NdotV'].reshape(size+(1, ))

    #normals = brdf_data['normals']
    #H_vec = brdf_data['H_vec']
    #V_vec = brdf_data['V_vec']
    #L_vec = brdf_data['L_vec']

    #F = brdf_data['F']
    F = Fresnel_cal(brdf_data)
    D = D_cal(brdf_data)
    G2 = G2_cal(brdf_data)
    
    return F * G2 * D * NdotL(brdf_data)

# microfacet model, GGX_D
def D_cal(brdf_data):
    b = (alpha_squared_cal(brdf_data) - 1.0) * torch.pow(NdotH(brdf_data), 2) + 1.0
    return alpha_squared_cal(brdf_data)/(PI * b * b)

# microfacet model, Smith G2
def G2_cal(brdf_data):
     a = NdotV(brdf_data) * torch.sqrt(specular_alpha_cal(brdf_data) + NdotL(brdf_data) * (NdotL(brdf_data) - specular_alpha_cal(brdf_data) * NdotL(brdf_data)))
     b = NdotL(brdf_data) * torch.sqrt(specular_alpha_cal(brdf_data) + NdotV(brdf_data) * (NdotV(brdf_data) - specular_alpha_cal(brdf_data) * NdotV(brdf_data)))
     return 0.5/(a + b)

# Lambert diffuse
def diffuse_lighting(brdf_data):
    diffuse_reflectance = diffuse_reflectance_cal(brdf_data)
    return diffuse_reflectance * (ONE_OVER_PI * NdotL(brdf_data))

def brdf_data_cal(normals, verts, lights, cameras, texels):
    brdf_data = {}

    N, H, W, K, _ = texels.size()
    image_size = (N, H, W, K)

    base_color = texels[:, :, :, :, 0:3]
    metalness = texels[:, :, :, :, 3].unsqueeze(3)
    roughness = texels[:, :, :, :, 4].unsqueeze(3)

    #specular_alpha = torch.pow(roughness, 4)

    #V_vec = view_vec_cal(cameras, verts) #[N, H, W, K, 3]
    #L_vec = F.normalize(torch.tensor(lights.direction, dtype=torch.float32, device=device).squeeze(), dim=0) 
    #H_vec = half_vec_cal(V_vec, L_vec) #[N, H, W, K, 3]
    
    #NdotL = torch.sum(normals * L_vec, dim=1)
    #back_light = (NdotL(normals, V_vec) <= 0).reshape(N, H, W, K, 1)
    #NdotL = torch.clamp(NdotL, 0.00001, 1.0) 
    #HdotS = torch.clamp(torch.sum(H_vec * V_vec, dim=1), 0.0, 1.0)
    #NdotH = torch.clamp(torch.sum(normals * H_vec, dim=1), 0.0, 1.0)
    #NdotV = torch.clamp(torch.sum(normals * V_vec, dim=1), 0.00001, 1.0)

    #Fresnel_term = Fresnel_cal(base_color, metalness, HdotS(H_vec, V_vec))

    #diffuse_reflectance = diffuse_reflectance_cal(base_color, metalness)

    brdf_data['base_color'] = base_color
    brdf_data['metalness'] = metalness
    brdf_data['roughness'] = roughness
    brdf_data['normals'] = normals

    brdf_data['verts'] = verts
    brdf_data['lights'] = lights
    brdf_data['cameras'] = cameras

    #brdf_data['specular_alpha'] = specular_alpha
    #brdf_data['back_light'] = back_light

    #brdf_data['V_vec'] = V_vec
    #brdf_data['L_vec'] = L_vec
    #brdf_data['H_vec'] = H_vec

    brdf_data['image_size'] = image_size
    #brdf_data['NdotL'] = NdotL
    #brdf_data['NdotH'] = NdotH
    #brdf_data['NdotV'] = NdotV

    #brdf_data['F'] = Fresnel_term
    #brdf_data['diffuse_reflectance'] = diffuse_reflectance.reshape(image_size+(1, ))

    return brdf_data

def alpha_squared_cal(brdf_data):
    return torch.pow(brdf_data['roughness'], 2)

def specular_alpha_cal(brdf_data):
    return torch.pow(brdf_data['roughness'], 4)

def view_vec_cal(cameras, verts):
    camera_position = cameras.get_camera_center().squeeze()
    view = F.normalize((camera_position - verts), dim=4)
    return view

def light_vec_cal(lights):
    return F.normalize(-lights.direction, dim=0)

def half_vec_cal(cameras, verts, lights):
    return F.normalize(view_vec_cal(cameras, verts) + light_vec_cal(lights), dim=4)

def back_light(brdf_data):
    return (NdotL(brdf_data) <= 0)

def NdotL(brdf_data):
    return torch.clamp(torch.sum(brdf_data['normals'] * light_vec_cal(brdf_data['lights']), dim=4).unsqueeze(3), 0.00001, 1.0)

def HdotS(brdf_data):
    return torch.clamp(torch.sum(half_vec_cal(brdf_data['cameras'], brdf_data['verts'], brdf_data['lights']) * view_vec_cal(brdf_data['cameras'], brdf_data['verts']), dim=4).unsqueeze(3), 0.0, 1.0)

def NdotH(brdf_data):
    return torch.clamp(torch.sum(brdf_data['normals'] * half_vec_cal(brdf_data['cameras'], brdf_data['verts'], brdf_data['lights']), dim=4).unsqueeze(3), 0.0, 1.0)


def NdotV(brdf_data):
    return torch.clamp(torch.sum(brdf_data['normals'] * view_vec_cal(brdf_data['cameras'], brdf_data['verts']), dim=4).unsqueeze(3), 0.00001, 1.0)

def Fresnel_cal(brdf_data):
    mindielF0 = torch.tensor([MIN_DIELECTRICS_F0], dtype=torch.float32, device=device)
    F0 = torch.lerp(mindielF0, brdf_data['base_color'], brdf_data['metalness'])
    F90 = 1.0/MIN_DIELECTRICS_F0 * torch.sum(F0 * torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32, device=device), dim=-1).unsqueeze(3)
    F90 = torch.minimum(torch.ones_like(F90), F90)

    F = F0 + (F90 - F0) * torch.pow(1.0 - HdotS(brdf_data), 5.0)
    return F

def diffuse_reflectance_cal(brdf_data):
    return brdf_data['base_color'] *  (1.0 - brdf_data['metalness'])


        

