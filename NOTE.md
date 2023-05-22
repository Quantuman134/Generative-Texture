# NOTE

## Reference 

### Texture Fields: Learning Texture Representations in Function Space (2019 ICCV)
1. input: 3D mesh and one 2D image

### TANGO: Text-driven Photorealistic and Robust 3D Stylization via Lighting Decomposition
1. input: mesh & text. output: albedo(brdf, normal...)
2. trained: directly generate albedo, then computed by CLIP (compare the images rendered from generated albedo and mesh and the text prompt)

### https://zhuanlan.zhihu.com/p/525106459
1. a introduction of **diffusion model** in Zhihu

### https://www.zhihu.com/question/536012286/answer/2533146567 by 路橙LuChengTHU
1. a introduction of **diffusion model** in Zhihu (math principle in this is easier to be understood)

### Denoising Diffusion Probabilistic Models(DDPM)
1. principle of diffusion model

### https://nn.labml.ai/diffusion/ddpm/unet.html
1. U-Net code with interpretation of DDPM

### High-Resolution Image Synthesis with Latebt Diffusion Model (stable diffusion)
1. model of stable diffusion, DDPM added with classfier-free guidance

### Text2Mesh: Text-Driven Neural Stylization for Meshes
1. input mesh, output color and displacement of each vertex
2. network and training: total 6 layers MLP, 25 minutes for training

### DreamFusion: Text-to-3D using 2D Diffusion (Highlight)
1. text to 3D nerf

### https://github.com/ashawkey/stable-dreamfusion
1. A pytorch implementation of text-to-3D dreamfusion (not original work)

### Zero-Shot Text-Guided Object Generation with Dream Fields
    


## Temporary Conclusions
1. random seed can generate high-quality results
2. training a complete diffusion model for 3D is unpractical ()
    * 3D space is much larger than image space
    * compelete training is unaffordable
3. The problems of TEXTure, which also probably are occurs in all texture generation based on depth-to-image method.
    * multiple generations cause the discontinuity
    * semantic inconsistence of each image from iteractions
    * According to the results, depth support informations only about what the prompt is, instead the concrete topology information. This cause the generated image usually does not align with mesh.
4. Why to use MLP to represent color
    * the result degenerate under back propogation without MLP

## Idea
1. there are two routine for albedo generation
    1. image space: generate image
    2. 3d space: generate a latent 3D space

### Image Space
1. code to albedo
    * problem: no existing network and training set to realize this scheme

2. based on TEXTure
    * 

### 3D Space
1. code to a 3D field such as nerf
    * unconditional: random Gauss noise to 3D field
    * unconditional (mesh is a base like that in nerf but not a condition): train a model based on a input mesh, a noise input to the model and output a 3D field
    * conditional: text and mesh to a 3D field
2. Test: train 1 input to output

## Problem
1. texture align with topology

## Goal in This Week
1. give a overall scheme of the texture generation, and an anlysis of feasibility.

## Question
1. The result(loss) of clamped mlp is worse than that of no clamped mlp, why?

## Programe Requirements
1. python=3.8
2. diffusers=0.14.0

## TODOLIST
1. pytorch program with a MLP
    1. ~~how to use cuda correctly in pytorch~~
    2. color output clamp
    3. ~~512x512 img and that shall require position encoding~~
    4. improve the image renderer efficiency
    5. infinite resolution
2. The environment conflictions are too many, maybe reconstruct an conda environment

## TODOLIST_NEXT_WEEK
1. try recreating an conda environment
2. create a custom method to load mlp texture in pytorch3d
3. 

