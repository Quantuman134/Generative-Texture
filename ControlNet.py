from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.utils.import_utils import is_xformers_available
import torch
import cv2
import numpy as np
import time

from torch.cuda.amp import custom_bwd, custom_fwd 

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, latents, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=latents.device, dtype=latents.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class ControlNet():
    def __init__(self, device='cpu', num_inference_steps=50, controlnet_library='lllyasviel/sd-controlnet-canny', sd_library="runwayml/stable-diffusion-v1-5") -> None:
        # num_inference_step: 50 or 1000?
        self.resolution = 512
        self.device = device
        
        self.edge_detector = cv2.Canny
        self.controlnet = ControlNetModel.from_pretrained(controlnet_library, torch_dtype=torch.float32).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(sd_library, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_library, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_library, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(sd_library, subfolder="unet").to(self.device)
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(sd_library, subfolder="scheduler")
        self.scheduler.set_timesteps(num_inference_steps) 

        self.num_inference_steps = num_inference_steps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def edge_detect(self, imgs: torch.Tensor, low_threshold=100, high_threshold=200):
        # input: image_tensor with size of [B, C, H, W], range of value [0, 1] 
        # output: image_tensor with size of [B, C, H, W], range of value [0, 1]
        imgs = np.uint8(((imgs + 1.0) / 2.0 * 255.0).type(torch.IntTensor).squeeze().permute(1, 2, 0).detach().cpu().numpy())
        edge_map = self.edge_detector(imgs, low_threshold, high_threshold)[:, :, None]
        edge_map = torch.tensor(edge_map, dtype=torch.float32, device=self.device)/255.0
        edge_map = torch.cat([edge_map] * 3, dim=2).unsqueeze(0).permute(0, 3, 1, 2)

        return edge_map

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    
    def train_step(self, latents, edge_maps, text_embeddings, min_t=0.02, max_t=0.98, 
                   guidance_scale=7.5, controlnet_conditioning_scale=1.0):
        # edge_maps: for classfier-free guidance, the batchsize is 2

        # config time sample
        # due to the list timesteps is reversed order e.g.[1000, 900, 800...]
        # the max_step_index and min_step_index are exchanged
        max_step_index = int(self.num_inference_steps * min_t)
        min_step_index = int(self.num_inference_steps * max_t)
        t_index = torch.randint(max_step_index, min_step_index, [1], dtype=torch.long, device=self.device)   
        time_step = self.scheduler.timesteps[t_index.item()]
        t = torch.tensor((time_step, ), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            # for CFG, you know...
            latent_model_input = torch.cat([latents_noisy] * 2)

            # controlnet part
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=edge_maps,
                        return_dict=False,
                    )

            down_block_res_samples = [
                        down_block_res_sample * controlnet_conditioning_scale
                        for down_block_res_sample in down_block_res_samples
                    ]
            mid_block_res_sample *= controlnet_conditioning_scale

            # unet part
            noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        #peseudo-loss
        p_loss = torch.sqrt(torch.mean(torch.pow((noise_pred - noise), 2))).item()

        grad = torch.nan_to_num(grad)
        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        tensor_for_backward = SpecifyGradient.apply(latents, grad) 

        return tensor_for_backward, p_loss 

    def produce_latents(self, edge_maps, prompts, negative_prompts='', num_inference_steps=100, guidance_scale=7.5, controlnet_conditioning_scale=1.0):
        prompts = prompts + ', best quality, extremely detailed'
        negative_prompts = negative_prompts + ', longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # prepare edge_map
        edge_maps = torch.cat([edge_maps] * 2)

        # get a sample latent from Gaussian noise
        latents = torch.randn((1, 4, 64, 64), dtype=torch.float32, device=self.device)

        # scheduler timesteps config
        self.scheduler.set_timesteps(num_inference_steps)

        # denoise loop
        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            with torch.no_grad():
            # controlnet
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeds,
                        controlnet_cond=edge_maps,
                        return_dict=False,
                    )

                down_block_res_samples = [
                        down_block_res_sample * controlnet_conditioning_scale
                        for down_block_res_sample in down_block_res_samples
                    ]
                mid_block_res_sample *= controlnet_conditioning_scale

                # noise predict from unet
                noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                
                # classifer-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents

def main_contronet_generate():
    from utils import device, net_config, read_img_tensor, show_img_tensor, seed_everything
    net_config()

    seed_everything(51644)

    # config
    img_dir = './helmet.png'
    text_prompt = 'a starwar helmet'
    num_inference_steps = 100
    guidance_scale=9.0
    controlnet_conditioning_scale=1.0

    cn = ControlNet(device=device)

    img = read_img_tensor(img_dir=img_dir, device=device)

    # edge map
    edge_map = cn.edge_detect(imgs=img)
    
    # latent denoised
    latents_output = cn.produce_latents(edge_map, text_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, controlnet_conditioning_scale=controlnet_conditioning_scale)

    # image decode
    img_out = cn.decode_latents(latents_output)
    show_img_tensor(img_out)

if __name__ == "__main__":
    main_contronet_generate()
