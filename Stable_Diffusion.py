from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 

import numpy as np

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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        #self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps #default is 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

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


    def train_step(self, pred_tensor, text_embeddings=None, guidance_scale=100, min_t=0.02, max_t=0.98, detailed=False, latent_input=False):

        # interp to 512x512 to be fed into vae.
        if latent_input:
            latents = pred_tensor
        else:
            pred_rgb_512 = F.interpolate(pred_tensor, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        self.min_step = int(self.num_train_timesteps * min_t)
        self.max_step = int(self.num_train_timesteps * max_t)
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        #peseudo-loss
        p_loss = torch.sqrt(torch.mean(torch.pow((noise_pred - noise), 2))).item()

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        tensor_for_backward = SpecifyGradient.apply(latents, grad) 

        #output detailed information
        if detailed:
            self.scheduler.set_timesteps(self.num_train_timesteps)

            return tensor_for_backward, p_loss, latents, self.decode_latents(noise), \
            self.decode_latents(latents_noisy), self.decode_latents(noise_pred), \
            self.decode_latents(self.scheduler.step(noise_pred, t, latents)['prev_sample']), \
            self.decode_latents(noise_pred - noise), t
        return tensor_for_backward, p_loss 

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

                #test
                #path = "./Experiments/Differentiable_Image_Generation/structure_noise_comparison/ancestral_sampling/"
                #img = self.decode_latents(latents)
                #img_array = img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                #img_array = np.clip(img_array, 0, 1)
                #plt.imsave(path + f"_{t}.png", img_array)

        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

def main():
    import argparse
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--img', type=str, default=None)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    if opt.img is not None:
        img = Image.open(opt.img)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)[:, 0:3, :, :]
        img_tensor = img_tensor.to(device)
        #img_tensor = F.interpolate(img_tensor, (512, 512), mode='bilinear', align_corners=False)
        print(img_tensor.device)
        latents = sd.encode_imgs(imgs=img_tensor)
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, latents=latents)
    else:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()

#update img using SDS in latent space
def main_2():
    from utils import device
    import utils
    from Stable_Diffusion import StableDiffusion, SpecifyGradient
    import matplotlib.pyplot as plt

    guidance = StableDiffusion(device=device)
    seed = 0
    utils.seed_everything(seed)
    guidance_scale = 100 #100
    lr = 0.005
    epochs = 1000
    text_prompt = "an orange cat head"
    text_embeddings = guidance.get_text_embeds(text_prompt, '')

    latents = torch.randn((1, 4, 256, 256), dtype=torch.float32, device=device, requires_grad=True)
    latents = torch.nn.Parameter(latents)
    optimizer = torch.optim.Adam([latents], lr=lr)

    info_update_period = 50

    for epoch in range(epochs):
        optimizer.zero_grad()
        latents_input = F.interpolate(latents, (64, 64), mode='bilinear')
        t = torch.randint(guidance.min_step, guidance.max_step + 1, [1], dtype=torch.long, device=device)

        with torch.no_grad():
        # add noise
            noise = torch.randn_like(latents_input)
            latents_noisy = guidance.scheduler.add_noise(latents_input, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = guidance.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample        

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - guidance.alphas[t])
        grad = w * (noise_pred - noise)
        
        #peseudo-loss
        p_loss = torch.sqrt(torch.mean(torch.pow((noise_pred - noise), 2))).item()

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        tensor_for_backward = SpecifyGradient.apply(latents_input, grad) 

        tensor_for_backward.backward()
        optimizer.step()
        
        #info update
        if (epoch+1) % info_update_period == 0:
            print(f"[INFO] epoch: {epoch+1}, loss = {p_loss}")    

    #show result
    latents_input = F.interpolate(latents, (64, 64), mode='bilinear')
    img_tensor = guidance.decode_latents(latents_input)
    img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img_array = np.clip(img_array, 0, 1)
    plt.imshow(img_array)
    plt.show()

if __name__ == '__main__':
    main_2()