from utils import net_config
net_config()
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# image
img_dir = './helmet.png'
img = np.asarray(Image.open(img_dir))
# edge map
low_threshold = 100
high_threshold = 200

image = cv2.Canny(img, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

prompt = ", best quality, extremely detailed"
prompt = ["golden helmet" + prompt ]
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

output = pipe(
    prompt,
    canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
    num_inference_steps=20,
    generator=generator,
)


print(output.size())
plt.imshow(image[0, :, :, :])
plt.show()
plt.imshow(image[1, :, :, :])
plt.show()
plt.imshow(image[2, :, :, :])
plt.show()
plt.imshow(image[3, :, :, :])
plt.show()