# README.md

## package
* pytorch 2.0
* pytorch3D 0.7.3
* cuda 11.8
* diffusers 0.14.0
* huggingface-hub 0.14.2
* matplotlib
* tensorboard
* tokenizers 0.13.3
* transformers 4.28.1

## Img_Asset.py (disable)
A **Dataset** class of pytorch to collect pixel coordinates and pixel colors of an image as the traning set. This traning set is used to train the **NeuralTextureField**

## Img_Resize.py
A individual python script to resize a image and store it.

## Neural_Texture_Field.py
### PositionEncoding
* a layer for position encoding

### NeuralTextureField
A mlp represents a image, and this mlp can be used to represents a texture of mesh. This mlp should be trained to simulate an input image.


## One_Layer_Image.py
Based on **nn.Moudule**, a linear layer net is used to represent an image, each neural represent one color component of one pixel of the input image. For example, three neural represent one RGB color, and an 512x512 image require a linear layer whose neurals include 512x512x3 data. This class is like a differentiable renderer of the input image. When render the image using this class, just input a tensor whose value is 1.

## Stable_Diffusion.py
Based on Stable-Diffusion, the class is built as a guidance for optimize neural represented field, such as **NeuralTextureField**, **OneLayerImage**. train_step() is the core function for calculate loss from predicted noise.

## utils.py
1. device: the device of pytorch, cuda or cpu
2. seed_everything(seed=0), give every random function used in the project a seed 