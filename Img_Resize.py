from PIL import Image

# Load the input image
img = Image.open("./Assets/Images/test_image_16_16.png")

# Decrease the resolution by a factor of 2
new_size = (512, 512)
resized_img = img.resize(new_size)

# Save the resized image
resized_img.save("hulk_body_texture_256_256.jpg")