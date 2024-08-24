from src.okrolearn.okrolvision import *
from src.okrolearn.okrolearn import np, Tensor
# Load and process an image
img = Image.from_file('image.jpg')
resized_img = img.resize((224, 224))
gray_img = resized_img.to_grayscale()

# Apply computer vision operations
edges = ComputerVision.detect_edges(gray_img, 100, 200)
corners = ComputerVision.detect_corners(gray_img)

data = np.random.rand(3, 64, 64)  # RGB image
img_tensor = Image(data)

# Save the image
img_tensor.save("sample_image.png")

# Display the image
img_tensor.display()

# Create a batch of images
batch_data = np.random.rand(16, 3, 32, 32)  # 16 RGB images of size 32x32
batch_tensor = Tensor(batch_data)

# Create a grid of images
grid = Decoder.batch_to_grid(batch_tensor, nrow=4)

# Save the grid
Decoder.save_image(grid, "image_grid.png")

print("Images have been saved and displayed.")

# Load and process an image
img = Image.from_file('image.jpg')

# Apply pixel shuffle
upscaled_img = img.pixel_shuffle(upscale_factor=2)

# Save or display the result
Image.save(upscaled_img, "upscaled_image.png")
upscaled_img.display()
