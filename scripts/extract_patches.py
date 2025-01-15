from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def visualize_patches(patches):
    """
    Visualize extracted patches in a grid.

    Parameters:
        patches (torch.Tensor): Tensor of patches with shape (N, C, patch_height, patch_width).
    """
    num_patches = patches.shape[0]
    patch_height, patch_width = patches.shape[2], patches.shape[3]

    # Determine grid size for visualization
    grid_size = int(num_patches**0.5) + 1

    # Create a figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for idx, ax in enumerate(axes.flat):
        if idx < num_patches:
            # Convert patch to PIL Image for visualization
            patch = patches[idx]
            img = to_pil_image(patch)
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def extract_patches(image_tensor, patch_size, stride):
    """
    Extract patches from a bird's eye view image tensor.

    Parameters:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (tuple): Patch size as (patch_height, patch_width).
        stride (tuple): Stride size as (stride_height, stride_width).

    Returns:
        torch.Tensor: Extracted patches of shape (N, C, patch_height, patch_width),
                      where N is the number of patches.
    """
    # Ensure input tensor is 3D (C, H, W)
    if len(image_tensor.shape) != 3:
        raise ValueError("Input tensor must have shape (C, H, W)")
    
    C, H, W = image_tensor.shape
    ph, pw = patch_size
    sh, sw = stride

    # Check if patch size and stride are valid
    if ph > H or pw > W:
        raise ValueError("Patch size must be smaller than the image dimensions")
    if sh <= 0 or sw <= 0:
        raise ValueError("Stride values must be positive")

    # Use unfold to create sliding windows
    patches = image_tensor.unfold(1, ph, sh).unfold(2, pw, sw)

    # Reshape to (N, C, patch_height, patch_width)
    N_patches_h = patches.size(1)
    N_patches_w = patches.size(2)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, ph, pw)
    
    return patches

def load_image_as_tensor(image_path):
    """
    Load a .png file and convert it to a PyTorch tensor with shape (C, H, W).

    Parameters:
        image_path (str): Path to the .png image.

    Returns:
        torch.Tensor: Image tensor with shape (C, H, W).
    """
    # Open the image using Pillow
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)

    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
    ])

    # Apply transformations
    image_tensor = transform(image)

    return image_tensor

def main(image_path, patch_size, stride):
    """
    Load an image, extract patches, and return the patches.

    Parameters:
        image_path (str): Path to the .png image.
        patch_size (tuple): Patch size as (height, width).
        stride (tuple): Stride as (height, width).

    Returns:
        torch.Tensor: Extracted patches of shape (N, C, patch_height, patch_width).
    """
    # Step 1: Load the image as a tensor
    image_tensor = load_image_as_tensor(image_path)

    # Step 2: Extract patches
    patches = extract_patches(image_tensor, patch_size, stride)

    return patches

# Example usage
image_path = "scripts/clusters/Warped perspective_screenshot_14.01.2025.png"
patch_size = (64, 64)  # Define patch size
stride = (64, 64)  # Define stride

patches = main(image_path, patch_size, stride)
print("Number of patches:", patches.shape[0])
print("Shape of each patch:", patches.shape[1:])

visualize_patches(patches)
