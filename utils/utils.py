import re
import os
import glob
import matplotlib.pyplot as plt

def sorted_file_paths(directory):
    """
    Get sorted file paths from the directory based on the batch number.

    Args:
        directory (str): The directory to search for files.

    Returns:
        List[str]: A sorted list of file paths.
    """
    return sorted(get_file_paths(directory), key=extract_batch_number)

def get_file_paths(directory):
    """
    Get all file paths in the specified directory with the specified file extension.

    Args:
    - directory (str): The directory to search for files.
    - file_extension (str): The file extension to filter by.

    Returns:
    - List[str]: A list of file paths.
    """
    file_paths = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)
    return file_paths

def extract_batch_number(file_path):
    """
    Extract the batch number from the filename using a regular expression.

    Args:
        file_path (str): The file path containing the batch number.

    Returns:
        int: The extracted batch number, or -1 if the pattern is not found.
    """
    match = re.search(r"data_batch_(\d+)", file_path)
    if match:
        return int(match.group(1))
    else:
        return -1  # If for some reason a file doesn't match the pattern

def normalize(x):
    x_normalized = (x - x.min()) / (x.max() - x.min())
    return x_normalized

def visualize_multispectral_images(image_tensor, alpha=0.25, heatmap=None):
    """
    Visualize multispectral images with optional heatmap overlay.

    Args:
        image_tensor (torch.Tensor): Tensor of shape [24, H, W] containing spectral bands.
        alpha (float, optional): Transparency level for the heatmap overlay. Defaults to 0.25.
        heatmap (torch.Tensor, optional): Heatmap tensor to overlay on the images. Defaults to None.
    """
    _, axs = plt.subplots(4, 4, figsize=(20, 20))
    
    # Define RGB bands in indices as per given channel order
    rgb_indices = [
        [0, 1, 2],  # B4_1, B3_1, B2_1
        [3, 4, 5],  # B4_2, B3_2, B2_2
        [6, 7, 8],  # B4_3, B3_3, B2_3
        [9, 10, 11] # B4_4, B3_4, B2_4
    ]
    
    # NIR/SWIR bands in indices
    b8_indices = [12, 15, 18, 21]
    b11_indices = [13, 16, 19, 22]
    b12_indices = [14, 17, 20, 23]

    # Normalize each channel for visualization
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

    if heatmap is not None:
        # Normalize and prepare heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_rgba = plt.cm.jet(heatmap.cpu().numpy())  # Apply colormap to create RGBA heatmap

    # Plotting RGB images
    for i, indices in enumerate(rgb_indices):
        rgb_img = image_tensor[indices].permute(1, 2, 0).cpu().numpy()  # Shape [H, W, 3]
        axs[0, i].imshow(rgb_img)
        if heatmap is not None:
            axs[0, i].imshow(heatmap_rgba, alpha=alpha)  # Overlay the heatmap with transparency
        axs[0, i].set_title(f'RGB_{i+1}')
        axs[0, i].axis('off')

    # Plotting B8, B11, B12 images as grayscale
    for row, indices in enumerate([b8_indices, b11_indices, b12_indices], start=1):
        for i, index in enumerate(indices):
            grayscale_img = image_tensor[index].cpu().numpy()
            axs[row, i].imshow(grayscale_img, cmap='gray')
            if heatmap is not None:
                axs[row, i].imshow(heatmap_rgba, alpha=alpha)  # Overlay the heatmap
            axs[row, i].set_title(f'B{8 + 3*(row-1)}_{i+1}')
            axs[row, i].axis('off')

    plt.tight_layout()
    plt.show()

def find_latest_checkpoint(directory_path):
    """
    Find the latest checkpoint file in the specified directory.

    Args:
        directory_path (str): Path to the directory containing checkpoint files.

    Returns:
        str: Path to the latest checkpoint file.
    """
    # Pattern to match 'checkpoint_epoch_{number}.pth' and extract the number
    pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')

    # List all files matching the pattern in the directory
    files = glob.glob(os.path.join(directory_path, 'checkpoint_epoch_*.pth'))

    # Initialize variables to keep track of the highest epoch number and the corresponding file
    max_epoch = -1
    latest_checkpoint_file = None

    # Loop through the files to find the one with the highest epoch number
    for file in files:
        match = pattern.search(os.path.basename(file))
        if match:
            epoch_number = int(match.group(1))
            if epoch_number > max_epoch:
                max_epoch = epoch_number
                latest_checkpoint_file = file

    return latest_checkpoint_file