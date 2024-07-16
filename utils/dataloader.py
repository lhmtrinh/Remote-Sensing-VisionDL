import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import h5py

def create_dataloader(dataset, batch_size, dense_weight_model=None, num_workers=0):
    """
    Creates a DataLoader for the given dataset with optional weighted sampling.

    Args:
        dataset (Dataset): The dataset to be loaded.
        batch_size (int): Number of samples per batch.
        dense_weight_model (Optional): Model to compute weights for downsampling.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A DataLoader object for the dataset.
    """
    if dense_weight_model != None:
        # Calculate weights for each data point in the combined dataset for downsampling        
        weights = dense_weight_model.dense_weight(dataset.all_labels)
        sampler = WeightedRandomSampler(weights, len(weights))
        # Create a DataLoader with the custom sampler
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

def create_dataset(file_paths, half=False, reshape_3d=False):
    """
    Creates a concatenated dataset from multiple HDF5 files.

    Args:
        file_paths (list of str): List of file paths to the HDF5 files.
        half (bool): Whether to use half-precision for the data.
        reshape_3d (bool): Whether to reshape the data into 3D.

    Returns:
        ConcatDataset: A concatenated dataset containing all the data and labels.
    """

    # Initialize lists to store datasets and labels
    datasets = []
    all_labels = []

    # Load each dataset and collect labels
    for file_path in file_paths:
        dataset = ConcatenatedDataset(file_path, half, reshape_3d)
        datasets.append(dataset)
        labels = dataset.labels.numpy() 
        all_labels.extend(labels)
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Store the combined labels as an attribute of the combined dataset
    combined_dataset.all_labels = torch.tensor(all_labels)
    
    return combined_dataset

class ConcatenatedDataset(Dataset):
    def __init__(self, data_file, half, reshape_3d):
        with h5py.File(data_file, 'r') as h5f:
            if half: 
                self.data = torch.from_numpy(h5f['data'][:]).float().half()
            else:
                self.data = torch.from_numpy(h5f['data'][:]).float()
            self.labels = torch.from_numpy(h5f['labels'][:]).float()
            self.locations = torch.from_numpy(h5f['locations'][:]).float()

        transformations = [ResizeTransform(), NormalizeChannels()]
        if reshape_3d:
            transformations += [ReshapeAndReorganize()]

        self.transform = transforms.Compose(transformations)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        location = self.locations[idx]
        label = self.labels[idx]
        data = self.transform(data)
        return data, location, label

class ResizeTransform:
    """Transform to crop and resize the image to 224x224."""
    def __call__(self, x):
        return x[:, :224, :224]

class NormalizeChannels:
    """
    Normalize each set of RGB channels independently.
    ImageNet mean and std are used for normalization of RGB channels,
    specific mean and std are used for non-RGB channels.
    """
    def __init__(self):
        self.mean_RGB = [0.485, 0.456, 0.406]
        self.std_RGB = [0.229, 0.224, 0.225]
        self.mean_nonRGB = [0.2599, 0.1838, 0.0986] # Based on training set
        self.std_nonRGB = [0.109, 0.0976, 0.07714] # Based on training set

    def __call__(self, x):
        x = x.clone()  # Clone to avoid modifying the original tensor

        # Normalize RGB channels
        mean_RGB = self.mean_RGB * 4  # Repeat mean for RGB channels
        std_RGB = self.std_RGB * 4    # Repeat std for RGB channels
        x[:12] = transforms.functional.normalize(x[:12], mean=mean_RGB, std=std_RGB)
        
        # Normalize Non-RGB channels
        mean_nonRGB = self.mean_nonRGB * 4  # Repeat mean for non-RGB channels
        std_nonRGB = self.std_nonRGB * 4    # Repeat std for non-RGB channels
        x[12:24] = transforms.functional.normalize(x[12:24], mean=mean_nonRGB, std=std_nonRGB)

        return x   

class ReshapeAndReorganize():
    """
    Reshape and reorganize the input tensor from [batch, 24, height, width] to
    [batch, channels, time, height, width] with specified channel groups.
    """
    def __init__(self):
        # Define the groups for channel and time dimensions
        self.channel_groups = [
            [0, 1, 2, 12, 13, 14],
            [3, 4, 5, 15, 16, 17],
            [6, 7, 8, 18, 19, 20],
            [9, 10, 11, 21, 22, 23]
        ]

    def __call__(self, x):
        _, height, width = x.shape
        channels_per_group = 6
        time_steps = 4

        # Create a new tensor to hold the reorganized data
        new_x = torch.zeros((channels_per_group, time_steps, height, width), dtype=x.dtype, device=x.device)

        # Reassign the channels according to the predefined groups
        for time_idx, group in enumerate(self.channel_groups):
            for channel_idx, ch in enumerate(group):
                new_x[channel_idx, time_idx, :, :] = x[ch, :, :]  # Correctly index into new_x and 

        return new_x

