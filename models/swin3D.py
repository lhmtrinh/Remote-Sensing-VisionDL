import torchvision
import torch

def load_swin3d(version="base"):
    """
    Loads a pre-trained Swin3D model, modifies it to accept 6 input channels, and adjusts the classification head.

    Args:
        version (str): Specifies the version of the Swin3D model to load ("base" or "tiny").

    Returns:
        model (torch.nn.Module): Modified Swin3D model, or None if an invalid version is specified.
    """
    
    if version == "base":
        model = torchvision.models.video.swin3d_b(weights="Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1")
    elif version == "tiny":
        model = torchvision.models.video.swin3d_t(weights="Swin3D_T_Weights.KINETICS400_V1")
    else: 
        return None

    # Access the first Conv3d layer which is part of the PatchEmbed3d module
    first_conv_layer = model.patch_embed.proj

    # Create a new Conv3d layer with 6 input channels
    new_first_conv = torch.nn.Conv3d(
        in_channels=6,  # Increase from 3 to 6
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )

    head = model.head
    new_head = torch.nn.Linear(in_features=head.in_features, out_features=1, bias = True)

    # Initialize the weights for the new Conv3d layer
    # One common method is to average the original RGB weights and replicate them for the additional channels
    with torch.no_grad():
        original_weights = first_conv_layer.weight
        # Extend the weights by repeating the mean of the original three channels
        new_weights = torch.cat([original_weights, original_weights.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)], dim=1)
        scale_factor = (3 / 6) ** 0.5
        new_first_conv.weight.data = new_weights * scale_factor
        
    # Replace the original first convolutional layer with the new one
    model.patch_embed.proj = new_first_conv
    model.head = new_head

    return model