import torch
from utils.utils import normalize
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def LOTS(imageinit, iterations, model, get_feature_maps, device,tau=0.1, alpha=0.1):
    """
    Performs the LOTS on the initial image tensor.

    Args:
        imageinit (torch.Tensor): The initial image tensor.
        iterations (int): The number of iterations to perform.
        model (torch.nn.Module): The model to be used for feature extraction.
        get_feature_maps (function): A function to get feature maps from the model.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        tau (float, optional): The distance threshold. Default is 0.1.
        alpha (float, optional): The step size scaling factor. Default is 0.1.

    Returns:
        torch.Tensor: The adversarial image tensor after LOTS attack.
        torch.Tensor: The distance between the final feature map and the target feature map.
    """
    imageadv = imageinit.clone().requires_grad_(True)
    for _ in range(iterations):
        model.zero_grad()
        Fs = get_feature_maps(imageadv.to(device))
        Ft = torch.zeros(Fs.shape).to(device)
        distance = torch.norm(Fs - Ft)
        if distance > tau:
            loss = F.mse_loss(Fs, Ft)
            loss.backward()
            gradient = imageadv.grad.data
            gradient_step = alpha * gradient / gradient.abs().max()
            imageadv.data -= gradient_step
            imageadv.data.clamp_(0, 1)  # Assuming image pixel values are in the range [0, 1]
            imageadv.grad.data.zero_()
        else:
            break
    return imageadv.detach(), distance.detach()  # Detach the image from the current graph to prevent further gradient computation

def calculate_activation_map(imageinit, imageadv, filter_size, with_normalize = False):
    """
    Calculates the activation map based on the perturbation between the initial and adversarial images.

    Args:
        imageinit (torch.Tensor): The initial image tensor of size [C, H, W].
        imageadv (torch.Tensor): The adversarial image tensor of size [C, H, W].
        filter_size (int): The size of the Gaussian filter.
        with_normalize (bool, optional): Whether to normalize the activation map. Default is False.

    Returns:
        torch.Tensor: The calculated activation map.
    """
    perturbation = torch.abs(imageinit - imageadv).mean(dim = 0) # mean across channels
    if not with_normalize:
        return perturbation
    perturbation_norm = normalize(perturbation)
    perturbation_blurred = TF.gaussian_blur(perturbation_norm.unsqueeze(0).unsqueeze(0),
                                        kernel_size=[filter_size, filter_size],
                                        sigma=(1.5, 1.5)).squeeze()
    # Normalize again after blurring
    activation_map = normalize(perturbation_blurred)
    return activation_map