import torch
import torch.nn.functional as F

class BMCLoss(torch.nn.modules.loss._Loss):
    """
    A class to compute the Batched-based Monte Carlo (BMC) loss, where the noise variance is a learnable parameter.
    """
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

def bmc_loss(pred, target, noise_var):
    """
    Compute the Batched-based Monte Carlo (BMC) loss between `pred` and the ground truth `target`.

    Args:
        pred (torch.Tensor): The input predictions.
        target (torch.Tensor): The target labels.
        noise_var (float or torch.Tensor): The variance of the noise.

    Returns:
        torch.Tensor: The computed BMC loss.
    """
    pred = pred.unsqueeze(dim=1)
    target = target.unsqueeze(dim=1)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]

    device = pred.device
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=device))
    loss = loss * (2 * noise_var).detach()

    return loss
