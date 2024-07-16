import torch
import torch.nn.functional as F

class WeightedMSELoss(torch.nn.modules.loss._Loss):
    """
    A class to compute the weighted Mean Squared Error (MSE) loss, where weights are calculated
    based on the density of the target labels using a provided DenseWeight model.
    """

    def __init__(self, dense_weight_model, size_average=None, reduce=None, reduction='mean'):
        """
        Initialize the WeightedMSELoss class.

        Args:
            dense_weight_model (DenseWeight): An instance of the DenseWeight model used to calculate weights.
            size_average (bool, optional): Deprecated (see torch.nn.modules.loss._Loss).
            reduce (bool, optional): Deprecated (see torch.nn.modules.loss._Loss).
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                       'mean': the sum of the output will be divided by the number of elements in the output.
                                       'sum': the output will be summed. Default is 'mean'.
        """
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
        self.dense_weight_model = dense_weight_model

    def forward(self, input, target):
        """
        Compute the weighted MSE loss.

        Args:
            input (torch.Tensor): The input predictions.
            target (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed weighted MSE loss.
        """
        weights = torch.tensor(self.dense_weight_model.dense_weight(target.cpu().numpy()), dtype=torch.float32, device=input.device)
        mse_loss = F.mse_loss(input, target, reduction='none')
        weighted_mse_loss = weights * mse_loss
        if self.reduction == 'mean':
            return torch.mean(weighted_mse_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_mse_loss)
        else:
            return weighted_mse_loss
