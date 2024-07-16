import torch
import torch.nn.functional as F    

class L3Loss(torch.nn.modules.loss._Loss):
    """
    A class to compute the L3 loss, which is the product of the MSE loss and the MAE loss.
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        """
        Initialize the L3Loss class.

        Args:
            size_average (bool, optional): Deprecated (see torch.nn.modules.loss._Loss).
            reduce (bool, optional): Deprecated (see torch.nn.modules.loss._Loss).
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                       'mean': the sum of the output will be divided by the number of elements in the output.
                                       'sum': the output will be summed. Default is 'mean'.
        """
        super(L3Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        """
        Compute the L3 loss.

        Args:
            input (torch.Tensor): The input predictions.
            target (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed L3 loss.
        """
        mse_loss = F.mse_loss(input, target, reduction='none')

        with torch.no_grad():
            mae_loss = F.l1_loss(input, target, reduction= 'none')

        l3_loss = mse_loss * mae_loss
        
        if self.reduction == 'mean':
            return torch.mean(l3_loss)
        elif self.reduction == 'sum':
            return torch.sum(l3_loss)
        else:
            return l3_loss