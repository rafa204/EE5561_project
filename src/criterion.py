import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        denominator = (pred ** 2).sum(dim=(2, 3)) + (target ** 2).sum(dim=(2, 3))
        dice = (2. * intersection) / (denominator + self.epsilon)
        return dice.mean()

def kl_loss(mu, logvar):
    var = torch.exp(logvar)
    kl = (mu**2 + var - logvar - 1).sum(dim=1).mean()
    return kl