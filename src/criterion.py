import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return soft_dice_loss(pred, target, self.epsilon)
    
def soft_dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    denominator = (pred ** 2).sum(dim=(2, 3)) + (target ** 2).sum(dim=(2, 3))
    dice = (2. * intersection) / (denominator + epsilon)
    return dice.mean()

def kl_loss(mu, logvar):
    var = torch.exp(logvar)
    kl = (mu**2 + var - logvar - 1).sum(dim=1).mean()
    return kl

MSE_loss = nn.MSELoss()

def combined_loss(seg_out, seg_target, vae_out, vae_target, mu, logvar, w1 = 0.1, w2 = 0.1):
    
    lossSD = soft_dice_loss(seg_out, seg_target.long()) 
    lossL2 = MSE_loss(vae_out, vae_target)
    lossKL = kl_loss(mu, logvar)
        
    return lossSD + w1 * lossL2 + w2 * lossKL