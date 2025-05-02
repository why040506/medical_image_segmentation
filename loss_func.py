import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    for the simplicity,
    we don't want this experiment to involve too many packages.
    so we calculate the loss by ourselves
    """
    def __init__(self,epsilon=1e-6):
        super().__init__()
        self.epsilon=epsilon

    def forward(self,preds,targets):
        preds=F.sigmoid(preds)
        preds=rearrange(preds,"b h w -> b (h w)")
        targets=rearrange(targets,"b h w -> b (h w)")
        intersection=(preds*targets).sum(dim=1)
        union=preds.sum(dim=1)+targets.sum(dim=1)
        dice=(2.*intersection+self.epsilon)/(union+self.epsilon)
        return 1-dice.mean()

