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
        preds=F.softmax(preds,dim=1)
        preds=rearrange(preds,"b c h w -> b c (h w) ")
        targets=rearrange(targets,"b h w -> b (h w)")
        targets_one_hot=torch.zeros(targets.shape[0],3,targets.shape[1],device=preds.device)
        targets_one_hot.scatter_(1,targets.unsqueeze(1),1)
        class_exists=(targets_one_hot.sum(dim=2)>0).float()

        intersection=(preds*targets_one_hot).sum(dim=2)
        union=preds.sum(dim=2)+targets_one_hot.sum(dim=2)
        dice=(2.*intersection+self.epsilon)/(union+self.epsilon)
        dice_loss=1-dice
        foreground_dice_loss=dice_loss[:,1:]
        foreground_exists=class_exists[:,1:]
        num_foreground_classes=foreground_exists.sum(dim=1)
        batch_loss=[]
        for i in range(foreground_dice_loss.shape[0]):
            if num_foreground_classes[i]>0:
                sample_loss=(foreground_dice_loss[i]*foreground_exists[i]).sum()/num_foreground_classes[i]
                batch_loss.append(sample_loss)

        if len(batch_loss)==0: return torch.tensor(0.,device=preds.device)
        return torch.stack(batch_loss).mean()


class DiceLoss_class_reduction(nn.Module):
    """
    for the simplicity,
    we don't want this experiment to involve too many packages.
    so we calculate the loss by ourselves
    """
    def __init__(self,epsilon=1e-6):
        super().__init__()
        self.epsilon=epsilon

    def forward(self,preds,targets):
        preds=F.softmax(preds,dim=1)
        preds=rearrange(preds,"b c h w -> b c (h w) ")
        targets=rearrange(targets,"b h w -> b (h w)")
        targets_one_hot=torch.zeros(targets.shape[0],3,targets.shape[1],device=preds.device)
        targets_one_hot.scatter_(1,targets.unsqueeze(1),1)
        class_exists=(targets_one_hot.sum(dim=2)>0).float()

        intersection=(preds*targets_one_hot).sum(dim=2)
        union=preds.sum(dim=2)+targets_one_hot.sum(dim=2)
        dice=(2.*intersection+self.epsilon)/(union+self.epsilon)
        dice_loss=1-dice
        foreground_dice_loss=dice_loss[:,1:]
        foreground_exists=class_exists[:,1:]

        foreground_sum=(foreground_dice_loss*foreground_exists).sum(dim=0)
        foreground_count=foreground_exists.sum(dim=0)
        valid_class_mask=(foreground_count>0)

        foreground_mean=torch.zeros_like(foreground_sum)
        foreground_mean[valid_class_mask]=foreground_sum[valid_class_mask]/foreground_count[valid_class_mask]
        final_loss=foreground_mean[valid_class_mask].mean() if valid_class_mask.any() else torch.tensor(0.,device=preds.device)
        return final_loss


