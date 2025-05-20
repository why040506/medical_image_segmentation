import copy
import os
import numpy as np
import cv2
import torch
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler


# 获取模型总内存大小（MB或GB）
def get_model_size(model):
    """
    计算模型参数占用的内存大小，并转换为易读的单位（MB或GB）

    Args:
        model: PyTorch模型

    Returns:
        str: 格式化后的模型大小，带单位（MB或GB）
    """
    # 计算总字节数
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # 转换为MB
    total_mb = total_bytes / (1024 * 1024)

    # 如果超过1024MB，转换为GB
    if total_mb >= 1024:
        total_gb = total_mb / 1024
        return f"{total_gb:.2f} GB"
    else:
        return f"{total_mb:.2f} MB"


# calculate the iou
def calculate_iou(pred, target):
    '''

    :param pred: the output of the model
    :param target: the true mask
    :return: iou
    '''

    # Get predicted class (argmax)
    pred=F.softmax(pred,dim=1)
    pred_class = pred.argmax(dim=1)
    # Reshape tensors
    pred_class = rearrange(pred_class, "b h w -> b (h w)")
    target = rearrange(target, "b h w -> b (h w)")
    # Number of classes (from model output channels)
    num_classes = pred.shape[1]
    ious=[]
    for cls in range(num_classes):
        pred_mask=(pred_class==cls).float()
        target_mask=(target==cls).float()
        intersection=(pred_mask*target_mask).sum(dim=1)
        union=pred_mask.sum(dim=1)+target_mask.sum(dim=1)-intersection
        valid_samples=(target_mask.sum(dim=1)>0).sum().item()
        if valid_samples==0:
            class_iou=-1
        else:
            class_iou=(intersection/(union+1e-6)).sum().item()/valid_samples
        ious.append(class_iou)

    return ious


def visual_mask_comparison(imgs, preds, targets,epoch,i, logdir=None):
    """
    compare the pred mask and the target mask, and concat the comparision with the img.
    :param imgs: The original img, the shape is [B, 1, H, W]
    :param preds: The prediction of the model. (with sigmoid)
    :param targets: The target mask
    :param epoch: The epoch
    :param threshold: The threshold used to create the prediction of the mask
    :param logdir: The path of the dir to save the result.
    :return: None.
    """
    if logdir is not None:
        vis_dir=os.path.join(logdir,"visual",f"{epoch}")
        os.makedirs(vis_dir,exist_ok=True)
    preds_probs=F.softmax(preds,dim=1)
    preds_class=preds_probs.argmax(dim=1)
    batch_size=imgs.shape[0]
    # 定义颜色映射（每个类别一种颜色）

    color_map = {
        0: [0, 0, 0],  # 背景 - 黑色
        1: [255, 0, 0],  # 类别1 - 红色
        2: [0, 255, 0],  # 类别2 - 绿色
    }

    for b in range(batch_size):
        img=imgs[b].detach().cpu().permute(1,2,0).numpy()
        if img.max()<=1.:
            img=(img*255).astype(np.uint8)
        if img.shape[2]==1:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        pred_mask=preds_class[b].detach().cpu().numpy()
        target_mask=targets[b].detach().cpu().numpy()
        h,w=pred_mask.shape
        pred_vis=np.zeros((h,w,3),dtype=np.uint8)
        target_vis=np.zeros((h,w,3),dtype=np.uint8)
        for cls in range(3):
            pred_vis[pred_mask==cls]=color_map[cls]
            target_vis[target_mask==cls]=color_map[cls]

        comp_mask=np.zeros((h,w,3),dtype=np.uint8)

        #comp_mask[(pred_mask==target_mask)]=[0,0,255]
        comp_mask[(pred_mask!=target_mask)]=[255,0,0]


        # Concat the original img to the comp_mask
        comp_img=np.hstack((img,pred_vis,target_vis,comp_mask))
        if logdir is not None:
            cv2.imwrite(os.path.join(vis_dir,f"comp_batch_{i}_{b}.png"),cv2.cvtColor(comp_img,cv2.COLOR_RGB2BGR))

def contains_experiments(path:str)->bool:
    norm_path=os.path.normpath(path)
    parts=norm_path.split(os.sep)
    return "experiments" in parts


def get_lr_scheduler(args,optimizer,train_dataloader):
    if args.lr_schedule == "cos_anneling_warmstart":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_dataloader),
            T_mult=2,
            eta_min=args.learning_rate_min
        )
    elif args.lr_schedule == "warmup_cos_anneling":
        warmup_iters = int(0.1 * (args.epochs * len(train_dataloader)))
        warmup_schuduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.learning_rate_min / args.learning_rate,
            end_factor=1.,
            total_iters=warmup_iters
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_dataloader) - warmup_iters,
            eta_min=args.learning_rate_min
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_schuduler, cosine_scheduler],
            milestones=[warmup_iters]
        )
    else:
        scheduler = None
    return scheduler

class ModelEMA:
    def __init__(self,model:nn.Module,decay=0.999):
        self.ema_model=copy.deepcopy(model).to(next(model.parameters()).device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay=decay
    def update(self,model):
        with torch.no_grad():
            msd=model.state_dict()
            for k,ema_v in self.ema_model.state_dict().items():
                model_v=msd[k].detach()
                if torch.is_floating_point(model_v):
                    ema_v.copy_(ema_v*self.decay+(1.-self.decay)*model_v)
                else:
                    ema_v.copy_(model_v)



