
import os
import numpy as np
import cv2
from einops import rearrange

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
def calculate_iou(pred, target, threshold=0.5):
    '''

    :param pred: the output of the model (after sigmoid)
    :param target: the true mask
    :param threshold: the threshold of the pred mask
    :return: iou
    '''
    pred_binary = (pred > threshold).float()
    pred_binary=rearrange(pred_binary,"b h w -> b (h w)")
    target=rearrange(target,"b h w -> b (h w)")
    intersection = (pred_binary * target).sum(dim=1)
    union = pred_binary.sum(dim=1) + target.sum(dim=1) - intersection
    num_non_zeros=(target.sum(dim=1)!=0.).sum().item()
    if num_non_zeros ==0:
        return -1
    return ((intersection) / (union+1e-6)).sum().item()/num_non_zeros


def visual_mask_comparison(imgs, preds, targets,epoch,i, threshold=0.5, logdir=None):
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
    preds_binary=(preds > threshold).float()
    batch_size=imgs.shape[0]

    for b in range(batch_size):
        img=imgs[b].detach().cpu().permute(1,2,0).numpy()
        if img.max()<=1.:
            img=(img*255).astype(np.uint8)
        if img.shape[2]==1:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        pred_mask=preds_binary[b].detach().cpu().numpy()
        target_mask=targets[b].detach().cpu().numpy()
        h,w=pred_mask.shape
        comp_mask=np.zeros((h,w,3),dtype=np.uint8)

        # TP
        comp_mask[(pred_mask==1)&(target_mask==1)]=[0,255,0]
        # FP
        comp_mask[(pred_mask==1)&(target_mask==0)]=[255,0,0]
        # FN
        comp_mask[(pred_mask==0)&(target_mask==1)]=[0,0,255]


        # Concat the original img to the comp_mask
        comp_img=np.hstack((img,comp_mask))
        if logdir is not None:
            cv2.imwrite(os.path.join(vis_dir,f"comp_batch_{i}_{b}.png"),cv2.cvtColor(comp_img,cv2.COLOR_RGB2BGR))