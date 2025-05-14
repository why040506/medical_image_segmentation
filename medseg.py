import os
import wandb
import torch
from data import ImgDataset
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  einops import rearrange
from triton.language import dtype

# ====================custom class and function====================

from net import UNET
from data import Jointrandomcrop,Jointvflip,JointTransforms,JointPILToTensor,Jointhflip
from utils import  get_model_size,calculate_iou,visual_mask_comparison,contains_experiments
from loss_func import DiceLoss,DiceLoss_class_reduction
from args import  Args
from ViTnet import ViT


# ====================hyperparameter setting====================

args=Args()

args.log_dir=os.path.join("experiments",args.time)
os.makedirs(args.log_dir,exist_ok=True)

# ====================log setting====================

if args.track==True:
    run=wandb.init(
        project=args.wandb_project_name,
        dir=args.log_dir,
        name=args.time,
        config=args.__dict__,
        save_code=True
    )
    wandb.run.log_code(".",exclude_fn=contains_experiments)


# ====================prepare train data and eval data====================

train_transforms=JointTransforms([
    JointPILToTensor(),
    Jointhflip(),
    Jointvflip(),
    Jointrandomcrop()
])
val_transforms=JointTransforms([
    JointPILToTensor(),
])

train_dataset=ImgDataset(data_dirs=args.datadir_path,mode="train",transform=train_transforms)
val_dataset=ImgDataset(data_dirs=args.datadir_path,mode="eval",transform=val_transforms)

train_dataloader=DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True,num_workers=args.num_workers)
val_dataloader=DataLoader(val_dataset,batch_size=args.batchsize,shuffle=False,num_workers=args.num_workers)


# ====================set the net and the loss and the optimizer====================
if args.net=='dpt':
    model=ViT()
elif args.net=='unet':
    model=UNET(args)
print(f"model size is: {get_model_size(model)}")
model=model.to(args.device)
celoss=nn.CrossEntropyLoss()
if args.class_reduction:
    diceloss=DiceLoss_class_reduction()
else:
    diceloss=DiceLoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=args.learning_rate,)

# ====================train start!====================

for epoch in range(1,args.epochs+1):
    print(f"epoch: {epoch} train start!")
    model.train()
    train_progress=tqdm(train_dataloader,desc=f"epoch:{epoch},train iteration")
    train_loss=0

    # ====================one train epoch====================

    for i, (img,segmentation) in enumerate(train_progress):
        img=img.to(device=args.device,dtype=torch.float32)
        segmentation=segmentation.to(device=args.device,dtype=torch.long)
        optimizer.zero_grad()
        pred=model(img)
        loss= celoss(pred, segmentation) + diceloss(pred, segmentation)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

        train_progress.set_postfix(loss=f"{train_loss/(i+1)}")

        # ====================log the metrics after 10 batchs====================


        with torch.no_grad():
            ious=calculate_iou(pred=pred,target=segmentation)
            for id,iou in enumerate(ious):
                if iou==-1:
                    print(id,"no mask!")
                else:
                    print(f"the{id} iou is: {iou}")
                    if args.track:
                        wandb.log({
                            "epoch": epoch-1+i/len(train_dataloader),
                            "train_loss": train_loss/(i+1),
                            f"{id}train_iou":iou,
                        })

            # ====================save the check point after 500 batchs====================

            if i%500==499:
                print("save check point")
                torch.save(model.state_dict(), f"{args.log_dir}/epoch_{epoch-1}_iter_{i}.pth")
                print("checkpoint save down")


    print(f"epoch: {epoch} train finished!")

    # ====================one val epoch====================

    print(f"epoch: {epoch} eval start!")
    model.eval()
    eval_progress=tqdm(val_dataloader,desc=f"epoch:{epoch},eval iteration")
    eval_loss=0
    allious=[[],[],[]]
    for i,(img,segmentation) in enumerate(eval_progress):
        img = img.to(device=args.device, dtype=torch.float32)
        segmentation = segmentation.to(device=args.device, dtype=torch.long)
        with torch.no_grad():
            pred=model(img)
            loss= celoss(pred, segmentation) + diceloss(pred, segmentation)
            eval_loss+=loss.item()
            eval_progress.set_postfix(loss=f"{eval_loss/(i+1)}")

            # ====================log the metrics after 10 batchs(not update to wandb)====================

            if i%10==9:
                ious=calculate_iou(pred=pred,target=segmentation)
                for id,iou in enumerate(ious):

                    if iou==-1:
                        print(id,"no iou!")
                    else:
                        allious[id].append(iou)
                        print(f"the{id} iou is: {iou}")
                visual_mask_comparison(img,pred,segmentation,epoch,i,logdir=args.log_dir)

    # ====================log the metrics after whole val data====================
    if args.track:
        for id in range(3):
            wandb.log({
                "epoch":epoch,
                f"eval_loss": eval_loss/len(val_dataloader),
                f"{id}eval_iou":np.mean(allious[id]).item()
            })
    print(f"epoch: {epoch} eval finished!")

