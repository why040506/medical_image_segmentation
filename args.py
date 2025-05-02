import os
from dataclasses import dataclass,field
import torch
import time


@dataclass
class Args:
    exp_name:str=os.path.basename(__file__)[:-len(".py")]
    """the name of the experiment"""
    track:bool=True
    """if toggled, this experiment will be tracked with w&b"""
    wandb_project_name:str="medicine_segment"
    """the wandb's project name"""
    learning_rate:float=2.5e-4
    """the learning rate of the optimizer"""
    epochs:int=4
    """the epochs of the experiment"""
    batchsize:int=4
    """the batchsize of one train iteration"""
    datadir_path:str="data_split"
    """the path of the data's dir."""
    """the train data should has the path f'{datadir_path}/train', the val data is similar"""
    channel_list:list=field(default_factory=lambda: [64, 128, 256])
    """the channel num of each layer"""
    """the depth of the unet is (len(channel_list)+1)"""
    time:str=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    """init in run"""
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    """the device of the model will run in"""
    threshold:float=0.5
    """the threshold of the mask"""
    num_workers:int=4
    """num_workers used in dataloader"""
    log_dir:str=""
    '''the log dir. will be set when running'''

