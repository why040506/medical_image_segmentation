import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from  args import  Args


class ResBlock(nn.Module):
    """
    The basic block used to build unet.
    Using batchnorm by default.
    """
    def __init__(self,in_channels,out_channels,num):
        super().__init__()
        self.m=nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ])
        self.m.extend([
            nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        for _ in range(num-1)])
        self.skip=nn.Identity() if in_channels==out_channels else nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        res=x
        for model in self.m:
            res=model(res)
        return res+self.skip(x)


class UNET(nn.Module):
    """
    UNET class implements a U-Net architecture for image segmentation tasks.

    U-Net is a convolutional network architecture designed primarily for biomedical image segmentation.
    This class defines a customizable U-Net with parameterized channel sizes and block structures,
    allowing modular construction and flexibility for different segmentation tasks. The architecture
    follows an encoder-decoder structure with skip connections between corresponding levels of the encoder
    and decoder. The encoder compresses the input into a latent space representation while the decoder
    gradually reconstructs the output, using skip connections to combine contextual and spatial information.

    :ivar squeezes: A list of `ResBlock` modules for the encoder path that progressively reduces
        spatial dimensions while increasing the feature depth.
    :type squeezes: torch.nn.ModuleList
    :ivar downs: A list of `MaxPool2d` layers that perform down-sampling for each level of the
        encoder path.
    :type downs: torch.nn.ModuleList
    :ivar plane: A central `ResBlock` module that operates on the compressed latent space.
    :type plane: ResBlock
    :ivar unsqueezes: A list of `ResBlock` modules for the decoder path that progressively
        reconstructs high-dimensional representations.
    :type unsqueezes: torch.nn.ModuleList
    :ivar ups: A list of transposed convolutional layers with activations for up-sampling
        during the decoder phase.
    :type ups: torch.nn.ModuleList
    """
    def __init__(self,args:Args):
        super().__init__()

        self.squeezes=nn.ModuleList([
                ResBlock(1,args.channel_list[0],num=2)
        ])
        self.squeezes.extend([
           ResBlock(channel,channel*2,2)
        for channel in args.channel_list])

        self.downs = nn.ModuleList([
            nn.MaxPool2d(stride=2,kernel_size=2)
        for _ in range(len(args.channel_list)+1)])

        self.plane=ResBlock(args.channel_list[-1]*2,args.channel_list[-1]*4,2)

        self.unsqueezes=nn.ModuleList([
            ResBlock(channel*4,channel*2,2)
        for channel in reversed(args.channel_list)])

        self.unsqueezes.append(nn.Sequential(
            ResBlock(args.channel_list[0]*2,args.channel_list[0],2),
            nn.Sequential(
                nn.Conv2d(args.channel_list[0], 1, kernel_size=3, padding=1),
            )
        ))


        self.ups=nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channel*4,channel*2,kernel_size=2,stride=2),
                nn.ReLU()
            )
        for channel in reversed(args.channel_list)])
        self.ups.append(nn.Sequential(
            nn.ConvTranspose2d(args.channel_list[0]*2,args.channel_list[0],kernel_size=2,stride=2),
            nn.ReLU()
        ))

    def forward(self,x):

        res=[]
        for squeeze,down in zip(self.squeezes,self.downs):
            x=squeeze(x)
            res.append(x)
            x=down(x)

        x=self.plane(x)
        for idx,(unsqueeze,up) in enumerate(zip(self.unsqueezes,self.ups)):
            x=up(x)
            x=torch.cat((x,res[-(idx+1)]),dim=1)
            x=unsqueeze(x)
        x=rearrange(x,"b c h w -> (b c) h w")
        return x