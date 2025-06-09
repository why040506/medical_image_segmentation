import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import numpy as np
from  args import  Args

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

class ChangeInformationExtractionModule(nn.Module):
    def __init__(self):
        super(ChangeInformationExtractionModule, self).__init__()
        # self.in_d = in_d
        # self.out_d = out_d
        # self.ba = BoundryAttention((64+128+256+512), ratio=16)
        self.sobel_x, self.sobel_y = get_sobel((64+128+256+512), 1)


        self.conv_dr = nn.Sequential(
            nn.Conv2d((64+128+256+512),64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, e4, e3, e2, e1):
        # upsampling
        e4 = F.interpolate(e4, e1.size()[2:], mode='bilinear', align_corners=True)
        e3 = F.interpolate(e3, e1.size()[2:], mode='bilinear', align_corners=True)
        e2 = F.interpolate(e2, e1.size()[2:], mode='bilinear', align_corners=True)
        # fusion
        x = torch.cat([e4, e3, e2, e1], dim=1)

        s = run_sobel(self.sobel_x, self.sobel_y, x)


        # edge = self.eam(s4, s1)
        # edge_att = torch.sigmoid(edge)  # [16, 1, 64, 64]
        # x_ba = self.ba(x)
        x = x * s
        x = self.conv_dr(x)

        # feature = x[0:1, 0:64, 0:64, 0:64]
        # vis.visulize_features(feature)

        # pooling
        e1 = x
        e2 = self.conv_pool1(x)
        e3 = self.conv_pool2(x)
        e4 = self.conv_pool3(x)

        return e4, e3, e2, e1


class GuidedRefinementModule(nn.Module):
    def __init__(self):
        super(GuidedRefinementModule, self).__init__()
        # self.in_d = in_d
        # self.out_d = out_d
        self.conv_e1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_e4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, e4, e3, e2, e1, e4_p, e3_p, e2_p, e1_p):
        # feature refinement
        e4 = self.conv_e4(e4_p + e4)
        e3 = self.conv_e3(e3_p + e3)
        e2 = self.conv_e2(e2_p + e2)
        e1 = self.conv_e1(e1_p + e1)

        return e4, e3, e2, e1

class ResBlock(nn.Module):
    """
    The basic block used to build unet.
    Using batchnorm by default.
    """
    def __init__(self,in_channels,out_channels,num,p=0.1):
        super().__init__()
        self.dropout=nn.Dropout(p=p)
        self.m=nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.dropout
            )
        ])
        self.m.extend([
            nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.dropout
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

        self.CIEM1 = ChangeInformationExtractionModule()
        self.GRM1 = GuidedRefinementModule()

        self.CIEM2 = ChangeInformationExtractionModule()
        self.GRM2 = GuidedRefinementModule()





    def forward(self,x):

        res=[]
        for squeeze,down in zip(self.squeezes,self.downs):
            x=squeeze(x)
            res.append(x)
            x=down(x)

        # # # change information guided refinement 1
        e1,e2,e3,e4=res[0],res[1],res[2],res[3]
        e4_p, e3_p, e2_p, e1_p = self.CIEM1(e4, e3, e2, e1)
        e4, e3, e2, e1 = self.GRM1(e4, e3, e2, e1, e4_p, e3_p, e2_p, e1_p)
        #
        # # # change information guided refinement 2
        e4_p, e3_p, e2_p, e1_p = self.CIEM2(e4, e3, e2, e1)
        e4, e3, e2, e1 = self.GRM2(e4, e3, e2, e1, e4_p, e3_p, e2_p, e1_p)

        res[0], res[1], res[2], res[3]=e1,e2,e3,e4

        x=self.plane(x)



        for idx,(unsqueeze,up) in enumerate(zip(self.unsqueezes,self.ups)):
            x=up(x)
            x=torch.cat((x,res[-(idx+1)]),dim=1)
            x=unsqueeze(x)
        x=rearrange(x,"b c h w -> (b c) h w")
        return x