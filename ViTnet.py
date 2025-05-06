
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

from net import ResBlock


class PatchEmbedding(nn.Module):
    def __init__(self,in_channels:int,patch_size:int,emb_size:int,img_size:int):
        super().__init__()



        self.projection=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=emb_size,kernel_size=patch_size,stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        # the position embedding here is randomly initialized and needs to be trained as other parameters.
        self.positions=nn.Parameter(torch.randn((img_size//patch_size)**2,emb_size))

    def forward(self,x:Tensor)->Tensor:
        x=self.projection(x)
        x+=self.positions
        return x


# the rope is useful. It's from this blog: https://spaces.ac.cn/archives/8265
class PatchEmbedding_rope(nn.Module):
    def __init__(self,in_channels:int,patch_size:int,emb_size:int,img_size:int):
        super().__init__()



        self.projection=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=emb_size,kernel_size=patch_size,stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        self.emb_size=emb_size
        self.seq_len=(img_size//patch_size)**2
        self.register_buffer(
            'freqs',
            self._get_freqs(self.seq_len,emb_size//2)
        )

    def _get_freqs(self,seq_len,dim):
        freqs=torch.exp(
            -torch.arange(0,dim,dtype=torch.float32)*(math.log(10000)/(dim-1))
        )
        positions=torch.arange(0,seq_len,dtype=torch.float32)
        phases=torch.einsum('i,j->ij',positions,freqs)
        return phases

    def _apply_rope(self,x):
        batch,seq_len,emb_size=x.shape
        x_reshape=rearrange(x,'b n (e d)->b n e d',d=2)
        x_complex=torch.complex(x_reshape[...,0],x_reshape[...,1])
        freqs=self.freqs[:seq_len,:]
        cos=torch.cos(freqs)
        sin=torch.sin(freqs)
        rot_complex=torch.complex(cos,sin).unsqueeze(0)
        x_rotated=x_complex*rot_complex

        x_out=torch.stack([x_rotated.real,x_rotated.imag],dim=-1)

        return rearrange(x_out,'b s e d->b s (e d)')


    def forward(self,x:Tensor)->Tensor:
        x=self.projection(x)
        x=self._apply_rope(x)
        return x




class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size:int,num_heads:int,dropout:float):
        super().__init__()
        self.emb_size=emb_size
        self.num_heads=num_heads
        self.ln=nn.LayerNorm(emb_size)
        self.qkv=nn.Linear(emb_size,emb_size*3)
        self.att_drop=nn.Dropout(dropout)
        self.mlp_drop=nn.Dropout(dropout)
        self.projection=nn.Linear(emb_size,emb_size)
        self.scaling=(self.emb_size//num_heads)**-0.5

    def forward(self,x:Tensor)->Tensor:
        res=x
        x=self.ln(x)
        qkv=rearrange(self.qkv(x),'b n (h d qkv)->(qkv) b h n d',qkv=3,h=self.num_heads)
        queries,keys,values=qkv[0],qkv[1],qkv[2]
        energy=torch.einsum('bhqd,bhkd->bhqk',queries,keys)
        att=F.softmax(energy*self.scaling,dim=-1)
        att=self.att_drop(att)
        out=torch.einsum('bhal,bhlv->bhav',att,values)
        out=rearrange(out,'b h n d-> b n (h d)')
        out=self.projection(out)
        out=self.mlp_drop(out)
        out+=res
        return out



class FeedForwardBlock(nn.Module):
    def __init__(self,emb_size:int,expansion:int=4,drop_p:float=0.):
        super().__init__()
        self.ln=nn.LayerNorm(emb_size)
        self.mlp=nn.Sequential(
            nn.Linear(emb_size,emb_size*expansion),
            nn.GELU(),
            nn.Linear(emb_size*expansion,emb_size),
            nn.GELU(),
            nn.Dropout(drop_p)
        )
    def forward(self,x):
        res=x
        x=self.ln(x)
        out=self.mlp(x)
        out+=res
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size:int,
                 drop_p:float,
                 expansion:int,
                 num_heads:int   ):
        super().__init__(
                MultiHeadAttention(emb_size,num_heads=num_heads,dropout=drop_p),
                FeedForwardBlock(emb_size,expansion,drop_p)
        )

class ConvDecoder(nn.Module):
    def __init__(self, features_indices, patch_grid_size, emb_size, drop_p):
        super().__init__()
        self.features_indices=features_indices
        self.num_features=len(features_indices)
        self.patch_grid_size=patch_grid_size
        self.rearrange=nn.ModuleList()
        for i in range(1,self.num_features):
            # the first features don't need to be rearranged
            self.rearrange.append(nn.Sequential(
                nn.Conv2d(emb_size,emb_size//(2**(i+1)),kernel_size=1),
                nn.ConvTranspose2d(emb_size//(2**(i+1)),emb_size//(2**(i+1)),kernel_size=2**i,stride=2**i)
            ))
        self.fusion=nn.ModuleList()
        for i in range(self.num_features):
            self.fusion.append(nn.Sequential(
                ResBlock(emb_size//(2**i),emb_size//(2**(i+2)),3,drop_p),
                nn.ConvTranspose2d(emb_size//(2**(i+2)),emb_size//(2**(i+2)),kernel_size=2,stride=2)
            ))
        self.final_layer=ResBlock(emb_size//(2**(self.num_features+1)),1,3,drop_p)



    def forward(self,features):
        for i,feature in enumerate(reversed(features)):
            feature=rearrange(feature,"b (h w) d->b d h w",h=self.patch_grid_size)
            if(i!=0):
                feature=self.rearrange[i-1](feature)
                feature=torch.cat([out,feature],dim=1)
            out=self.fusion[i](feature)
        out=self.final_layer(out)


        return out




class ViT(nn.Module):
    """
    A common Vit architechture with a dpt head.
    The customization of the huperparameters is badly supported.
    """
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 512,
                 img_size: int = 512,
                 depth: int = 12,
                 drop_p:float=0.1,
                 expansion:int=4,
                 features_indices:tuple=(3, 6, 9, 12),
                 num_heads=8
                 ):
        super().__init__()
        self.embedding=PatchEmbedding_rope(in_channels,patch_size,emb_size,img_size)
        self.transformerencoder=nn.ModuleList([TransformerEncoderBlock(emb_size,drop_p,expansion,num_heads) for _ in range(depth)])
        self.convdecoder=ConvDecoder(features_indices, img_size // patch_size, emb_size, drop_p)
        self.features_indices=features_indices
    def forward(self,x):
        x=self.embedding(x)
        features=[]
        tokens=x
        for i,model in enumerate(self.transformerencoder):
            tokens=model(tokens)
            if i+1 in self.features_indices:
                features.append(tokens)
        out=self.convdecoder(features)
        out=rearrange(out,'b c h w-> (b c) h w')
        return out
