import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
from abc import ABC,abstractmethod




# ====================the customized dataset====================

class NiiSliceDataset(Dataset):
    '''
    the first version of the dataset.
    however, because of the correlation between data in one nii docu, and some bugs in loading nii docu using multiple num_workers,
    we don't use this form of dataset anymore.
    '''
    def __init__(self, data_dirs, mode="train", transform=None, cache_size=10):
        self.dirs = f"{data_dirs}/{mode}"
        self.transform = transform
        
        # 预加载所有数据索引
        self.data_idx = []
        self.cache = {}  # 简单缓存，不使用共享内存
        
        docu_len = len(os.listdir(self.dirs)) // 2
        for idx in range(docu_len):
            img_name = f"{self.dirs}/volume-{idx}.nii"
            img_file = nib.load(img_name)
            img_len = img_file.shape[2]
            for i in range(img_len):
                self.data_idx.append((idx, i))
    
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, id):
        idx, i = self.data_idx[id]
        
        # 简单缓存策略，不涉及多进程共享
        if idx not in self.cache:
            # 加载数据
            img_name = f"{self.dirs}/volume-{idx}.nii"
            segment_name = f"{self.dirs}/segmentation-{idx}.nii"
            
            # 整个文件加载到内存
            img_data = nib.load(img_name).get_fdata()
            seg_data = nib.load(segment_name).get_fdata()
            
            self.cache[idx] = {
                'img': img_data,
                'seg': seg_data
            }
        
        # 从缓存获取数据切片
        img = self.cache[idx]['img'][:, :, i].copy()
        segmentation = self.cache[idx]['seg'][:, :, i].copy()
        
        # 转换为张量
        img_tensor = torch.tensor(img).unsqueeze(0)  # 添加通道维度
        segmentation_tensor = torch.tensor(segmentation)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, segmentation_tensor


class ImgDataset(Dataset):
    """
    the default dataset uded in experiments.
    we load the data when it's needed to be used because of the memory limits.
    """
    def __init__(self, data_dirs, mode="train", transform=None):
        self.dirs = f"{data_dirs}/{mode}"
        self.transform = transform
        data_len = len(os.listdir(self.dirs)) // 2
        self.data_len=data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, id):
        # 读取对应id的灰度图和分割图
        img_path = os.path.join(self.dirs, f"volume-{id}.png")
        seg_path = os.path.join(self.dirs, f"segmentation-{id}.png")

        # 使用PIL读取灰度图像
        img = Image.open(img_path)
        segmentation = Image.open(seg_path)

        if self.transform:
            img,segmentation = self.transform(img,segmentation)

        return img, segmentation




# ====================the cuntomized transform class====================

class JointTransforms:
    """
    the container of multiple transforms.
    we don't use the torchvision.transforms.xxxx because it doesn't support the transforms of the img and the segmentation in the same way
    """
    def __init__(self,trans):
        self.transforms=[]
        self.transforms.extend(trans)
    def __call__(self, img,segmentation):
        for transform in self.transforms:
            img,segmentation=transform(img,segmentation)


        return img,segmentation.squeeze()




class Basetransform(ABC):
    """
    the basic class of the transforms.
    """
    def __init__(self,p=0.5):
        """
        :param p: the probability of applying the transforms.
        """
        self.p=p
    @abstractmethod
    def __call__(self,img,segmentation):
        pass


class JointPILToTensor:
    """
    load the sample pair and transform it to tensor and normalize it.
    """
    def __call__(self,img,segmentation):
        img=TF.pil_to_tensor(img).to(dtype=torch.float32)
        segmentation=TF.pil_to_tensor(segmentation).to(dtype=torch.float32)
        if segmentation.max()==255:
            segmentation=segmentation/255
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val + 1e-6)
        return img,segmentation



class Jointhflip(Basetransform):
    def __call__(self, img,segmentation):
        if random.random()>self.p:
            return img,segmentation
        img=TF.hflip(img)
        segmentation=TF.hflip(segmentation)
        return img,segmentation

class Jointvflip(Basetransform):
    def __call__(self, img,segmentatino):
        if random.random()>self.p:
            return img,segmentatino
        img=TF.vflip(img)
        segmentatino=TF.vflip(segmentatino)
        return img,segmentatino

class Jointrandomcrop(Basetransform):
    def __init__(self,p=0.5,crop_ratio=0.5):
        super().__init__(p=p)
        self.crop_ratio=crop_ratio
    def __call__(self,img,segmentation):
        if random.random()>self.p:
            return img,segmentation
        original_height,original_width=img.shape[-2],img.shape[-1]
        crop_ratio=self.crop_ratio+(1-self.crop_ratio)*random.random()
        crop_height=int(crop_ratio*original_height)
        crop_width=int(crop_ratio*original_width)
        i=random.randint(0,original_height-crop_height)
        j=random.randint(0,original_width-crop_width)
        img=TF.crop(img,i,j,crop_height,crop_width)
        segmentation=TF.crop(segmentation,i,j,crop_height,crop_width)
        img=TF.resize(img,[original_height,original_width])
        segmentation=TF.resize(segmentation,[original_height,original_width],interpolation=TF.InterpolationMode.NEAREST)
        return img,segmentation



