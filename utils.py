import os
import torch
import numpy as np
import cv2
import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from timm.utils import AverageMeter
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from loss import *
import torchvision
import scipy.io as io
from PIL import Image


class SaliconT(Dataset):
    def __init__(self,root,df_x, df_y,transform=None,size=(352,352)) -> None:
        super().__init__()
        self.root = root
        self.img_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD),
            ])
        self.size = size 
        self.images = df_x.tolist()
        self.maps = df_y.tolist()

    def __getitem__(self, idx):
        img_path = os.path.join(self.root,'images','train',self.images[idx])
        map_path = os.path.join(self.root,'maps','train',self.maps[idx])

        image = Image.open(img_path).convert("RGB")
        # ground-truth
        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map2 = cv2.resize(map,(self.size[0]//8,self.size[1]//8)) 
        map3 = cv2.resize(map,(self.size[0]//4,self.size[1]//4)) 
        map = cv2.resize(map,(self.size[1],self.size[0])) 

        # transform
        image = self.img_transform(image)
        if np.max(map) > 1.0:
            map = map / 255.0
        assert np.min(map) >= 0.0 and np.max(map) <= 1.0

        return image,torch.FloatTensor(map),torch.FloatTensor(map2),torch.FloatTensor(map3)

    def __len__(self):
        return len(self.images)

class SaliconVal(Dataset):
    def __init__(self,root,df_x, df_y,transform=None,size=(352,352)) -> None:
        super().__init__()
        self.root = root
        self.img_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD),
            ])
        self.size = size 
        self.images = df_x.tolist()
        self.maps = df_y.tolist()


    def __getitem__(self, idx):
        img_path = os.path.join(self.root,'images','val',self.images[idx])
        map_path = os.path.join(self.root,'maps','val',self.maps[idx])

        image = Image.open(img_path).convert("RGB")
        # ground-truth
        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map = cv2.resize(map,(self.size[1],self.size[0])) 

        # transform
        image = self.img_transform(image)
        if np.max(map) > 1.0:
            map = map / 255.0
        assert np.min(map) >= 0.0 and np.max(map) <= 1.0, "Ground-truth not in [0,1].{} {}".format(np.min(map), np.max(map))

        return image,torch.FloatTensor(map)

    def __len__(self):
        return len(self.images)


class MIT1003(Dataset):
    def __init__(self,root,ims,gts,im_size=(352,352)):
        super().__init__()
        self.root = root
        self.images_path = ims
        self.gts_path = gts
        self.im_size = im_size
        self.img_transform = transforms.Compose([
            transforms.Resize((im_size[0],im_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                 IMAGENET_DEFAULT_STD)
        ])    

    def __getitem__(self,idx):
        img_path = os.path.join(self.root, self.images_path[idx])
        map_path = os.path.join(self.root, self.gts_path[idx])

        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)

        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map = cv2.resize(map,(self.im_size[1],self.im_size[0])) 
        if np.max(map) > 1.0:
            map = map / 255.0

        return image,torch.FloatTensor(map)

    def __len__(self):
        return self.images_path.shape[0]

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out
