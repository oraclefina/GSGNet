import os
import torch
import cv2
import numpy as np
import glob
import argparse
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from tqdm import tqdm
from models.models import GSGNet_T

def read_image(p):
    img = Image.open(p).convert('RGB')
    size = img.size
    img = img_transform(img)
    return img[None,:,:,:],size

img_transform = transforms.Compose([
    transforms.Resize((352,352)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
])

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--weight_path", type=str)
parser.add_argument("--format", default='jpg', type=str,choices=['jpg','jpeg','png'])
args = parser.parse_args()

os.makedirs(os.path.join(args.path, "output"),exist_ok=True)
data_root = args.path
paths = glob.glob(os.path.join(data_root, "*.{}".format(args.pattern)))
if len(paths) == 0:
    raise Exception("NO IMAGES.")

model = GSGNet_T()
weight = torch.load(args.weight_path,map_location='cpu')
model.load_state_dict(weight)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

for p in tqdm(paths):
    img,img_size = read_image(p)
    filename = p.split('\\')[-1].split('.')[0]
    with torch.no_grad():
        img = img.to(device)
        pred,_,_ = model(img)

        pred_map = pred[0].detach().cpu().numpy()
        pred_map = cv2.resize(pred_map,img_size)
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min())
        pred_map = np.clip(np.round(pred_map*255+0.5),0,255)
    
        cv2.imwrite(os.path.join(args.path, "output",filename+".png"),pred_map)