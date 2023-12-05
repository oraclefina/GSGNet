import os
import random
import time
import torch
import torch.optim as opt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import SaliconT,SaliconVal
from loss import *
from models.models import GSGNet_T
from config import cfg
from engine_train import train_one_epoch_salicon, validation_one_epoch_salicon


def set_seeds(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True



def main():
    os.makedirs(cfg.DATA.LOG_DIR,exist_ok=True)
    mark = time.time()
    # Load Dataset
    train_pd = pd.read_csv(cfg.DATA.SALICON_TRAIN)
    val_pd = pd.read_csv(cfg.DATA.SALICON_VAL) 
    trainset = SaliconT(cfg.DATA.SALICON_ROOT,train_pd['X'],train_pd['Y'],size=cfg.DATA.RESOLUTION)
    valset = SaliconVal(cfg.DATA.SALICON_ROOT,val_pd['X'],val_pd['Y'],size=cfg.DATA.RESOLUTION)

    train_loader = DataLoader(trainset,batch_size=cfg.TRAIN.BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(valset,batch_size=cfg.TRAIN.BATCH_SIZE,shuffle=False,num_workers=6)

    total_batch_size = cfg.TRAIN.BATCH_SIZE
    num_training_steps_per_epoch = len(trainset) // total_batch_size
    num_testing_steps_per_epoch = len(valset) // total_batch_size

    # Initialize model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GSGNet_T().to(device)
    optimizer = opt.Adam(model.parameters(),lr=cfg.SOLVER.LR)

    lr_schedule_values_by_epoch = []
    _LR = cfg.SOLVER.LR
    for i in range(cfg.SOLVER.MAX_EPOCH):
        lr_schedule_values_by_epoch.append(_LR)
        if i in {1,6,11}:
            _LR = _LR * 0.01
        _LR = max(_LR, cfg.SOLVER.MIN_LR)

    
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values_by_epoch[epoch]

        train_one_epoch_salicon(model, optimizer, train_loader, device=device,epoch=epoch,
                                num_training_steps_per_epoch=num_training_steps_per_epoch)
        
        val_kl, val_cc = validation_one_epoch_salicon(model, val_loader, device=device,epoch=epoch,
                                     num_testing_steps_per_epoch=num_testing_steps_per_epoch)

        if val_cc > 0.91:
            torch.save(model,os.path.join(cfg.DATA_LOG_DIR,"weight_{}_ep{}_{}_{}.pt".format(mark, epoch+1, val_cc,val_kl)))


if __name__ == "__main__":
    set_seeds()
    main()






