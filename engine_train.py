import torch
from tqdm import tqdm
from timm.utils import AverageMeter
from loss import *

def loss_fn(pred_maps,gts):
    loss = kldiv(pred_maps,gts) - cc(pred_maps,gts)
    return loss

def train_one_epoch_salicon(model, optimizer, data_loader,device,epoch,num_training_steps_per_epoch,alpha=0.001):
    cur_lr = 0
    for i, param_group in enumerate(optimizer.param_groups):
        cur_lr = param_group["lr"] 

    loss_train = AverageMeter()
    model.train()
    with tqdm(total=num_training_steps_per_epoch,desc="[Training]") as tbar:
        for batch_data in iter(data_loader):
            images, gts, g1, g2 = batch_data
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            if epoch == 0:
                g1 = g1.to(device, non_blocking=True)
                g2 = g2.to(device, non_blocking=True)


            optimizer.zero_grad()
            pred_maps,m1,m2 = model(images)
             
            loss_KL = kldiv(pred_maps,gts)
            loss_CC = cc(pred_maps,gts)
                
            if epoch == 0:
                loss =  1.0 * (loss_KL - loss_CC) + alpha * F.binary_cross_entropy_with_logits(m1,g1) + alpha*F.binary_cross_entropy_with_logits(m2,g2)
            else:
                loss =  1.0 * (loss_KL - loss_CC)
            
            loss_train.update(loss.item())
            loss.backward()
            optimizer.step()

            tbar.update()
            tbar.set_postfix(kl=loss_KL.item(),cc=loss_CC.item())
    print("Epoch: {:d} | loss:{:.4f} | lr:{}".format(epoch,loss_train.avg,cur_lr))

@torch.no_grad()
def validation_one_epoch_salicon(model, data_loader, device, epoch, num_testing_steps_per_epoch):
    loss_val = AverageMeter()
    loss_kl = AverageMeter()
    loss_cc = AverageMeter()
    loss_sim = AverageMeter()
    model.eval()

    with tqdm(total=num_testing_steps_per_epoch,desc="[validating]") as tbar:
        for batch_data in iter(data_loader):
            images,gts = batch_data
            images = images.to(device,non_blocking=True)
            gts = gts.to(device,non_blocking=True)

            pred_maps,_,_ = model(images)

            loss_val.update(loss_fn(pred_maps,gts).item())
            loss_kl.update(get_kl_metric(pred_maps,gts))
            loss_cc.update(get_cc_metric(pred_maps,gts))
            loss_sim.update(get_sim_metric(pred_maps,gts))

            tbar.update()
            tbar.set_postfix(kl=loss_kl.avg,cc=loss_cc.avg)

    print("Epoch: {:d} | loss:{:.4f} CC:{:.4f} KL:{:.4f} SIM:{:.4f}".format(epoch,loss_val.avg,loss_cc.avg,loss_kl.avg,loss_sim.avg))

    return loss_kl.avg, loss_cc.avg
