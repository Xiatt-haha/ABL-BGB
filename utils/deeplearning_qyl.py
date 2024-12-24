import os
import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from glob import glob
from PIL import Image
from tqdm import tqdm
from .custom_lr import ShopeeScheduler
from .ranger import Ranger
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.utils import AverageMeter #, inial_logger
from .log import get_logger
from .metric import IOUMetric
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
Image.MAX_IMAGE_PIXELS = 1000000000000000

def dice_loss(
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    #mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Implenation of Dice loss.
    """
    intersection = torch.sum(pred * gt)
    union = torch.sum(pred) + torch.sum(gt) + eps
    loss = 1 - 2.0 * intersection / union
    return loss

def step_function(x, y, k=50):
    return torch.reciprocal(1 + torch.exp(-k * (x - y)))

def train_net_qyl(param, model, train_data, valid_data, device='cuda', use_db=True):
    # 初始化参数
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    iter_inter      = param['iter_inter']
    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']
    save_epoch=param['save_epoch']
    T0=param['T0']
    scaler = GradScaler() 

    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4 ,weight_decay=5e-4)
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    #optimizer=Ranger(model.parameters(),lr=1e-3)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-5, last_epoch=-1)
    #scheduler=ShopeeScheduler(optimizer,**scheduler_params)
    #criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    DiceLoss_fn=DiceLoss(mode='multiclass')
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    #logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) +'_'+model_name+ '.log'))
    logger = get_logger(os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) +'_'+model_name+ '.log'))
    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_f = 0
    best_epoch=0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))
    #
    for epoch in range(epoch_start, epochs):
        train_data.set_epoch(epoch)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=0)
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target, dct_coef, qs= batch_samples['image'], batch_samples['label'],batch_samples['dct'], batch_samples['qtb']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            with autocast(): #need pytorch>1.6
                if use_db:
                    target_tampered = (target == 1).float()
                    s = 11
                    target_tampool = F.max_pool2d(target_tampered, kernel_size=s, stride=1, padding=s//2)
                    target_edge = target_tampool - target_tampered
                    target_modi = target.clone()
                    target_modi [target_edge == 1] = 2 
                    #target [target_edge == 1] = 2  #
                    #for i in range(data.size(0)): 
                        #save_combined_image(data[i], target[i], target_modi[i], epoch, batch_idx)
                pred = model(data,dct_coef,qs)
                pred_seg = pred['seg_out']
                loss = 0
                if use_db:
                    loss_seg = LovaszLoss_fn(pred_seg, target_modi) + SoftCrossEntropy_fn(pred_seg, target_modi)
                else:
                    loss_seg = LovaszLoss_fn(pred_seg, target) + SoftCrossEntropy_fn(pred_seg, target)
                loss += loss_seg
                if use_db:
                    prob = F.softmax(pred_seg, dim=-1)
                    prob_b = step_function(prob[:,1], prob[:, 2])
                    loss_binary = dice_loss(prob_b, target)
                    loss += loss_binary
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / train_loader_size) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                if use_db:
                    logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss_seg:{:.6f} loss_edge:{:.6f} ETA:{}min'.format(
                        epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                        optimizer.param_groups[-1]['lr'],
                        loss_seg,loss_binary,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                else:
                    logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                        epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                        optimizer.param_groups[-1]['lr'],
                        train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        #scheduler.step()
        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        iou=IOUMetric(2)
        precisons = []
        recalls = []
        with torch.no_grad():
            for batch_idx_val, batch_samples_val in enumerate(valid_loader):
                data_val, target_val, dct_coef_val, qs_val = batch_samples_val['image'], batch_samples_val['label'],batch_samples_val['dct'], batch_samples_val['qtb']
                data_val, target_val, dct_coef_val, qs_val = Variable(data_val.to(device)), Variable(target_val.to(device)), Variable(dct_coef_val.to(device)), Variable(qs_val.unsqueeze(1).to(device))
                pred = model(data_val,dct_coef_val,qs_val)
                pred = pred['seg_out']
                if use_db:
                    pred[pred == 2] = 0
                    pred = pred[:,:2,:,:]
                predt = pred.argmax(1)
                pred = pred.cpu().data.numpy()
                targt = target_val.squeeze(1)
                matched = (predt*targt).sum((1,2))
                pred_sum = predt.sum((1,2))
                target_sum = targt.sum((1,2))
                precisons.append((matched/(pred_sum+1e-8)).mean().item())
                recalls.append((matched/target_sum).mean().item())
                pred = np.argmax(pred,axis=1)
                iou.add_batch(pred,target_val.cpu().data.numpy())
            acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            precisons = np.array(precisons).mean()
            recalls = np.array(recalls).mean()
            f_score = 2*precisons*recalls/(precisons+recalls+1e-8)
            logger.info('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu[1],precisons,recalls,f_score))
                

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch in save_epoch[T0]:
            torch.save(model.state_dict(),'{}/cosine_epoch{}.pth'.format(save_ckpt_dir,epoch))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        filename = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        '''
        if iu[1] > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = iu[1]
            best_mode = copy.deepcopy(model)'''
        if f_score > best_f:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_f = f_score
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        #scheduler.step()
            
    return best_mode, model
#
