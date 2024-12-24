import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from utils import train_net_qyl
from dataset import DocTamperDataset, TrainDocTamperDataset, TestDataset
from dataset import train_transform, val_transform
from torch.cuda.amp import autocast
from models.dtd import *
from models.dtd import DTD, seg_dtd
#
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS = 1000000000000000

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda")
use_db = True
train_imgs_dir = os.path.join( "DocTamperV1-TrainingSet")
val_imgs_dir = os.path.join( "DocTamperV1-FCD")
train_data = TrainDocTamperDataset(train_imgs_dir, max_nums=100000000000000, epoch=0)
val_data = TestDataset(val_imgs_dir, max_nums=10000000000000, minq=75)

model_name = 'resnet18'
if use_db:
    n_class = 3
else:
    n_class = 2
model=seg_dtd(model_name,n_class).cuda()
model= torch.nn.DataParallel(model)

# 模型保存路径
sub_dir = 'BGDB'
save_ckpt_dir = os.path.join('./outputs/', sub_dir, model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', sub_dir, model_name)
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

# 参数设置
param = {}

param['epochs'] = 10          # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['batch_size'] = 12       # 批大小
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 1       # 保存间隔(epoch)
param['iter_inter'] = 50     # 显示迭代间隔(batch)
param['min_inter'] = 10

param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=3  #cosine warmup的参数
param['save_epoch']={2:[5,13,29,61],3:[7,8]}
# 加载权重路径（继续训练）
checkpoint_path = os.path.join('./outputs/', sub_dir, model_name, 'ckpt','checkpoint-latest.pth')
if os.path.exists(checkpoint_path):
    param['load_ckpt_dir'] = checkpoint_path
else:
    param['load_ckpt_dir'] = None

#
# 训练
best_model, model = train_net_qyl(param, model, train_data, val_data, use_db=use_db)

