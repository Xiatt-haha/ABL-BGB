import os
import cv2
import lmdb
import torch
import jpegio
import tempfile
import numpy as np
import pickle
import albumentations as A
import six
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import torchvision
import pywt
from itertools import product
from skimage.feature import local_binary_pattern

train_transform = A.Compose([
    # reszie
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        #A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        #A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        #A.CoarseDropout(p=0.2),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5, 0.5,0.5)),
    ToTensorV2(),
])


class TrainDocTamperDataset(Dataset):
    def __init__(self, roots, epoch,batch_size=12,max_nums=None, max_readers=64, transform=train_transform,jpeg_compress_times=[1,2,3],update_quality_setp=8192):
        self.envs = lmdb.open(roots,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))#120000
        if max_nums is None:
            self.max_nums = self.nSamples
        self.max_nums = min(max_nums, self.nSamples)
        self.roots=roots
        self.epoch=epoch
        self.batch_size=batch_size
        self.jpeg_compress_times=jpeg_compress_times
        self.update_quality_setp=update_quality_setp
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.transform = transform
        # Load the qt_table.pk file and store it in self.pks
        with open('qt_table.pk', 'rb') as fpk:
            self.pks = pickle.load(fpk)
        self.totsr = ToTensorV2()

    def set_epoch(self,epoch):
        self.epoch = epoch

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        quality_lower=99-((self.epoch)*self.nSamples+index)//self.update_quality_setp
        quality_lower = np.clip(quality_lower, 75, 100)
        jpeg_compress_time = np.random.choice(self.jpeg_compress_times)
        compress_qualities=np.random.randint(quality_lower,101,jpeg_compress_time)
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)
            mask = self.totsr(image=mask.copy())['image']
            choicei = len(compress_qualities)-1
            q=int(compress_qualities[-1])
            use_qtb = self.pks[q]
            if choicei>1:
                q2 = int(compress_qualities[-3])
            if choicei>0:
                q1 = int(compress_qualities[-2])
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                im = im.convert("L")
                if choicei > 1:
                    im.save(tmp, "JPEG", quality=q2)
                    im = Image.open(tmp)
                if choicei > 0:
                    im.save(tmp, "JPEG", quality=q1)
                    im = Image.open(tmp)
                im.save(tmp, "JPEG", quality=q)
                jpg = jpegio.read(tmp.name)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
                        
        return {
            'image': self.toctsr(im),
            'label': mask.long(),
            'dct': np.clip(np.abs(dct), 0, 20),
            'qtb':np.clip(use_qtb,0,63),
            'q': q,
            'img_key':img_key,
        }