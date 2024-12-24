import cv2
import lmdb
import torch
import jpegio
import numpy as np
import pickle
import six
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import torchvision
import tempfile

class DocTamperDataset(Dataset):
    def __init__(self, roots, max_nums=None, epoch=0, minq=75, qtb=90, max_readers=64, train=True, batch_size=12, jpeg_compress_times=[1,2,3], update_quality_setp=8192):
        self.envs = lmdb.open(roots,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = min(max_nums, self.nSamples)
        self.minq = minq
        self.train = train
        with open('qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        if not self.train:
            with open('pks/'+roots+'_%d.pk'%minq,'rb') as f:
                self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])
        if self.train:
            self.epoch = epoch
            self.batch_size = batch_size
            self.jpeg_compress_times = jpeg_compress_times
            self.update_quality_setp = update_quality_setp

    def set_epoch(self,epoch):
        self.epoch = epoch

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        if self.train:
            quality_lower = 99-((self.epoch)*self.nSamples+index)//self.update_quality_setp
            quality_lower = np.clip(quality_lower, 75, 100)
            jpeg_compress_time = np.random.choice(self.jpeg_compress_times)
            compress_qualities = np.random.randint(quality_lower,101,jpeg_compress_time)
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)!=0).astype(np.uint8)
            H,W = mask.shape
            if self.train:
                record = compress_qualities
            else:
                record = self.record[index]
            choicei = len(record)-1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei>1:
                q2 = int(record[-3])
                #use_qtb2 = self.pks[q2]
            if choicei>0:
                q1 = int(record[-2])
                #use_qtb1 = self.pks[q1]
            mask = self.totsr(image=mask.copy())['image']
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                im = im.convert("L")
                if choicei>1:
                    im.save(tmp,"JPEG",quality=q2)
                    im = Image.open(tmp)
                if choicei>0:
                    im.save(tmp,"JPEG",quality=q1)
                    im = Image.open(tmp)
                im.save(tmp,"JPEG",quality=q)
                jpg = jpegio.read(tmp.name)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'dct': np.clip(np.abs(dct),0,20),
                'qtb':np.clip(use_qtb,0,63),
                'q':q
            }