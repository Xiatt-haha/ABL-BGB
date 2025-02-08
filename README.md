### Requirements
- Linux with Python == 3.7(Theoretically, this configuration is intended for Python >= 3.6, but we have only conducted experimental validation under Python = 3.7.)
- `pip install -r requirements.txt`


### Prepare Datasets
The dataset was obtained from [DocTamper](https://github.com/qcf-568/DocTamper).

### Note
- Regarding the error with the two SOI markers, our solution is to use the training script `train_wrapper.py`. After encountering an error, we continue training with the previously saved weight file. Once training is complete, we proceed to testing, which saves time.However, it's important to ensure that there are no logical errors in the training and testing processes before using this script to prevent infinite loops.
- `eval.sh` is the training script included in `train_wrapper.py`. It is important to note that you need to grant Linux permissions by using the command `chmod +x eval.sh` beforehand.
- Due to package import issues, it's necessary to create the required symbolic links before training.
```bash
    cd DocTamper
    ln -s models/dtd.py dtd.py
    ln -s models/swins.py swins.py
    ln -s models/fph.py fph.py
```


### command
The number of GPUs used, device settings, file save paths, and weight loading paths need to be configured in advance in `train.py`, `train_wrapper.py`, and `eval.py`.

- Training + Testing
```bash
#CUDA_VISIBLE_DEVICES=0 python train.py
python train_wrapper.py
```

- Testing Only
```bash
# CUDA_VISIBLE_DEVICES=0 python eval.py --lmdb_name DocTamperV1-FCD --pth outputs/base/resnet18/ckpt/checkpoint-best.pth --minq 75
# chmod +x eval.sh
./eval.sh
```
