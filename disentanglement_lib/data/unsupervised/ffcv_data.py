import os
from disentanglement_lib.data.unsupervised.unsupervied_data import UnsupervisedData, preprocess
from torchvision import transforms, datasets
import gin
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout,NormalizeImage
from ffcv.fields.decoders import *
import tqdm, torch
import numpy as np

IMAGENET_MEAN = np.array([0]*3) * 255
IMAGENET_STD = np.array([1]*3) * 255

class FFCV_DATA():
    def __init__(self,
                 path):
        self.root= os.environ.get("DISENTANGLEMENT_LIB_DATA", ".")
        self.train_data = os.path.join(self.root,path)
        
        
        # Random resized crop
        decoder = CenterCropRGBImageDecoder((64,64),1)

        # Data decoding and augmentation
        image_pipeline = [decoder, ToTensor(),
                          ToDevice(torch.device('cuda'), non_blocking=True),
                          ToTorchImage(),
                          NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
                          ]
        label_pipeline = [IntDecoder(), ToTensor()]

        # Pipeline for each data field
        self.pipelines = {
            'image': image_pipeline,
            'label': label_pipeline
        }
        
        
    def dataloader(self,distributed=False):
        if distributed:
            return Loader(self.train_data, batch_size=256, num_workers=4,
                order=OrderOption.RANDOM, pipelines=self.pipelines,
                distributed=True,os_cache=True)
        else:
            return Loader(self.train_data, batch_size=256, num_workers=4,
                order=OrderOption.QUASI_RANDOM, pipelines=self.pipelines)
        
        
if __name__ == '__main__':
    import pytorch_lightning as pl
    import torch, torch.nn as nn
    class TEST(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(1,1)
        def training_step(self, batch, batch_idx):
            return torch.rand(1,requires_grad=True).sum()
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(),1e-4,betas=[0.9,0.999])
        
        def train_dataloader(self):
            data = FFCV_DATA('../imagenet100/train_64_0.50_90.ffcv')
            dl  =Loader(data.train_data, batch_size=256, num_workers=4,
                    # distributed=True,
                    os_cache=True, 
                    order=OrderOption.RANDOM, pipelines=data.pipelines)
            return dl
    model=TEST()

    trainer = pl.Trainer(None,enable_checkpointing=False,max_epochs=3,enable_model_summary=False,
                        accelerator='gpu', devices=1,)

    trainer.fit(model)