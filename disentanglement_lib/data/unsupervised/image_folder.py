import os
from torchvision import transforms, datasets
import logging
from disentanglement_lib.data.unsupervised.unsupervied_data import UnsupervisedData, preprocess
from torchvision import transforms, datasets
import gin

gin.configurable('image_folder')
class ImageFolder(datasets.ImageFolder, UnsupervisedData):
    def __init__(self,
                 path,
                 size=(64,64),
                 logger=logging.getLogger(__name__)):
        UnsupervisedData.__init__(self)
        
        self.train_data = os.path.join(self.root,path)
        self.transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(self.train_data):
            print('path does not exist: ', self.train_data)
            raise NotADirectoryError()

        datasets.ImageFolder.__init__(self, self.train_data, transform=self.transforms)
