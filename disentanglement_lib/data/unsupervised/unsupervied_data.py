import subprocess
import os
import abc
import hashlib
import zipfile
import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
import numpy as np
import logging


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


class UnsupervisedData(abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """
    root = os.environ.get("DISENTANGLEMENT_LIB_DATA", ".")

    def __init__(self, transforms_list=[], logger=logging.getLogger(__name__)):
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(self.root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    @property
    def observation_shape(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        raise NotImplementedError()
