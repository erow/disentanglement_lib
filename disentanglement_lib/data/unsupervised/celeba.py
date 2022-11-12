import os
from torchvision import transforms as tfms, datasets
from disentanglement_lib.data.unsupervised.unsupervied_data import UnsupervisedData

ROOT=os.environ.get("DISENTANGLEMENT_LIB_DATA", ".")

class CelebA(datasets.CelebA, UnsupervisedData):
    def __init__(self, ) -> None:
        transform=tfms.Compose([
            tfms.Resize((128,128)),
            tfms.ToTensor(),
            # tfms.Normalize([0.5423088669776917, 0.4302210807800293, 0.3723295032978058], 
            #             [0.29796865582466125, 0.2654363512992859, 0.25773411989212036])
        ])
        root= os.environ.get("DISENTANGLEMENT_LIB_DATA", ".")
        super().__init__(root, download=True,transform=transform)