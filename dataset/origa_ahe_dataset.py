import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from glob import glob
from PIL import Image
import random

from torchvision.transforms import Normalize, ToTensor
import torchvision.transforms as transforms

# for test
import cv2
from torch.utils.data import DataLoader


# global scope
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]

class Origa650TrainingDS(Dataset):
    def __init__(self, root, size):
        super(Origa650TrainingDS, self).__init__()
        self.root = root
        self.ahe_path = os.path.join(root, 'ahe')
        self.ahe_mask_path = os.path.join(root, 'ahe_mask')
        self.size = size
        _images_list = glob(os.path.join(self.ahe_path, '*.png'))
        _masks_list = glob(os.path.join(self.ahe_mask_path, '*.png'))
        _masks_list = [os.path.basename(i).split('.')[0] for i in _masks_list]
        self.images_list = []
        for img in _images_list:
            base_name = os.path.basename(img).split('.')[0]+'_mask'
            if base_name in _masks_list:
                self.images_list.append(img)
        self.input_trans = transforms.Compose([
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.target_trans = transforms.Compose([
            # ToLabel(),
            # Relabel(255,1)
            ToTensor(),
        ])

    def __getitem__(self, item):
        image_file = self.images_list[item]
        mask_file = os.path.join(self.ahe_mask_path, os.path.basename(image_file).split('.')[0]+'_mask.png')
        img = Image.open(image_file)
        mask = Image.open(mask_file)
        return self.input_trans(img), torch.LongTensor(np.array(mask, dtype=int))


    def __len__(self):
        return len(self.images_list)


def test_train_ds():
    root = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/origa650'
    color_trans = transforms.ToPILImage()
    ds = Origa650TrainingDS(root, 512)
    dataloader = DataLoader(ds, batch_size=2, shuffle=True, pin_memory=True)
    for index, (images, masks) in enumerate(dataloader):
        img = np.array(color_trans(images[0]))
        cv2.imshow('test', img)
        cv2.waitKey(2000)
        print(images)
        print(masks)


if __name__ == '__main__':
    test_train_ds()