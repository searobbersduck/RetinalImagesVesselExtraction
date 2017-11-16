'''

This file is used to handle patches include lesions region

'''

import os
from glob import glob

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose, Scale, CenterCrop, Normalize, ToTensor

from PIL import Image, ImageOps
import numbers
import random

import numpy as np

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor > 10] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class RandomCropImage(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), x1, y1

# patch_input_transform = Compose([
#     Scale(256),
#     CenterCrop(256),
#     ToTensor(),
#     Normalize([.485, .456, .406], [.229, .224, .225]),
# ])
#
# patch_label_transform = Compose([
#     Scale(256),
#     CenterCrop(256),
#     ToLabel(),
#     Relabel(255,1)
# ])

patch_input_transform = Compose([
    Scale(256),
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

patch_label_transform = Compose([
    Scale(256),
    CenterCrop(256),
    ToLabel(),
    Relabel(255,1)
])


class PatchesTrainingDS(Dataset):
    '''
    root: the dir include followed subdirs, 'ahe', 'mask'
    input_trans:
    '''
    def __init__(self, root, input_trans = None, target_trans=None):
        self.root = root
        self.images_root = os.path.join(root, 'ahe')
        self.labels_root = os.path.join(root, 'mask')
        images_list = glob(os.path.join(self.images_root, '*.png'))
        labels_list = glob(os.path.join(self.labels_root, '*.png'))
        self.image_list = []
        for index in images_list:
            basename = os.path.basename(index).split('.')[0].replace('_ahe', '')
            if os.path.join(self.labels_root, basename+'.png') in labels_list:
                self.image_list.append(index)

        self.input_trans = input_trans
        self.target_trans = target_trans

    def __getitem__(self, item):
        image_file = self.image_list[item]
        label_file = os.path.join(self.labels_root, os.path.basename(image_file).replace('_ahe.png', '.png'))
        # print(image_file)
        with open(image_file, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(label_file, 'rb') as f:
            label = Image.open(f).convert('P')

        image = Scale(256)(image)
        label = Scale(256)(label)

        image, x1, y1 = RandomCropImage(224)(image)
        label = label.crop((x1, y1, x1 + 224, y1 + 224))

        if self.input_trans is not None:
            image = self.input_trans(image)
        if self.target_trans is not None:
            label = self.target_trans(label)
        return image, label

    def __len__(self):
        return len(self.image_list)



