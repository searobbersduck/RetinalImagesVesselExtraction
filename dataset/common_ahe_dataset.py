# make sure the input images preprocessed with scaling and ahe

import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from glob import glob
from PIL import Image
import random

from torchvision.transforms import Normalize, ToTensor
import torchvision.transforms as transforms


from dataset_utils import extract_random, extract_ordered_with_mask, extract_ordered, \
    extract_ordered_overlap, recompone_overlap

# for test
import cv2


# global scope
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


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

class CommonTrainingDS(Dataset):
    def __init__(self, root, size, patch_size, patches_per_img):
        super(CommonTrainingDS, self).__init__()
        self.root = root
        self.ahe_path = os.path.join(root, 'ahe')
        self.ahe_mask_path = os.path.join(root, 'ahe_mask')
        self.size = size
        self.patch_size = patch_size
        ahe_list = glob(os.path.join(self.ahe_path, '*.png'))
        self.Nimgs = len(ahe_list)
        self.N_patches = patches_per_img * self.Nimgs
        imgs = np.empty((self.Nimgs, size, size, 3))
        gts=np.empty((self.Nimgs,size, size))
        for i, ahe_img_file in enumerate(ahe_list):
            base_name = os.path.basename(ahe_img_file).split('.')[0]
            ahe_mask_file = os.path.join(self.ahe_mask_path, base_name+'_mask.png')
            img = Image.open(ahe_img_file)
            imgs[i] = np.asarray(img)
            mask = Image.open(ahe_mask_file).convert('1', dither=None)
            # mask = Image.open(ahe_mask_file)
            gts[i] = np.asarray(mask)
        # gts = np.reshape(gts, (gts.shape[0], gts.shape[1], gts.shape[2], 1))
        # gts = gts/255.
        self.patches, self.patches_mask = extract_random(imgs, gts, patch_size, patch_size, self.N_patches, False)
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
        # return self.input_trans(self.patches[item]), self.target_trans(self.patches_mask[item])
        return self.input_trans(self.patches[item]), \
               torch.Tensor(np.reshape(self.patches_mask[item], (1, self.patch_size, self.patch_size)))


    def __len__(self):
        assert self.patches.shape[0] == self.patches_mask.shape[0]
        return self.patches_mask.shape[0]

class CommonValidationDS(Dataset):
    def __init__(self, root, size, patch_size):
        super(CommonValidationDS, self).__init__()
        self.root = root
        self.ahe_path = os.path.join(root, 'ahe')
        self.ahe_mask_path = os.path.join(root, 'ahe_mask')
        self.size = size
        self.patch_size = patch_size
        ahe_list = glob(os.path.join(self.ahe_path, '*.png'))
        self.Nimgs = len(ahe_list)
        imgs = np.empty((self.Nimgs, size, size, 3))
        gts=np.empty((self.Nimgs,size, size))
        for i, ahe_img_file in enumerate(ahe_list):
            base_name = os.path.basename(ahe_img_file).split('.')[0]
            ahe_mask_file = os.path.join(self.ahe_mask_path, base_name+'_mask.png')
            img = Image.open(ahe_img_file)
            imgs[i] = np.asarray(img)
            mask = Image.open(ahe_mask_file).convert('1', dither=None)
            gts[i] = np.asarray(mask)
        # gts = np.reshape(gts, (gts.shape[0], gts.shape[1], gts.shape[2], 1))
        # gts = gts/255.
        self.patches, self.patches_mask = extract_ordered_with_mask(
            imgs, gts, patch_size, patch_size)
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
        # return self.input_trans(self.patches[item]), self.target_trans(self.patches_mask[item])
        return self.input_trans(self.patches[item]), \
               torch.Tensor(np.reshape(self.patches_mask[item], (1, self.patch_size, self.patch_size)))


    def __len__(self):
        assert self.patches.shape[0] == self.patches_mask.shape[0]
        return self.patches_mask.shape[0]

class CommonPredictImage(Dataset):
    def __init__(self, image_file, input_transform, size, patch_size, stride=None):
        super(CommonPredictImage, self).__init__()
        self.image_file = image_file
        self.size = size
        self.patch_size = patch_size
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride
        self.input_transform = input_transform
        imgs = np.empty((1, size, size, 3))
        img = Image.open(image_file)
        imgs[0] = np.asarray(img)
        self.patches = extract_ordered_overlap(imgs, self.patch_size,
                                self.patch_size, self.stride, self.stride)

    def __getitem__(self, item):
        return self.input_transform(self.patches[item])

    def __len__(self):
        return self.patches.shape[0]


def test_ds():
    from torch.utils.data import DataLoader
    import cv2
    color_trans = transforms.ToPILImage()
    # root = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/training'
    # size = 512
    root = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX'
    size = 1024
    patch_size = 256
    patches_per_img = 100
    ds = CommonTrainingDS(root, size, patch_size, patches_per_img)
    data_loader = DataLoader(ds,batch_size=1, shuffle=True, pin_memory=True)
    for index, (images, targets) in enumerate(data_loader):
        # img = np.array(color_trans(images[0]))
        image = images[0]
        image[0] = image[0] * .229 + .485
        image[1] = image[1] * .224 + .456
        image[2] = image[2] * .225 + .406
        img = np.array(color_trans(image))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('test', img)
        # cv2.waitKey(2000)
        mask = np.array(color_trans(targets[0]))
        # cv2.imshow('test', mask)
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        stitch_img = np.empty((img.shape[0] * 2, img.shape[1], 3), dtype=np.uint8)
        stitch_img[:img.shape[0], :] = img
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        stitch_img[img.shape[0]:, :] = mask
        cv2.imshow('test', stitch_img)

        cv2.waitKey(2000)

def test_val_ds():
    from torch.utils.data import DataLoader
    import cv2
    color_trans = transforms.ToPILImage()
    # root = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test'
    # size = 512
    root = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX'
    size = 1024
    patch_size = 256
    patches_per_img = 100
    ds = CommonValidationDS(root, size, patch_size)
    data_loader = DataLoader(ds, batch_size=1, shuffle=True, pin_memory=True)
    for index, (images, targets) in enumerate(data_loader):
        image = images[0]
        image[0] = image[0] * .229 + .485
        image[1] = image[1] * .224 + .456
        image[2] = image[2] * .225 + .406
        img = np.array(color_trans(image))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('test', img)
        # cv2.waitKey(2000)
        mask = np.array(color_trans(targets[0]))
        # cv2.imshow('test', mask)
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        stitch_img = np.empty((img.shape[0] * 2, img.shape[1], 3), dtype=np.uint8)
        stitch_img[:img.shape[0], :] = img
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        stitch_img[img.shape[0]:, :] = mask
        cv2.imshow('test', stitch_img)

        cv2.waitKey(2000)

def test_predict_image():
    image_file = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test/ahe/02_test_ahe.png'
    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(MEAN, STD)
        ]
    )
    color_trans = transforms.ToPILImage()
    size = 512
    patch_size = 256
    stride = 64
    ds = CommonPredictImage(image_file, input_transform, size, patch_size, stride)
    data_loader = DataLoader(ds, batch_size=2, shuffle=False, pin_memory=False)
    # pred_image = torch.FloatTensor((len(data_loader)))
    pred_image_patches = torch.FloatTensor(len(data_loader.dataset), 3, patch_size, patch_size)
    cnt = 0
    for index, (images) in enumerate(data_loader):
        img = np.array(color_trans(images[0]))
        # cv2.imshow('test', img)
        # cv2.waitKey(2000)
        for i in range(images.shape[0]):
            pred_image_patches[cnt,:] = images[i]
            cnt = cnt+1
    pred_image_patches = pred_image_patches.numpy()
    pred_image = recompone_overlap(pred_image_patches, size, size, stride, stride)
    cv_img = np.transpose(pred_image[0], (1,2,0))
    cv2.imshow('test', cv_img)
    cv2.waitKey(2000)

# test_ds()
if __name__ == '__main__':
    # test_ds()
    # test_val_ds()
    test_predict_image()