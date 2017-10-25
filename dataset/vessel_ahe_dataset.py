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

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==3)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    # assert (full_masks.shape[3]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[1] == full_masks.shape[1])
    patches = np.empty((N_patches, patch_h, patch_w, full_imgs.shape[3]))
    patches_masks = np.empty((N_patches, patch_h, patch_w))
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    patch_per_img = int(N_patches / full_imgs.shape[0])
    print("patches per full image: {}".format(patch_per_img))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            #check whether the patch is fully contained in the FOV
            # if inside==True:
            #     if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
            #         continue
            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2), :]
            patch_mask = full_masks[i, y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks


class RetinalVesselTrainingDS(Dataset):
    def __init__(self, root, size, patch_size, patches_per_img):
        super(RetinalVesselTrainingDS, self).__init__()
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
            mask = Image.open(ahe_mask_file)
            gts[i] = np.asarray(mask)
        # gts = np.reshape(gts, (gts.shape[0], gts.shape[1], gts.shape[2], 1))
        gts = gts/255.
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


def test_ds():
    from torch.utils.data import DataLoader
    import cv2
    color_trans = transforms.ToPILImage()
    root = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/training'
    size = 512
    patch_size = 256
    patches_per_img = 100
    ds = RetinalVesselTrainingDS(root, size, patch_size, patches_per_img)
    data_loader = DataLoader(ds,batch_size=1, shuffle=True, pin_memory=True)
    for index, (images, targets) in enumerate(data_loader):
        # img = np.array(color_trans(images[0]))
        # cv2.imshow('test', img)
        # cv2.waitKey(2000)
        mask = np.array(color_trans(targets[0]))
        cv2.imshow('test', mask)
        cv2.waitKey(2000)

# test_ds()
if __name__ == '__main__':
    test_ds()