import h5py
import numpy as np
from PIL import Image
import cv2
import random

def load_hdf5(infile):
    with h5py.File(infile, 'r') as f:
        return f['image'][()]

def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


# load_hdf5('/home/weidong/code/dr/RetinalImagesVesselExtraction/DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5')

#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
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
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks


def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)

def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same
    # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train


    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks/255.

    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print("\ntrain PATCHES images/masks shape:")
    print(patches_imgs_train.shape)
    # print("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train))())
    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test

path_data = '../DRIVE_datasets_training_testing/'

# patches_imgs_train, patches_masks_train = get_data_training(
#     path_data + 'DRIVE_dataset_imgs_train.hdf5',
#     path_data + 'DRIVE_dataset_groundTruth_train.hdf5',
#     256,
#     256,
#     1000,
#     False
# )
#
#
# import torch
# from torchvision.transforms import ToPILImage



# img_train = patches_imgs_train[0,:,:,:]
# mask_train = patches_masks_train[0,:,:,:]
# # img_train = np.transpose(img_train, (1,2,0))
#
# tensor = torch.FloatTensor(img_train)
# trans = ToPILImage()
# pil_img = trans(tensor)
# pil_img.show()
# tensor = torch.FloatTensor(mask_train)
# trans = ToPILImage()
# pil_img = trans(tensor)
# pil_img.show()


from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch



class RetinalVesselTrainingDS(torch.utils.data.Dataset):
    def __init__(self, data_root, img_file, mask_file, patch_w, patch_h, patch_num_per_img, inside_FOV = False):
        self.img_file = data_root + img_file
        self.mask_file = data_root + mask_file
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.patch_num_per_img = patch_num_per_img
        self.patch_num = patch_num_per_img * 20
        self.inside_FOV = inside_FOV
        self.patches_imgs_train, self.patches_masks_train = get_data_training(
            self.img_file,
            self.mask_file,
            self.patch_h,
            self.patch_w,
            self.patch_num,
            self.inside_FOV
        )

    def __getitem__(self, item):
        img_train = self.patches_imgs_train[item]
        mask_train = self.patches_masks_train[item]
        img_train = torch.Tensor(img_train)
        mask_train = torch.Tensor(mask_train)
        return img_train, mask_train

    def __len__(self):
        assert self.patches_imgs_train.shape[0] == self.patches_masks_train.shape[0]
        return self.patches_imgs_train.shape[0]

def test_RetinalVesselTrainingDS():
    import cv2
    import torchvision
    data_root = './DRIVE_datasets_training_testing/'
    img_file = 'DRIVE_dataset_imgs_train.hdf5'
    mask_file = 'DRIVE_dataset_groundTruth_train.hdf5'
    patch_w = 256
    patch_h = 256
    patch_num_per_img = 2000
    data_set = RetinalVesselTrainingDS(data_root, img_file, mask_file,
                                          patch_w, patch_h, patch_num_per_img)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=10, shuffle=True, pin_memory=True)
    for i, (imgs, masks) in enumerate(data_loader):
        trans = torchvision.transforms.ToPILImage()
        np_img = np.array(trans(imgs[0]))
        np_mask = np.array(trans(masks[0]))
        im_show = np.empty((np_img.shape[0]+np_mask.shape[0], np_img.shape[1]), dtype=np.uint8)
        im_show[0:np_img.shape[0]] = np_img
        im_show[np_img.shape[0]:(np_img.shape[0]+np_mask.shape[0]), :] = np_mask
        cv2.imshow('test_RetinalVesselTrainingDS', im_show)
        cv2.waitKey(2000)

if __name__ == '__main__':
    test_RetinalVesselTrainingDS()

