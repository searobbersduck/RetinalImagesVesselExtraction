import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='image preprocessing with scale and ahe operation')
    parser.add_argument('--root', required=True)
    # parser.add_argument('--scales', metavar='N', type=int, nargs='+')
    return parser.parse_args()

args = parse_args()


import numpy as np

from skimage.filters import threshold_otsu
from skimage import measure, exposure

import skimage

import scipy.misc
import scipy.io as sio

from PIL import Image
import os
import cv2

def tight_crop(img, size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1

    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs>0)
    ys = np.where(img_ys>0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]

    return img_crop


# adaptive historgram equlization
def channelwise_ahe(img):
    img_ahe = img.copy()
    for i in range(img.shape[2]):
        img_ahe[:,:,i] = exposure.equalize_adapthist(img[:,:,i], clip_limit=0.03)
    return img_ahe


def scale_image(pil_img, scale_size):
    w, h = pil_img.size
    tw, th = (min(w, h), min(w, h))
    image = pil_img.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
    w, h = image.size
    tw, th = (scale_size, scale_size)
    ratio = tw / w
    assert ratio == th / h
    if ratio < 1:
        image = image.resize((tw, th), Image.ANTIALIAS)
    elif ratio > 1:
        image = image.resize((tw, th), Image.CUBIC)
    return image

def scale_image_mask(pil_img, scale_size):
    w, h = pil_img.size
    tw, th = (min(w, h), min(w, h))
    image = pil_img.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
    w, h = image.size
    tw, th = (scale_size, scale_size)
    ratio = tw / w
    assert ratio == th / h
    if ratio < 1:
        image = image.resize((tw, th), Image.NEAREST)
    elif ratio > 1:
        image = image.resize((tw, th), Image.CUBIC)
    return image


raw_path = os.path.join(args.root, 'images')
raw_mask_path = os.path.join(args.root, 'manual marking')
ahe_path = os.path.join(args.root, 'ahe')
ahe_mask_path = os.path.join(args.root, 'ahe_mask')

assert os.path.isdir(raw_path)
assert os.path.isdir(raw_mask_path)
os.makedirs(ahe_path, exist_ok=True)
os.makedirs(ahe_mask_path, exist_ok=True)

from glob import glob

raw_image_list = glob(os.path.join(raw_path, '*.jpg'))

for image_file in raw_image_list:
    base_name = os.path.basename(image_file).split('.')[0]
    raw_mask_file = os.path.join(raw_mask_path, base_name+'.mat')
    assert os.path.exists(raw_mask_path)
    out_ahe_file = os.path.join(ahe_path, base_name+'_ahe.png')
    out_ahe_mask_file = os.path.join(ahe_mask_path, base_name+'_ahe_mask.png')
    # img = scipy.misc.imread(image_file)
    # img = img.astype(np.float32)
    # img /= 255
    # img_ahe = channelwise_ahe(img)
    # out_ahe_img = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
    out_ahe_img = Image.open(image_file)

    out_ahe_mask = sio.loadmat(raw_mask_file)
    out_ahe_mask = out_ahe_mask['mask']
    # out_ahe_mask = np.reshape(out_ahe_mask, (out_ahe_mask.shape[0], out_ahe_mask.shape[1], 1))
    out_ahe_mask = out_ahe_mask
    out_ahe_mask_pil = Image.fromarray(out_ahe_mask)

    print('{}\t'.format(base_name))
    # assert out_ahe_img.size == out_ahe_mask_pil.size
    if out_ahe_img.size != out_ahe_mask_pil.size:
        continue
    out_ahe_img = scale_image(out_ahe_img, 512)
    out_ahe_mask = scale_image_mask(out_ahe_mask_pil, 512)
    out_ahe_img.save(out_ahe_file)
    print('====>save: {}'.format(out_ahe_file))
    out_ahe_mask.save(out_ahe_mask_file)
    print('====>save: {}'.format(out_ahe_mask_file))



