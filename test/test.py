from PIL import Image
import argparse
from glob import glob
import os
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='blend image')
    parser.add_argument('--root', required=True)
    parser.add_argument('--ratio', default=0.2, type=float)
    return parser.parse_args()

args = parse_args()

ratio = args.ratio
root = args.root
bg_root = os.path.join(root, 'raw')
fg_root = os.path.join(root, 'mask')

images_list = glob(os.path.join(bg_root, '*.jpg'))
for bg_img_file in images_list:
    basename = os.path.basename(bg_img_file)
    fg_img_file = os.path.join(fg_root, basename.split('.')[0]+'_EX.png')
    if not os.path.exists(fg_img_file):
        continue
    bg_img = Image.open(bg_img_file)
    fg_img = Image.open(fg_img_file)
    fg_img_rgb = Image.new('RGB', fg_img.size)
    fg_img_rgb.paste(fg_img)
    blend_image = Image.blend(bg_img, fg_img_rgb, ratio)


    cv_bk = np.array(bg_img)
    cv_bk = cv2.cvtColor(cv_bk, cv2.COLOR_RGB2BGR)
    cv2.imshow(basename, cv_bk)
    cv2.waitKey(0)
    cv_image = np.array(blend_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(basename, cv_image)
    cv2.waitKey(0)
    cv_bk = np.array(bg_img)
    cv_bk = cv2.cvtColor(cv_bk, cv2.COLOR_RGB2BGR)
    cv2.imshow(basename, cv_bk)
    cv2.waitKey(0)



bg_img_file = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX/raw/C0001273.jpg'
fg_img_file = '/home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX/mask/C0001273_EX.png'

bg_img = Image.open(bg_img_file)
fg_img = Image.open(fg_img_file)
fg_img_rgb = Image.new('RGB', fg_img.size)
fg_img_rgb.paste(fg_img)

bg_img = bg_img.resize((1024,1024))
fg_img_rgb = fg_img_rgb.resize((1024,1024))

Image.blend(bg_img, fg_img_rgb, ratio).show()
