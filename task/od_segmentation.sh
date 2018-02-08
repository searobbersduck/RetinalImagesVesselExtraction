#!/usr/bin/env bash
#python wholeimage_segmentation.py --root '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 3' --root_val '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 3' --lr 0.001 --batch 1 --port 8097 --exp idrid_challenge3_od

python wholeimage_segmentation.py --root '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 3' --root_val '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 3' --lr 0.001 --batch 1 --port 8097 --exp idrid_challenge3_od --weight /home/weidong/code/dr/RetinalImagesVesselExtraction/task/output/common_lesions_segmentaion_train_20180205151524_unet_idrid_challenge3_od/common_segmentation_unet_0039_best.pth --phase predict
