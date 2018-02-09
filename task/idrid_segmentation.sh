#!/usr/bin/env bash

# challenge 1
CUDA_VISIBLE_DEVICES=0 python wholeimage_segmentation.py --root '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/EX' --root_val '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/EX' --lr 0.001 --batch 1 --port 8097 --exp idrid_challenge1_ex

CUDA_VISIBLE_DEVICES=1 python wholeimage_segmentation.py --root '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/HE' --root_val '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/HE' --lr 0.001 --batch 1 --port 8098 --exp idrid_challenge1_he

CUDA_VISIBLE_DEVICES=0 python wholeimage_segmentation.py --root '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/MA' --root_val '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/MA' --lr 0.001 --batch 1 --port 8099 --exp idrid_challenge1_ma

CUDA_VISIBLE_DEVICES=1 python wholeimage_segmentation.py --root '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/SE' --root_val '/media/weidong/seagate_data/dataset/IDRID Challenge/IDRID 1/preprocessed/SE' --lr 0.001 --batch 1 --port 8096 --exp idrid_challenge1_se