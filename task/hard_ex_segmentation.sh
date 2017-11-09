python common_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX --size 1024 --patch_size 256 --model duc_hdc --batch 4 --worker 4 --port 8098 --exp hard_ex
# python common_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/training --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test --patch_size 128 --model duc_hdc --batch 20 --worker 4 --port 8099

# python common_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/training --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test --patch_size 512 --model unet --batch 2 --worker 1 --port 8097 # --weight ./models/common_segmentation_unet_0002_best.pth

