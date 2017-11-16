#python common_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX --size 1024 --patch_size 256 --model duc_hdc --batch 4 --worker 1 --port 8098 --exp hard_ex --weight /home/weidong/code/dr/RetinalImagesVesselExtraction/task/output/common_segmentaion_train_20171109182417_duc_hdc_hard_ex/common_segmentation_duc_hdc_0020_best.pth
# python common_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/training --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test --patch_size 128 --model duc_hdc --batch 20 --worker 4 --port 8099

# python common_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/training --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/DRIVE/test --patch_size 512 --model unet --batch 2 --worker 1 --port 8097 # --weight ./models/common_segmentation_unet_0002_best.pth



# CUDA_VISIBLE_DEVICES=0,1 python patches_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/ex_patches --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/ex_patches --weight /home/weidong/code/dr/RetinalImagesVesselExtraction/task/output/common_lesions_segmentaion_train_20171114155158_unet_ex_patches/common_segmentation_unet_1528_best.pth --phase predict

#python wholeimage_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/e_optha_EX --batch 1 --weight /home/weidong/code/dr/RetinalImagesVesselExtraction/task/output/common_lesions_segmentaion_train_20171114173605_unet_whole_images_ex/common_segmentation_unet_1634_best.pth --lr 0.001 --exp whole_images_ex_predict --phase predict

python wholeimage_segmentation.py --root /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/ex_whole_image --root_val /home/weidong/code/dr/RetinalImagesVesselExtraction/data/e_ophtha/ex_whole_image --lr 0.001 --batch 1 --port 8098 --exp ex_with_healthy