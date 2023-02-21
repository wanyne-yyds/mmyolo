CUDA_VISIBLE_DEVICES=0 python video_demo.py \
/mmyolo/data/HODTestDataVideo/single_hand_weel \
/mmyolo/code/configs/custom_dataset/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco.py \
/mmyolo/code/work_dir/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco/2023-02-13/best_coco/bbox_mAP_epoch_70.pth \
--out /mmyolo/code/output/single_hand_weel