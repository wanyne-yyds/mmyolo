python ./projects/easydeploy/tools/export.py \
	/mmyolo/code/configs/custom_dataset/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob.py \
	--checkpoint /mmyolo/code/work_dir/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob/2023-02-20_06-46-08/best_coco/bbox_mAP_epoch_70.pth \
	--work-dir /mmyolo/code/work_dir/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob/2023-02-20_06-46-08/best_coco \
    --img-size 352 640 \
	--batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 1 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25 \
	--model-only

