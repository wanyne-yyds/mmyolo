python projects/easydeploy/tools/image-demo.py \
    /mmyolo/data/NewHOD/JPEGImages/NEW-HOD-Muilt-Class/train/HOD_frame20220815/1/0_one.jpg \
    /mmyolo/code/configs/custom_dataset/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob.py \
    /mmyolo/code/work_dir/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob/2023-02-20_06-46-08/best_coco/end2end.onnx \
    --out-dir /mmyolo/code/work_dir/pred \
    --device cuda:1