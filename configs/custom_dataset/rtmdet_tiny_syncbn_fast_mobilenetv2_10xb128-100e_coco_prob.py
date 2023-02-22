import time
_base_ = '../_base_/default_runtime.py'

# checkpoint = '/mmyolo/code/work_dir/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco/2023-02-13/best_coco/bbox_mAP_epoch_70.pth'  # noqa
load_from = '/mmyolo/code/work_dirs/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob/2023-02-20_06-46-08/best_coco/bbox_mAP_epoch_70.pth'
resume = False

data_root = '/mmyolo/data/MMYOLO_yoloFromat_2023-02-04/'
dataset_type = 'YOLOv5CocoDataset'

class_name = ('safety_belt','not_safety_belt',
              'person','wheel','dark_phone','bright_phone',
              'hand')  # 根据 class_with_id.txt 类别信息，设置 class_name

num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 160), (220, 20, 160), (220, 20, 160), (220, 20, 160), 
             (220, 20, 160), (220, 20, 160), (220, 20, 160)]  # 画图时候的颜色，随便设置即可
)

img_scale = (640, 352)  # width, height
deepen_factor = 0.33
widen_factor = 0.50
max_epochs = 100
stage2_num_epochs = 5
interval = 10

train_batch_size_per_gpu = 128
train_num_workers = 10
val_batch_size_per_gpu = 32
val_num_workers = 10
# persistent_workers must be False if num_workers is 0.
persistent_workers = True
strides = [8, 16, 32]
base_lr = 0.006

# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale,
    size_divisor=32,
    extra_pad_ratio=0.5)

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='mmdet.MobileNetV2',
        widen_factor=widen_factor,
        out_indices=(2,4,6),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        ), 
    neck=dict(
        type='YOLOXPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[32, 96, 320],
        out_channels=256,
        use_depthwise=True,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            widen_factor=widen_factor,
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='ReLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2)),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

with_mosiac_pipeline = [
    dict(
    type='Mosaic',
    img_scale=img_scale,
    use_cached=True,
    max_cached_images=20,
    random_pop=False,
    pad_val=114.0,
    pre_transform=pre_transform),
    dict(
    type='mmdet.RandomResize',
    # img_scale is (width, height)
    scale=(img_scale[0] * 2, img_scale[1] * 2),
    ratio_range=(0.5, 2.0),
    resize_type='mmdet.Resize',
    keep_ratio=True),
]

without_mosaic_pipeline = [
    dict(
    type='mmdet.RandomResize',
    # img_scale is (width, height)
    scale=(img_scale[0] * 2, img_scale[1] * 2),
    ratio_range=(0.5, 2.0),
    resize_type='mmdet.Resize',
    keep_ratio=True)
]

# Because the border parameter is inconsistent when
# using mosaic or not, `RandomChoice` is used here.
randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[with_mosiac_pipeline, without_mosaic_pipeline],
    prob=[0.3, 0.7])

train_pipeline = [
    *pre_transform, randchoice_mosaic_pipeline,
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    # dict(type='YOLOv5MixUp', use_cached=True, max_cached_images=20),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.1, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Reduce evaluation time
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/val.json',
    metric='bbox')

test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.5,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1.0e-5,
#         by_epoch=True,
#         begin=0,
#         end=5),
#     dict(type='MultiStepLR',
#          by_epoch=True,
#          milestones=[20, 40, 60, 80],
#          gamma=0.8),
# ]

# hooks
default_hooks = dict(checkpoint=dict(
        type='CheckpointHook',
        interval=interval,
        max_keep_ckpts=3,  # only keep latest 3 checkpoints
		save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), 
                                dict(type='WandbVisBackend', init_kwargs={'project': "MMYOLO",
                                    'name': "rtmdet-mobilenetv2-nano_prob%s"%(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
                                    })]) # WandB中的项目名称   