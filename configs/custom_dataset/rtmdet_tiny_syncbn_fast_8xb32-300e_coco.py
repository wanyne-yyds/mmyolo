import time

_base_ = '../rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

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

deepen_factor = 0.167
widen_factor = 0.375
img_scale = _base_.img_scale
train_batch_size_per_gpu = 32
train_num_workers = 10
val_batch_size_per_gpu = 32
val_num_workers = 10
persistent_workers = True
strides = [8, 16, 32]
base_lr = 0.01
max_epochs = 300

batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

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
        eta_min=base_lr * 0.05,
        begin=max_epochs // 3,
        end=max_epochs,
        T_max=max_epochs // 1.5,
        by_epoch=True,
        convert_to_iter_based=True),
]

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            num_classes=num_classes)
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes)
    ))


train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=20,  # note
        random_pop=False,  # note
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        random_pop=False,
        max_cached_images=10,
        prob=0.5),
    dict(type='mmdet.PackDetInputs')
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
        ))

# Reduce evaluation time
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/val.json',
    metric='bbox')

test_evaluator = val_evaluator

test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=5,  # only keep latest 3 checkpoints
        save_best='auto'
    ))

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', 
                                        init_kwargs={'project': "MMYOLO",
                                         'name': "rtmdet-tiny-%s"%(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
                                         })]) # WandB中的项目名称          
