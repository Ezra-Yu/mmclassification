_base_ = ['../_base_/models/resnet50.py', '../_base_/default_runtime.py']

# model settings
model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        )),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.2),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=2e-5),
    paramwise_cfg=dict(norm_decay_mult=0.),
)

# dataset settings
dataset_type = 'ImageNet'
preprocess_cfg = dict(
    num_classes=1000,
    mean=[123, 116, 103],
    std=[58, 57, 57],
    to_rgb=True,
)

bgr_mean = preprocess_cfg['mean'][::-1]
bgr_std = preprocess_cfg['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=176, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='vision_ta_wide',
        num_policies=1,
        total_level=31,
        magnitude_level=31,
        magnitude_std='inf',
        hparams=dict(pad_val=bgr_mean)),
    dict(
        type='RandomErasing',
        erase_prob=0.1,
        mode='const',
        fill_color=bgr_mean,
        min_area_ratio=0.02,
        max_area_ratio=0.33,
        aspect_range=(0.3, 3.3)),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=232, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True, num_repeats=4),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=6),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1.0e-6,
        by_epoch=True,
        begin=6)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# decay(torchvision) = 1 - momentum
custom_hooks = [
    dict(
        type='LazyEMAHook',
        momentum=0.0010923,
        begin_epoch=5,
        interval=32,
        update_buffers=True,
        evaluate_on_ema=True,
        evaluate_on_origin=True,
        priority='ABOVE_NORMAL')
]
