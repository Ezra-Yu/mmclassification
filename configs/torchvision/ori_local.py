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

# dataset settings
dataset_type = 'VisionImageNet'
data_preprocessor = dict(
    num_classes=1000,
    mean=[0., 0., 0.],
    std=[1., 1., 1.],
    to_rgb=False,
)

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file='./data/imagenet/meta/train.txt',
        data_prefix='./data/imagenet/train'),
    sampler=dict(type='RepeatAugSampler', shuffle=True, num_repeats=4),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_prefix='./data/imagenet/val',
        ann_file='./data/imagenet/meta/val.txt'),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=2e-5),
    paramwise_cfg=dict(norm_decay_mult=0.),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=6),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1.0e-6, by_epoch=True, begin=6)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()

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

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
