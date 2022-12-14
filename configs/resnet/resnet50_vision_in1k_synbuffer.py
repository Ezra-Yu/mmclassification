_base_ = ['../_base_/default_runtime.py']

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        zero_init_residual=False,
        init_cfg=[
            dict(type='Kaiming', layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
        topk=(1, 5),
    ),
    init_cfg=None,
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    broadcast_buffers=False,
)

data_preprocessor = dict(num_classes=1000, mean=None, std=None, to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ToPIL', to_rgb=True),
    dict(type='torchvision/RandomResizedCrop', size=176, interpolation=2),
    dict(type='torchvision/RandomHorizontalFlip', p=0.5),
    dict(type='torchvision/TrivialAugmentWide', interpolation=2),
    dict(type='torchvision/PILToTensor'),
    dict(type='torchvision/ConvertImageDtype', dtype=torch.float),
    dict(
        type='torchvision/Normalize',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    dict(type='torchvision/RandomErasing', p=0.1),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ToPIL', to_rgb=True),
    dict(type='torchvision/Resize', size=232, interpolation=2),
    dict(type='torchvision/CenterCrop', size=224),
    dict(type='torchvision/PILToTensor'),
    dict(type='torchvision/ConvertImageDtype', dtype=torch.float),
    dict(
        type='torchvision/Normalize',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)),
    dict(type='PackClsInputs'),
]

# dataset settings
dataset_type = 'ImageNet'
train_dataloader = dict(
    batch_size=128,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        ann_file='./data/imagenet/meta/train.txt',
        data_prefix='./data/imagenet/train',
        pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', num_repeats=4, shuffle=True))

val_dataloader = dict(
    batch_size=256,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_prefix='./data/imagenet/val',
        ann_file='./data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=2e-5),
    paramwise_cfg=dict(norm_decay_mult=0., ),
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
        type='EMAHook',
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

auto_scale_lr = dict(base_batch_size=1024)
