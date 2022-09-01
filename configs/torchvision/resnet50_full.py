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
        dict(type='Mixup', alpha=0.2, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]),
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(weight_decay=0.00002),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

# dataset settings
dataset_type = 'ImageNet'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = preprocess_cfg['mean'][::-1]
bgr_std = preprocess_cfg['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=176,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='vision_ta_wide',
        num_policies=1,
        total_level=31,
        magnitude_level=31,
        magnitude_std='inf',
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(
        type='RandomErasing',
        erase_prob=0.1,
        mode='rand',
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
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=2e-5))

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

    # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
    # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
    #
    # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps) = [(1287056 / 8) * 600] / (128 * 32) = 23567
    # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
    # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs = 8 * 128 * 32 * 23567 / 600 = 1287072
    # adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs  = 8 * 128 * 32 / 600 = 54.61333333333334
    # alpha = 1.0 - args.model_ema_decay     # 2-05   0.00002 
    # alpha = min(1.0, alpha * adjust)  # min(1, 0.00002 * 54.61333333333334 = 2.56e-5) = 0.0010923
    dict(
        type='LazyEMAHook',
        momentum=0.0010923,
        lazy_interal=5,
        interval=32,
        update_buffers=True,
        priority='ABOVE_NORMAL')
]
