import torch

_base_ = ['./ori_local']

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

# dataset settings
train_dataloader = dict(dataset=dict(type='ImageNet', pipeline=train_pipeline))
