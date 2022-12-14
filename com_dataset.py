import torch

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

dataset_cfg1 = dict(
    type='ImageNet',
    ann_file='./data/imagenet/meta/train.txt',
    data_prefix='./data/imagenet/train',
    pipeline=train_pipeline)

dataset_cfg2 = dict(
    type='VisionImageNet',
    ann_file='./data/imagenet/meta/train.txt',
    data_prefix='./data/imagenet/train')

from mmcls.datasets import build_dataset
from mmcls.registry import TRANSFORMS
from mmcls.utils import register_all_modules

register_all_modules()

dataset1 = build_dataset(dataset_cfg1)
print(dataset1.pipeline)
dataset2 = build_dataset(dataset_cfg2)

for i in range(len(dataset1)):
    print(i)
    print(dataset1[i]['inputs'].shape, dataset2[i]['inputs'].shape)
    torch.allclose(dataset1[i]['inputs'], dataset2[i]['inputs'])
    torch.allclose(dataset1[i]['data_samples'].gt_label.label,
                   dataset2[i]['data_samples'].gt_label.label)
    print()
