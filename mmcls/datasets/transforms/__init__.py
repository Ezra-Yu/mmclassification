# Copyright (c) OpenMMLab. All rights reserved.
from .auto_augment import (AutoAugment, AutoContrast, BaseAugTransform,
                           Brightness, ColorTransform, Contrast, Cutout,
                           Equalize, Invert, Posterize, RandAugment, Rotate,
                           Sharpness, Shear, Solarize, SolarizeAdd, Translate)
from .formatting import Collect, PackClsInputs, ToNumpy, ToPIL, Transpose
from .processing import (Albumentations, ColorJitter, EfficientNetCenterCrop,
                         EfficientNetRandomCrop, Lighting, RandomCrop,
                         RandomErasing, RandomResizedCrop, ResizeEdge)

__all__ = [
    'ToPIL', 'ToNumpy', 'Transpose', 'Collect', 'RandomCrop',
    'RandomResizedCrop', 'Shear', 'Translate', 'Rotate', 'Invert',
    'ColorTransform', 'Solarize', 'Posterize', 'AutoContrast', 'Equalize',
    'Contrast', 'Brightness', 'Sharpness', 'AutoAugment', 'SolarizeAdd',
    'Cutout', 'RandAugment', 'Lighting', 'ColorJitter', 'RandomErasing',
    'PackClsInputs', 'Albumentations', 'EfficientNetRandomCrop',
    'EfficientNetCenterCrop', 'ResizeEdge', 'BaseAugTransform'
]

import io

from mmcv.transforms.loading import LoadImageFromFile as BaseLoadImageFromFile
from PIL import Image

from mmcls.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PILLoadImageFromFile(BaseLoadImageFromFile):

    def transform(self, results: dict):
        filename = results['img_path']
        try:
            img_bytes = self.file_client.get(filename)
            buff = io.BytesIO(img_bytes)
            img = Image.open(buff)
            results['img'] = img.convert('RGB')

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['img_shape'] = results['img'].size
        results['ori_shape'] = results['img'].size
        return results
