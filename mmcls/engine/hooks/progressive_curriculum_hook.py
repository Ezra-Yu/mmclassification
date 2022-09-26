# Copyright (c) OpenMMLab. All rights reserved
from mmengine.hooks import Hook
from mmengine.runner import Runner, EpochBasedTrainLoop, IterBasedTrainLoop
from typing import List, Optional, Sequence
from collections import Iterable
from mmengine.logging import print_log

from mmcls.datasets.transforms import RandomResizedCrop
from mmcls.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]

@HOOKS.register_module()
class ProgressiveCurriculumHook(Hook):
    """Progressive Learning Curriculum Hook."""
    def __init__(self, curriculum_cfg: dict = dict()) -> None:
        self.curriculum_cfg = curriculum_cfg

    def before_train(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj: `IterBasedRunner`): Iter based Runner.
        """
        transforms = runner.train_loop.dataloader.dataset.pipeline.transforms
        assert isinstance(transforms, Iterable)
        self.rrc_index = None

        for i, sub_transform in enumerate(transforms):
            if isinstance(sub_transform, RandomResizedCrop):
                self.rrc_index = i
                return 
        
        print_log("There is no `RandomResizedCrop` in your training pipelie,"
                "The `ProgressiveCurriculumHook` would not work.", 
                level='WARNING')
        


    def before_train_epoch(self, runner: Runner) -> None:
        """Calculate prcise BN and broadcast BN stats across GPUs.

        Args:
            runner (obj:`Runner`): The runner of the training process.
        """
        if self.rrc_index and isinstance(runner.train_loop, EpochBasedTrainLoop) \
            and runner.epoch in self.curriculum_cfg:
            transforms = runner.train_loop.dataloader.dataset.pipeline.transforms
            transforms[self.rrc_index].scale = self.curriculum_cfg[runner.epoch]
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader = runner.train_loop.dataloader
            if hasattr(train_loader, 'persistent_workers'
                        ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            
            print_log(transforms[self.rrc_index])

    def before_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Calculate prcise BN and broadcast BN stats across GPUs.

        Args:
            runner (obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        if self.rrc_index and isinstance(runner.train_loop, IterBasedTrainLoop) \
            and runner.iter in self.curriculum_cfg:
            transforms = runner.train_loop.dataloader.dataset.pipeline.transforms
            transforms[self.rrc_index].scale = self.curriculum_cfg[runner.epoch]
            print_log(transforms[self.rrc_index])
