from typing import List, Optional, Sequence

from mmengine.runner import BaseLoop
import torch
from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmcls.registry import HOOKS, MODELS

DATA_BATCH = Optional[Sequence[dict]]

@HOOKS.register_module()
class LazyEMAHook(Hook):
    def __init__(self, 
                 ema_type: str = 'ExponentialMovingAverage',
                 lazy_interal=5,
                 **kwargs):
        self.ema_cfg = dict(type=ema_type, **kwargs)
        self.lazy_interal = lazy_interal

    def before_run(self, runner) -> None:
        """Create an ema copy of the model."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model
        self.ema_model = MODELS.build(
            self.ema_cfg, default_args=dict(model=self.src_model))

        if isinstance(runner.train_loop, BaseLoop):
            self.warmup_iters = self.lazy_interal * len(runner.train_loop.dataloader)
        else:
            self.warmup_iters = self.lazy_interal
    
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ema parameter."""
        if runner.iter < self.warmup_iters:
            self.ema_model.steps.fill_(0)
        self.ema_model.update_parameters(self.src_model)