import timm
import torch
from termcolor import colored
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from torch import nn


class ObjectFactory:
    def __init__(self, cfg, fabric):
        self.cfg = cfg
        self.train = cfg.train
        self.optim = cfg.train.optimizer
        self.scheduler = cfg.train.scheduler
        self.dataset = cfg.dataset
        self.model = cfg.model
        self.fabric = fabric

    def create_model(self):
        with self.fabric.init_module():
            model = timm.create_model(
                **self.model,
                in_chans=self.dataset.in_channels,
                num_classes=self.dataset.num_classes,
            )

        if self.train.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if self.model.scriptable:
            assert not self.train.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            model = torch.jit.script(model)

        return model

    def create_optimizer_and_scheduler(self, model, iter_per_epoch):
        self.cfg.train.iter_per_epoch = iter_per_epoch
        self.train.iter_per_epoch = iter_per_epoch
        self.check_total_batch_size()

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=self.optim), filter_bias_and_bn=True)

        updates_per_epoch = \
            (iter_per_epoch + self.optim.grad_accumulation - 1) // self.optim.grad_accumulation

        scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(self.scheduler),
            updates_per_epoch=updates_per_epoch,
        )

        if self.optim.clip_grad is not None:
            self.fabric.clip_gradients(model, optimizer, self.optim.clip_grad)
        return optimizer, scheduler, num_epochs

    def create_criterion(self):
        if self.dataset.augmentation.cutmix > 0 or self.dataset.augmentation.mixup > 0:
            if self.train.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=self.train.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()

        elif self.dataset.augmentation.smoothing > 0:
            if self.train.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=self.dataset.augmentation.smoothing,
                                                   target_threshold=self.train.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.dataset.augmentation.smoothing)

        else:
            train_loss_fn = nn.CrossEntropyLoss()

        train_loss_fn = train_loss_fn
        validate_loss_fn = nn.CrossEntropyLoss()

        self.cfg.train.criterion = train_loss_fn.__class__.__name__
        return train_loss_fn, validate_loss_fn

    def check_total_batch_size(self):
        if self.train.total_batch is None:
            self.train.total_batch = self.cfg.train.batch_size
            self.train.optimizer.grad_accumulation = 1
            return

        total_batch = self.train.total_batch
        grad_accum = self.optim.grad_accumulation
        batch_size = self.train.batch_size

        if total_batch < self.cfg.world_size * batch_size:
            batch_size = total_batch // self.cfg.world_size
            self.fabric.print(colored(
                f'[WARNING] Batch size({self.train.batch_size}) is too larger than total batch({total_batch}). Batch size will be set to {batch_size}',
                'red'))

        self.train.optimizer.grad_accumulation = total_batch // (self.cfg.world_size * batch_size)
        self.train.batch_size = batch_size
        self.train.total_batch = total_batch
        self.optim.grad_accumulation = self.train.optimizer.grad_accumulation

        if self.train.batch_size % grad_accum != 0:
            self.fabric.print(f'{colored("[WARNING]", "red")} '
                              f'The batch size({self.train.batch_size}) does not divided grad accumulation size({grad_accum})')
