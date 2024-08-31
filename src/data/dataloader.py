import logging

from torch.utils.data import default_collate, DataLoader
from torchvision.transforms import v2

from src.data.dataset import CIFAR100FS, CIFAR10FS
from src.data.transforms import create_train_transforms, create_test_transforms


def mixup_cutmix_collate_fn(mixup_alpha, cutmix_alpha, num_classes):
    mixup = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def _collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    return _collate_fn


DATASETS = {
    'cifar10': CIFAR10FS,
    'cifar100': CIFAR100FS,
    # 'imagenet': ImageNet,
}


def get_fixmatch_dataloader(cfg):
    aug = cfg.dataset.augmentation

    strong_train_transform = create_train_transforms(cfg.dataset.train_size[-1], aug.scale, aug.ratio, aug.mean,
                                                     aug.std, aug.auto_aug, aug.train_interpolation, aug.re_prob, as_list=True)
    weak_train_transform = create_train_transforms(cfg.dataset.train_size[-1], aug.scale, aug.ratio, aug.mean,
                                                   aug.std, None, aug.train_interpolation, aug.re_prob, as_list=False)

    test_transform = create_test_transforms(cfg.dataset.test_size[-1], aug.crop_pct, aug.mean, aug.std,
                                            aug.test_interpolation)

    labeled_train_ds = DATASETS[cfg.dataset.name](cfg.dataset.root, cfg.dataset.train, weak_train_transform,
                                                  samples_per_class=cfg.dataset.samples_per_class,
                                                  total_len=cfg.train.iter_per_epoch * cfg.train.batch_size)
    unlabeled_train_ds = DATASETS[cfg.dataset.name](cfg.dataset.root, cfg.dataset.train, strong_train_transform,
                                                    samples_per_class=cfg.dataset.samples_per_class,
                                                    total_len=cfg.train.iter_per_epoch * cfg.train.batch_size * cfg.train.mu)
    unlabeled_train_ds.to_unlabeled()

    test_ds = DATASETS[cfg.dataset.name](cfg.dataset.root, cfg.dataset.test, test_transform)

    collate_fn = None
    if aug.cutmix > 0. or aug.mixup > 0.:
        logging.warning('FixMatch is not used cutmix or mixup')

    labeled_train_dl = DataLoader(labeled_train_ds, cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.train.num_workers, collate_fn=collate_fn, pin_memory=True)
    unlabeled_train_dl = DataLoader(unlabeled_train_ds, cfg.train.batch_size * cfg.train.mu, shuffle=True,
                                    num_workers=cfg.train.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_dl = DataLoader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    return labeled_train_dl, unlabeled_train_dl, test_dl


def get_dataloader(setup, cfg):
    if 'fixmatch' == setup.lower():
        loaders = get_fixmatch_dataloader(cfg)
    elif 'fixmatchsimclr' == setup.lower():
        loaders = get_fixmatch_dataloader(cfg)
    else:
        loaders = get_base_dataloader(cfg)
    return loaders


def get_base_dataloader(cfg):
    aug = cfg.dataset.augmentation
    train_transform = create_train_transforms(cfg.dataset.train_size[-1], aug.scale, aug.ratio, aug.mean, aug.std,
                                              aug.auto_aug, aug.train_interpolation, aug.re_prob)
    test_transform = create_test_transforms(cfg.dataset.test_size[-1], aug.crop_pct, aug.mean, aug.std,
                                            aug.test_interpolation)

    train_ds = DATASETS[cfg.dataset.name](cfg.dataset.root, cfg.dataset.train, train_transform)
    test_ds = DATASETS[cfg.dataset.name](cfg.dataset.root, cfg.dataset.test, test_transform)

    collate_fn = None
    if aug.cutmix > 0. or aug.mixup > 0.:
        collate_fn = mixup_cutmix_collate_fn(aug.cutmix, aug.mixup, cfg.dataset.num_classes)

    train_dl = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
                          collate_fn=collate_fn, pin_memory=True)
    test_dl = DataLoader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    return train_dl, test_dl
