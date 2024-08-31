import re

import torch
from hydra import initialize, compose
from rich.pretty import pprint as print

from src.data.randaug import RandAugmentMC

try:
    from torchvision.transforms import v2

    use_V2 = True
except:
    from torchvision import transforms as v2

    use_V2 = False

INTERPOLATION = {
    'bicubic': v2.InterpolationMode.BICUBIC,
    'bilinear': v2.InterpolationMode.BILINEAR,
    'nearest': v2.InterpolationMode.NEAREST,
    'nearest_exact': v2.InterpolationMode.NEAREST_EXACT,
    'box': v2.InterpolationMode.BOX,
    'hamming': v2.InterpolationMode.HAMMING,
    'lanczos': v2.InterpolationMode.LANCZOS,
}


def create_train_transforms(size, scale, ratio, mean, std, auto_aug, interpolation, re_prob, as_list=False):
    if size == 224:  # ImageNet
        crop_fn = v2.RandomResizedCrop(size, scale, ratio, interpolation=INTERPOLATION[interpolation])
    else:  # CIFAR
        crop_fn = v2.RandomCrop(size, size // 8)
    primary_tfl = [
        v2.PILToTensor() if use_V2 else v2.Lambda(lambda x: x),
        crop_fn,
        v2.RandomHorizontalFlip(),
    ]
    secondary_tfl = list()
    if auto_aug:
        auto_aug = auto_aug.split('-')
        if auto_aug[0] == 'rand':
            num_ops = 2
            magnitude = 9
            for c in auto_aug[1:]:
                k, v, _ = re.split(r'(\d.*)', c)
                if k == 'n':
                    num_ops = int(v)
                elif k == 'm':
                    magnitude = int(v)
                else:
                    raise ValueError(f'Unknown RandAug option: {k}')
            secondary_tfl.append(v2.RandAugment(num_ops, magnitude))

        elif auto_aug[0] == 'randaugMC':
            num_ops = 2
            magnitude = 9
            for c in auto_aug[1:]:
                k, v, _ = re.split(r'(\d.*)', c)
                if k == 'n':
                    num_ops = int(v)
                elif k == 'm':
                    magnitude = int(v)
                else:
                    raise ValueError(f'Unknown RandAug option: {k}')
            secondary_tfl.extend([
                v2.ToPILImage() if use_V2 else v2.Lambda(lambda x: x),
                RandAugmentMC(n=num_ops, m=magnitude),
                v2.ToTensor() if use_V2 else v2.Lambda(lambda x: x),
            ])

        elif auto_aug[0] == 'autoaug':
            if 'imagenet' in auto_aug[1].lower():
                policy = v2.AutoAugmentPolicy.IMAGENET
            elif 'cifar' in auto_aug[1].lower():
                policy = v2.AutoAugmentPolicy.CIFAR10
            elif 'svhn' in auto_aug[1].lower():
                policy = v2.AutoAugmentPolicy.SVHN
            else:
                raise ValueError(f'Unknown AutoAug policy: {auto_aug[1]}')
            secondary_tfl.append(v2.AutoAugment(policy))

        elif auto_aug[1] == 'augmix':
            severity = 3
            mixture_width = 3
            chain_depth = -1
            alpha = 1.0
            for c in auto_aug[1:]:
                k, v, _ = re.split(r'(\d.*)', c)
                if k == 's':
                    severity = int(v)
                elif k == 'm':
                    mixture_width = int(v)
                elif k == 'c':
                    chain_depth = int(v)
                elif k == 'a':
                    alpha = float(v)
                else:
                    raise ValueError(f'Unknown AugMix option: {k}')
            secondary_tfl.append(v2.AugMix(severity, mixture_width, chain_depth, alpha))
    if re_prob:
        secondary_tfl.append(v2.RandomErasing(re_prob))

    final_tfl = [
        v2.ToDtype(torch.float32, True) if use_V2 else v2.ToTensor(),
        v2.Normalize(mean, std)
    ]
    if as_list:
        return v2.Compose(primary_tfl), v2.Compose(secondary_tfl), v2.Compose(final_tfl)
    return v2.Compose(primary_tfl + secondary_tfl + final_tfl)


def create_test_transforms(size, crop_pct, mean, std, interpolation):
    resize = int(size / crop_pct)

    return v2.Compose([
        v2.PILToTensor() if use_V2 else v2.Lambda(lambda x: x),
        v2.Resize(resize, interpolation=INTERPOLATION[interpolation]),
        v2.CenterCrop(size),
        v2.ToDtype(torch.float32, True) if use_V2 else v2.ToTensor(),
        v2.Normalize(mean, std)
    ])

if __name__ == '__main__':
    with initialize('../../configs', version_base='1.3'):
        cfg = compose('config', overrides=['+setup=resnet50_cifar', 'dataset.augmentation.auto_aug=null',
                                           '+dataset.augmentation.autoaug=True'])

    aug = cfg.dataset.augmentation
    print(aug.auto_aug)
    train_ts = create_train_transforms(cfg.dataset.train_size[-1], aug.scale, aug.ratio, aug.mean, aug.std,
                                       aug.auto_aug, aug.train_interpolation, aug.re_prob)
    test_ts = create_test_transforms(cfg.dataset.test_size[-1], aug.crop_pct, aug.mean, aug.std, aug.test_interpolation)

    print(train_ts)
    print(test_ts)
