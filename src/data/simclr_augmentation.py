'''
    modified by Mingyu Jung, code from
    https://github.com/sthalles/SimCLR/blob/master/data_aug/gaussian_blur.py
    https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    https://github.com/sthalles/SimCLR/blob/master/data_aug/view_generator.py
'''

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms, v2
from PIL import Image

np.random.seed(0)


class GaussianBlur(object):

    def __init__(self, kernel_size, device):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3).to(device)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3).to(device)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )


    def __call__(self, img):
        # img = self.pil_to_tensor(img).unsqueeze(0)
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img).unsqueeze(0)
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        return img


def simclr_transform(size, device, s=1):
    color_jitter = v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = v2.Compose([
        v2.RandomApply([color_jitter], p=0.8),
        v2.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size), device=device),
        v2.ToTensor()])
    return data_transforms.to(device)


class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
