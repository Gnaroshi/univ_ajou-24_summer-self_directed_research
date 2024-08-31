import copy
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from hydra.utils import get_original_cwd
from torchvision.datasets import CIFAR100, CIFAR10


def sampling(dataset_name, n_shot, imgs, targets):
    imgs = np.array(imgs)
    targets = np.array(targets)

    try:
        sample_path = Path(get_original_cwd(), 'sampling', f'{dataset_name}_{n_shot}s.json')
    except:
        sample_path = Path('sampling', f'{dataset_name}_{n_shot}s.json')

    if sample_path.exists():
        print(f'Load exist samples: {sample_path}')
        with open(sample_path, 'rb') as f:
            data = pickle.load(f)
            return data['labeled_sample_idx'], data['unlabeled_sample_idx']

    print(f'Generate new samples: {sample_path}')
    data_dict = defaultdict(list)
    for i in range(len(imgs)):
        data_dict[targets[i]].append(i)

    labeled_sample_idx = list()
    unlabeled_sample_idx = list()
    for class_num, idx in data_dict.items():
        idx = np.array(idx)
        idx = idx[np.random.permutation(idx.shape[0])]
        labeled_sample_idx.extend(idx[:n_shot])
        unlabeled_sample_idx.extend(idx[n_shot:])

    sample_path.parent.mkdir(exist_ok=True)
    with open(sample_path, 'wb') as f:
        pickle.dump(dict(labeled_sample_idx=labeled_sample_idx, unlabeled_sample_idx=unlabeled_sample_idx), f,
                    protocol=3)

    return labeled_sample_idx, unlabeled_sample_idx


class FewShotBaseClass:
    def to_unlabeled(self):
        self.data = self.origin_data[self.unlabeled_idx]
        self.targets = self.origin_targets[self.unlabeled_idx]
        self.dataset_size = len(self.data)

    def to_labeled(self):
        self.data = self.origin_data[self.labeled_idx]
        self.targets = self.origin_targets[self.labeled_idx]
        self.dataset_size = len(self.data)

    def two_way_getitem(self, index):
        index = index % self.dataset_size
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        weak_img = self.weak_transform(img)
        strong_img = self.strong_transform(weak_img)

        return self.normalize_transform(weak_img), self.normalize_transform(strong_img), target

    def one_way_getitem(self, index):
        index = index % self.dataset_size
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target


class CIFAR10FS(CIFAR10, FewShotBaseClass):
    def __init__(self, root, split, transform=None, target_transform=None, samples_per_class=None, total_len=None):
        train = True if split.lower() == 'train' else False

        if isinstance(transform, (list, tuple)):
            self.weak_transform, self.strong_transform, self.normalize_transform = transform
            transform = None
        self.total_len = total_len


        super().__init__(root, train, transform, target_transform, download=True)
        self.origin_data = np.array(copy.deepcopy(self.data))
        self.origin_targets = np.array(copy.deepcopy(self.targets))

        if samples_per_class is not None:
            self.labeled_idx, self.unlabeled_idx = sampling('cifar10', samples_per_class, self.data, self.targets)
            self.to_labeled()

        self.dataset_size = len(self.data)

        if transform is None:
            self.getitem_fn = self.two_way_getitem
        else:
            self.getitem_fn = self.one_way_getitem

    def __len__(self):
        return self.total_len if self.total_len is not None else len(self.data)

    def __getitem__(self, item):
        return self.getitem_fn(item)


class CIFAR100FS(CIFAR100, FewShotBaseClass):
    def __init__(self, root, split, transform=None, target_transform=None, samples_per_class=None, total_len=None):
        train = True if split.lower() == 'train' else False
        if isinstance(transform, (list, tuple)):
            self.weak_transform, self.strong_transform, self.normalize_transform = transform
            transform = None
        self.total_len = total_len
        super().__init__(root, train, transform, target_transform)
        self.origin_data = np.array(copy.deepcopy(self.data))
        self.origin_targets = np.array(copy.deepcopy(self.targets))

        if samples_per_class is not None:
            self.labeled_idx, self.unlabeled_idx = sampling('cifar10', samples_per_class, self.data, self.targets)
            self.to_labeled()
        self.dataset_size = len(self.data)
        if transform is None:
            self.getitem_fn = self.two_way_getitem
        else:
            self.getitem_fn = self.one_way_getitem

    def __len__(self):
        return self.total_len if self.total_len is not None else len(self.data)

    def __getitem__(self, item):
        return self.getitem_fn(item)


if __name__ == '__main__':
    ds = CIFAR100FS('/data', 'train', 10, samples_per_class=4)
    print(len(ds))
    print(np.unique(ds.targets, return_counts=True))
