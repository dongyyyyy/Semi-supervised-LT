import numpy as np
from PIL import Image

# For GaussianBlur
import random
from PIL import ImageFilter

import torchvision
from torchvision import datasets
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

from dataset.function import train_split

# Parameters for data
cifar100_mean = (0.5071, 0.4867, 0.4408)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2675, 0.2565, 0.2761)
# Augmentations.


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3
    
class TransformTrio:
    def __init__(self, transform, transform2,transform3):
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        out4 = self.transform3(inp)
        out5 = self.transform3(inp)
        return out1, out2, out3, out4, out5

def get_cifar100_fortest(root, l_samples, u_samples,num_classes=100, download=True):
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    base_dataset = datasets.CIFAR100(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs= train_split(base_dataset.targets, l_samples, u_samples,num_classes=num_classes)
    
    train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_val)
    train_unlabeled_dataset = CIFAR100_labeled(root, train_unlabeled_idxs, train=True,
                                                transform=transform_val)
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(root, l_samples, u_samples,train_strong=False,num_classes=100, download=True):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    transform_strong.transforms.insert(0, RandAugment(3, 4))
    transform_strong.transforms.append(CutoutDefault(16))

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    base_dataset = datasets.CIFAR100(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs= train_split(base_dataset.targets, l_samples, u_samples,num_classes=num_classes)
    if train_strong == False:
        train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    else:
        train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=TransformTwice(transform_train,transform_strong))
    train_unlabeled_dataset = CIFAR100_unlabeled(root, train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100_forContrastive(root, l_samples, u_samples,train_strong=False,num_classes=100, download=True):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    # augmentation_regular is same with transform_strong (skip...)    
    # augmentation_regular = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     CIFAR10Policy(),    # add AutoAug
    #     transforms.ToTensor(),
    #     Cutout(n_holes=1, length=16),
    #     transforms.Normalize(
    #         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    
    augmentation_sim_cifar = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_strong.transforms.insert(0, RandAugment(3, 4))
    transform_strong.transforms.append(CutoutDefault(16))

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    base_dataset = datasets.CIFAR100(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs= train_split(base_dataset.targets, l_samples, u_samples,num_classes=num_classes)
    if train_strong == False:
        train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    else:
        # train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_strong)
        train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=TransformTwice(transform_train,augmentation_sim_cifar))
        
    
    train_unlabeled_dataset = CIFAR100_unlabeled(root, train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



class CIFAR100_labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(CIFAR100_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100_unlabeled(CIFAR100_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(CIFAR100_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])