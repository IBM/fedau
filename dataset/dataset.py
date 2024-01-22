import os
import sys

import torch, torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from torchvision.datasets import FashionMNIST
from PIL import Image
from typing import Any, Callable, Optional, Tuple

import numpy as np

# These enhanced classes do transformation only once and save the data in memory, to speed up the overall computation

class FashionMNISTEnhanced(FashionMNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device: torch.device = None,
    ) -> None:
        super(FashionMNISTEnhanced, self).__init__(root, train, transform, target_transform, download)

        self.data_transformed = []
        self.targets_transformed = []

        for img in self.data:
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.stack(self.targets_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


class CIFAR10Enhanced(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device : torch.device = None,
    ) -> None:
        super(CIFAR10Enhanced, self).__init__(root, train, transform, target_transform, download)

        self.data_transformed = []
        self.targets_transformed = []

        for img in self.data:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.tensor(self.targets_transformed, dtype=torch.int64)  # Note: this is different from the MNIST class

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


class CIFAR100Enhanced(CIFAR100):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device : torch.device = None,
    ) -> None:
        super(CIFAR100Enhanced, self).__init__(root, train, transform, target_transform, download)

        self.data_transformed = []
        self.targets_transformed = []

        for img in self.data:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.tensor(self.targets_transformed, dtype=torch.int64)  # Note: this is different from the MNIST class

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


class SVHNEnhanced(SVHN):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device : torch.device = None,
    ) -> None:
        super(SVHNEnhanced, self).__init__(root, 'train' if train else 'test', transform, target_transform, download)

        # For consistency with other datasets
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.targets = self.labels

        self.data_transformed = []
        self.targets_transformed = []

        for img in self.data:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.tensor(self.targets_transformed, dtype=torch.int64)  # Note: this is different from the MNIST class

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


class CINIC10Enhanced(ImageFolder):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device : torch.device = None,
    ) -> None:
        if download:
            raise NotImplementedError('Download is not implemented for CINIC-10, please download manually '
                                      'from https://datashare.ed.ac.uk/handle/10283/3192 and extract the files into a folder'
                                      'named CINIC-10 inside the dataset main folder')
        if train:
            path = os.path.join(root, 'CINIC-10', 'train')
        else:
            path = os.path.join(root, 'CINIC-10', 'test')
        super(CINIC10Enhanced, self).__init__(path, transform, target_transform)

        self.data_transformed = []
        self.targets_transformed = []

        for img_metadata in self.imgs:
            img = Image.open(img_metadata[0]).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

            target = img_metadata[1]
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.tensor(self.targets_transformed, dtype=torch.int64)  # Note: this is different from the MNIST class

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


def load_data(dataset, data_path, device):
    if dataset == 'FashionMNIST':
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        data_train = FashionMNISTEnhanced(data_path,
                                  transform=transform_normalize,
                                  download=True, device=device)  # download=True for the first time
        data_test = FashionMNISTEnhanced(data_path,
                                 train=False,
                                 transform=transform_normalize, device=device)

    elif dataset == 'CIFAR10':
        transform_normalize_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_normalize_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR10Enhanced(data_path,
                             transform=transform_normalize_train,
                             download=True, device=device)  # download=True for the first time
        data_test = CIFAR10Enhanced(data_path,
                            train=False,
                            transform=transform_normalize_test, device=device)

    elif dataset == 'CIFAR100':
        transform_normalize_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_normalize_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR100Enhanced(data_path,
                                     transform=transform_normalize_train,
                                     download=True, device=device)  # download=True for the first time
        data_test = CIFAR100Enhanced(data_path,
                                    train=False,
                                    transform=transform_normalize_test, device=device)
    elif dataset == 'SVHN':
        transform_normalize_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        transform_normalize_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        data_train = SVHNEnhanced(data_path,
                                      transform=transform_normalize_train,
                                      download=True, device=device)  # download=True for the first time
        data_test = SVHNEnhanced(data_path,
                                     train=False,
                                     download=True,
                                     transform=transform_normalize_test, device=device)
    elif dataset == 'CINIC10':
        transform_normalize_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))
        ])
        transform_normalize_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))
        ])

        data_train = CINIC10Enhanced(data_path,
                                  transform=transform_normalize_train,
                                  download=False, device=device)  # Cannot automatically download, CINIC10 dataset needs to be downloaded from https://datashare.ed.ac.uk/handle/10283/3192
        data_test = CINIC10Enhanced(data_path,
                                 train=False,
                                 download=False,
                                 transform=transform_normalize_test, device=device)
    else:
        raise Exception('Unknown dataset name.')

    return data_train, data_test
