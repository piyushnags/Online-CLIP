from typing import Tuple
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100



def get_cifar100() -> Tuple[Dataset, Dataset]:
    train_set = CIFAR100(root='data/', train=True)
    test_set = CIFAR100(root='data/', train=False)

    return train_set, test_set



if __name__ == '__main__':
    pass