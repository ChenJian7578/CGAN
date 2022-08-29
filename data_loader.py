"""
数据集加载
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch



def data_load(batch_size, img_size):
    """
    数据加载\n
    :batch_size:    批量大小
    :img_size:      图片大小
    :return:        数据集迭代器
    """
    return torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,)




if __name__ == "__main__":
    data_loader = data_load(256, 32)

    for i, (imgs, labels) in enumerate(data_loader):
        print(f"i = {i}\n imgs.shape = {imgs.shape}\n labels = {labels}\n")
        if i == 0:
            break







