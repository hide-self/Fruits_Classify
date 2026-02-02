import os

import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_train_dataloader(batch_size: int):
    # 训练数据预处理(包含增强)
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.687,0.589,0.430],
            std=[0.298,0.293,0.340]
        )
    ])

    # 加载训练集数据
    train_dataset = torchvision.datasets.ImageFolder(
        root='./data/train',
        transform=train_transform
    )

    # 创建训练集数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return train_dataloader


def get_test_dataloader(batch_size: int):
    # 测试数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.687, 0.589, 0.430],
            std=[0.298, 0.293, 0.340]
        )
    ])

    # 加载测试集数据
    test_dataset = torchvision.datasets.ImageFolder(
        root='./data/test',
        transform=test_transform
    )

    # 创建训练集数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return test_dataloader


def getAllClasses():
    # 获取目录下所有文件和文件夹
    all_items = os.listdir('./data/train')

    # 筛选出文件夹
    folders = [item for item in all_items
               if os.path.isdir(os.path.join('./data/train', item))]

    return folders


if __name__ == '__main__':
    print(getAllClasses())  # 打印所有分类的名称

