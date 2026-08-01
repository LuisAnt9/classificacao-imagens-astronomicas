"""
Carregamento de dados e transformações (pré-processamento e data augmentation)
para o dataset de classificação de imagens astronômicas.

Estrutura esperada em data/:
    data/train/<classe>/*.jpg
    data/val/<classe>/*.jpg
    data/test/<classe>/*.jpg
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMG_SIZE = 128

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = IMG_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir: str = "data", img_size: int = IMG_SIZE,
                     batch_size: int = 32, num_workers: int = 0):
    train_transform, eval_transform = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
