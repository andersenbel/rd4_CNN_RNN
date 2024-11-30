import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=64):
    # Трансформація для навчальної вибірки з аугментацією
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Трансформація для тестової вибірки (без аугментації)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
