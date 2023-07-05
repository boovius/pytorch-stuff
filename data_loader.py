import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_training_data(batch_size: int = 64, download = False):
    training_data = datasets.FashionMNIST(root="data", train=True, download=download, transform=ToTensor())

    # create dataloaders
    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    for X, y in training_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return training_dataloader


def load_testing_data(batch_size: int = 64, download=False):
    test_data = datasets.FashionMNIST(root="data", train=False, download = download, transform=ToTensor())

    # create dataloaders
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return test_dataloader


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
