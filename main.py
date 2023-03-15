import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data import RadDataset
from utils import train
from model import UNet


if __name__ == '__main__':
    lr = 0.0001
    batch_size = 16
    num_epochs = 15

    transform = transforms.Compose([transforms.Normalize(0, 1)])

    train_loader = DataLoader(RadDataset('train', transform=transform),
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(RadDataset('validation', transform=transform),
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(RadDataset('test'),
                             batch_size=batch_size,
                             shuffle=False)

    model = UNet()

    train(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)

