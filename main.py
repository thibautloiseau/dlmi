import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import RadDataset, RadDatasetTest
from utils import train
from models.unet import UNet
from models.vision_transformer import SwinUnet


if __name__ == '__main__':
    # Create folder for logs
    runs = [dir for dir in os.listdir('logs/') if os.path.isdir(f'logs/{dir}')]

    if not runs:
        new_run = 1

    else:
        new_run = max([int(el.split('_')[1]) for el in os.listdir('logs/')]) + 1

    writer = SummaryWriter(f'logs/run_{new_run}')

    # Hyperparameters
    lr = 0.001
    batch_size = 32
    num_epochs = 50

    # Data
    transform = None

    train_loader = DataLoader(RadDataset('train', transform=transform),
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(RadDataset('val', transform=transform),
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(RadDatasetTest(),
                             batch_size=batch_size,
                             shuffle=False)

    # # Create model
    # model = UNet()
    model = SwinUnet()

    # Launch training
    train(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, logger=writer)

