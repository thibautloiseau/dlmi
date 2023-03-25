import os

import torch.hub
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn

from data import RadDataset, RadDatasetTest
from utils import train
from models.unet import UNet
# from models.vision_transformer import SwinUnet
from augment import Transform


if __name__ == '__main__':
    # Create folder for logs
    runs = [dir for dir in os.listdir('logs/') if os.path.isdir(f'logs/{dir}')]

    if not runs:
        new_run = 1

    else:
        new_run = max([int(el.split('_')[1]) for el in os.listdir('logs/')]) + 1

    writer = SummaryWriter(f'logs/run_{new_run}')

    # Hyperparameters
    lr = 1e-3
    batch_size = 32
    num_epochs = 60

    # Data
    transform_train = Transform(mode='train')
    transform_val = Transform(mode='val')

    train_loader = DataLoader(RadDataset('train', transform=transform_train),
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(RadDataset('val', transform=transform_val),
                            batch_size=batch_size,
                            shuffle=False)

    # Create model
    # UNet with upsample
    model = UNet()

    # # Pretrained UNet
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    #
    # # Adjusting last layer to finetune
    # model.conv = nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=1)

    # Launch training
    train(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, logger=writer, with_structure_masks=True)
