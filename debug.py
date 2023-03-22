import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from data import RadDataset

if __name__ == '__main__':
    lr = 0.0001
    batch_size = 16
    num_epoch = 15

    transform = transforms.Compose([transforms.Normalize(0, 1)])
    train_set = RadDataset('train', transform=transform)

    for sample in train_set:
        ct, dose = sample['ct'].permute(1, 2, 0), sample['dose'].permute(1, 2, 0)

        print(torch.unique(dose))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(ct, cmap='jet')
        axes[1].imshow(dose, cmap='jet')
        plt.show()
