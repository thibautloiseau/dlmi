from torch.utils.data import Dataset
import os
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import random


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class RadDataset(Dataset):
    def __init__(self, mode, transform=None):
        super().__init__()

        if mode not in ['train', 'val']:
            raise Exception('mode should be "train" or "val"')

        self.mode = mode
        self.transform = transform
        self.samples = []

        files = sorted(os.listdir(f'data/{self.mode}'))
        patient_ids = list(set([i.split('_')[1] for i in files]))

        for i in patient_ids:
            self.samples.append((os.path.join('data', self.mode, 'sample_' + i, 'ct.npy'),
                                 os.path.join('data', self.mode, 'sample_' + i, 'dose.npy'),
                                 os.path.join('data', self.mode, 'sample_' + i, 'possible_dose_mask.npy'),
                                 os.path.join('data', self.mode, 'sample_' + i, 'structure_masks.npy')))

    def __getitem__(self, item):
        ct, dose, possible_dose_mask, structure_masks = self.samples[item]

        ct = torch.from_numpy(np.load(ct))[None, :, :]
        dose = torch.from_numpy(np.load(dose))[None, :, :]
        possible_dose_mask = torch.from_numpy(np.load(possible_dose_mask))[None, :, :]
        structure_masks = torch.from_numpy(np.load(structure_masks))

        sample = {'ct': ct, 'dose': dose, 'possible_dose_mask': possible_dose_mask, 'structure_masks': structure_masks}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


class RadDatasetTest(Dataset):
    def __init__(self, transform=None):
        super().__init__()

        self.transform = transform
        self.samples = []

        files = sorted(os.listdir('data/test'))
        patient_ids = list(set([i.split('_')[1] for i in files]))

        for i in patient_ids:
            self.samples.append((os.path.join(f'data', 'test', 'sample_' + i, 'ct.npy'),
                                 os.path.join(f'data', 'test', 'sample_' + i, 'possible_dose_mask.npy'),
                                 os.path.join(f'data', 'test', 'sample_' + i, 'structure_masks.npy')))

    def __getitem__(self, item):
        ct, possible_dose_mask, structure_masks = self.samples[item]

        ct = torch.from_numpy(np.load(ct))[None, :, :]
        possible_dose_mask = torch.from_numpy(np.load(possible_dose_mask))[None, :, :]
        structure_masks = torch.from_numpy(np.load(structure_masks))

        sample = {'ct': ct, 'possible_dose_mask': possible_dose_mask, 'structure_masks': structure_masks}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

