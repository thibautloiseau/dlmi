from torch.utils.data import Dataset
import os
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import random


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

        seed = random.randint(0, 2**32-1)

        if self.transform is not None:
            ct, dose, possible_dose_mask, structure_masks = self.transform(ct, dose, possible_dose_mask, structure_masks, seed)

        return {'ct': ct, 'dose': dose, 'possible_dose_mask': possible_dose_mask, 'structure_masks': structure_masks}

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
        # Get no sample to create submission file
        no_sample = self.samples[item][0].split('\\')[2]
        ct, possible_dose_mask, structure_masks = self.samples[item]

        ct = torch.from_numpy(np.load(ct))[None, :, :]
        possible_dose_mask = torch.from_numpy(np.load(possible_dose_mask))[None, :, :]
        structure_masks = torch.from_numpy(np.load(structure_masks))

        if self.transform is not None:
            ct, possible_dose_mask, structure_masks = self.transform(ct, possible_dose_mask, structure_masks)

        return {'ct': ct, 'possible_dose_mask': possible_dose_mask, 'structure_masks': structure_masks, 'no_sample': no_sample}

    def __len__(self):
        return len(self.samples)
