from torch.utils.data import Dataset
import os
import torch
import numpy as np


class RadDataset(Dataset):
    def __init__(self, mode, transform=None):
        super().__init__()

        self.mode = mode
        self.transform = transform
        self.samples = []

        if self.mode in ['train', 'validation']:
            files = sorted(os.listdir(f'data/{self.mode}'))
            patient_ids = list(set([i.split('_')[1] for i in files]))

            for i in patient_ids:
                self.samples.append((os.path.join('data', self.mode, 'sample_' + i, 'ct.npy'),
                                     os.path.join('data', self.mode, 'sample_' + i, 'dose.npy'),
                                     os.path.join('data', self.mode, 'sample_' + i, 'possible_dose_mask.npy'),
                                     os.path.join('data', self.mode, 'sample_' + i, 'structure_masks.npy')))

        elif self.mode == 'test':
            files = sorted(os.listdir('data/test'))
            patient_ids = list(set([i.split('_')[1] for i in files]))

            for i in patient_ids:
                self.samples.append((os.path.join(f'data', 'test', 'sample_' + i, 'ct.npy'),
                                     '',
                                     os.path.join(f'data', 'test', 'sample_' + i, 'possible_dose_mask.npy'),
                                     os.path.join(f'data', 'test', 'sample_' + i, 'structure_masks.npy')))

    def __getitem__(self, item):
        ct, dose, possible_dose_mask, structure_masks = self.samples[item]

        ct = torch.from_numpy(np.load(ct))[None, :, :]
        dose = torch.from_numpy(np.load(dose))[None, :, :]
        possible_dose_mask = torch.from_numpy(np.load(possible_dose_mask))[None, :, :]
        structure_masks = torch.from_numpy(np.load(structure_masks))

        if self.transform is not None:
            ct = self.transform(ct)
            dose = self.transform(dose)

        return {'ct': ct, 'dose': dose, 'possible_dose_mask': possible_dose_mask, 'structure_masks': structure_masks}

    def __len__(self):
        return len(self.samples)
