import random
import torch
import torchvision.transforms as T

from utils import FixRandomSeed


class Transform:
    def __init__(self, n_geo=2, mode='train'):
        self.n_geo = n_geo
        self.mode = mode

        self.geo_list = [
            T.RandomRotation(180),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ]
        self.transform_input = [
            T.RandomErasing()
        ]

    def __call__(self, ct, dose, possible_dose_mask, structure_masks, seed):
        ct = T.Normalize(0, 0.5)(ct)
        structure_masks = T.Normalize([0 for _ in range(10)], [0.5 for _ in range(10)])(structure_masks)

        if self.mode == 'train':
            geo_transform = T.Compose([
                    random.choice(self.geo_list) for _ in range(self.n_geo)
                ])
            input_transform = T.Compose([t for t in self.transform_input])

            with FixRandomSeed(seed):
                ct = geo_transform(ct)
            with FixRandomSeed(seed):
                ct = input_transform(ct)
            with FixRandomSeed(seed):
                dose = geo_transform(dose)
            with FixRandomSeed(seed):
                possible_dose_mask = geo_transform(possible_dose_mask)
            with FixRandomSeed(seed):
                structure_masks = geo_transform(structure_masks)

        return ct, dose, possible_dose_mask, structure_masks


class TransformTest:
    def __call__(self, ct, possible_dose_mask, structure_masks):
        ct = T.Normalize(0, 0.5)(ct)
        structure_masks = T.Normalize([0 for _ in range(10)], [0.5 for _ in range(10)])(structure_masks)

        return ct, possible_dose_mask, structure_masks
