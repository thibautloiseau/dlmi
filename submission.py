import torch
import numpy as np
from tqdm import tqdm
import os

from models.unet import UNet
from data import RadDatasetTest


def load_model(path):
    model = UNet()
    model.load_state_dict(torch.load(path))
    return model


def create_submission_file(model, dataset, path):
    # Create path
    if not os.path.exists(path):
        os.makedirs(path)

    # Cuda available ?
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if cuda:
        model.cuda()

    with torch.no_grad():
        for sample in tqdm(dataset):
            # Get inputs
            ct, possible_dose_mask, structure_masks, no_sample = sample['ct'], sample['possible_dose_mask'], sample['structure_masks'], sample['no_sample']

            # Concatenate ct and structure masks
            x = torch.cat([ct, structure_masks], dim=0).type(Tensor)[None, :]  # Adding first dimension for batch size

            # Get prediction
            y_pred = model(x, possible_dose_mask.type(Tensor)).cpu().numpy().squeeze()

            # Save prediction
            np.save(os.path.join(path, no_sample), y_pred)


if __name__ == '__main__':
    # Loading trained model
    model = load_model('checkpoints/model.pt')
    dataset = RadDatasetTest()
    path = 'submissions/test'

    create_submission_file(model, dataset, path)


