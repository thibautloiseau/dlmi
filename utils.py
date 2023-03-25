import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import random


def train(model, train_loader, val_loader, num_epochs=10, lr=0.001, logger=None, with_structure_masks=False):
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss function
    criterion = torch.nn.L1Loss()

    # Cuda handling
    if cuda:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    # ----------
    #  Training
    # ----------

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}')
        train_loss, eval_loss = [], []

        print('Training')
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            # Get inputs
            ct = batch["ct"].type(Tensor)
            possible_dose_mask = batch["possible_dose_mask"].type(Tensor)
            structure_masks = batch["structure_masks"].type(Tensor)
            dose = batch["dose"].type(Tensor)

            # Remove stored gradients
            optimizer.zero_grad()

            # Generate output
            # With structure masks
            if with_structure_masks:
                x = torch.cat([ct, structure_masks], dim=1)
                y_pred = model(x, possible_dose_mask)

            # Without structure mask
            else:
                x = ct
                y_pred = model(x, possible_dose_mask)

            # Compute the corresponding loss
            loss = criterion(y_pred, dose)
            train_loss.append(loss.item())

            # Compute the gradient and perform one optimization step
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

        # Validation at end of epoch
        print('Validation')

        with torch.no_grad():  # Validation, no need to store gradients
            for i, batch in tqdm(enumerate(val_loader)):
                # Get inputs
                ct = batch["ct"].type(Tensor)
                possible_dose_mask = batch["possible_dose_mask"].type(Tensor)
                structure_masks = batch["structure_masks"].type(Tensor)
                dose = batch["dose"].type(Tensor)

                # Generate output
                # With structure masks
                if with_structure_masks:
                    x = torch.cat([ct, structure_masks], dim=1)
                    y_pred = model(x, possible_dose_mask)

                # Without structure mask
                else:
                    x = ct
                    y_pred = model(x, possible_dose_mask)

                # Compute the corresponding loss
                loss = criterion(y_pred, dose)
                eval_loss.append(loss.item())

        eval_loss = np.mean(eval_loss)

        print(f'Train loss: {train_loss}\n'
              f'Eval loss: {eval_loss}')

        # Logging results
        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('eval_loss', eval_loss, epoch)

        # Step for lr_scheduler
        scheduler.step()

        # Save model
        save_checkpoint(model)

    return


def save_checkpoint(model):
    torch.save(model.state_dict(), 'checkpoints/model.pt')
    return


class FixRandomSeed:
    """
    This class fixes the seeds for numpy, torch and random pkgs.
    """
    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()
        self.torchbackup = torch.get_rng_state()

    def __enter__(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)
        torch.set_rng_state(self.torchbackup)

