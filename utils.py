import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, train_loader, val_loader, num_epochs=10, lr=0.0001):
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss function
    criterion = torch.nn.L1Loss()

    if cuda:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----------
    #  Training
    # ----------

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}')
        train_loss, eval_loss = 0.0, 0.0

        print('Training')
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            ct = batch["ct"].type(Tensor)
            possible_dose_mask = batch["possible_dose_mask"].type(Tensor)
            structure_masks = batch["structure_masks"].type(Tensor)
            dose = batch["dose"].type(Tensor)

            # Remove stored gradients
            optimizer.zero_grad()

            # Generate output
            x = torch.cat([ct, structure_masks], dim=1)
            y_pred = model(x, possible_dose_mask)

            # Compute the corresponding loss
            loss = criterion(y_pred, dose)
            train_loss += loss.item()

            # Compute the gradient and perform one optimization step
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        # Validation at end of epoch
        print('Validation')

        # Take one random sample to visualize result
        rdm_sample = np.random.random_integers(0, len(val_loader))

        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader)):
                ct = batch["ct"].type(Tensor)
                possible_dose_mask = batch["possible_dose_mask"].type(Tensor)
                structure_masks = batch["structure_masks"].type(Tensor)
                dose = batch["dose"].type(Tensor)

                # Generate output
                x = torch.cat([ct, structure_masks], dim=1)
                y_pred = model(x, possible_dose_mask)

                # Compute the corresponding loss
                loss = criterion(y_pred, dose)
                eval_loss += loss.item()

                if i == rdm_sample:
                    pred_to_plot = y_pred[0].cpu().permute(1, 2, 0)  # Take first example in batch
                    ct_to_plot = ct[0].cpu().permute(1, 2, 0)
                    dose_to_plot = dose[0].cpu().permute(1, 2, 0)

        eval_loss /= len(val_loader.dataset)

        print(f'Train loss: {train_loss}\n'
              f'Eval loss: {eval_loss}')

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(pred_to_plot, cmap='jet')
        axes[1].imshow(ct_to_plot, cmap='jet')
        axes[2].imshow(dose_to_plot, cmap='jet')
        plt.show()

    return
