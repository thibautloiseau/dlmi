import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


def train(model, train_loader, val_loader, num_epochs=10, lr=0.0001, logger=None):
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
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # ----------
    #  Training
    # ----------

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}')
        train_loss, eval_loss = [], []

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
            train_loss.append(loss.item())

            # Compute the gradient and perform one optimization step
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

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
                eval_loss.append(loss.item())

                # if i == rdm_sample:
                #     pred_to_plot = y_pred[0].cpu().permute(1, 2, 0)  # Take first example in batch
                #     print(torch.unique(pred_to_plot))
                #     ct_to_plot = ct[0].cpu().permute(1, 2, 0)
                #     dose_to_plot = dose[0].cpu().permute(1, 2, 0)

        eval_loss = np.mean(eval_loss)

        print(f'Train loss: {train_loss}\n'
              f'Eval loss: {eval_loss}')

        # Logging results
        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('eval_loss', eval_loss, epoch)

        scheduler.step()

        save_checkpoint(model)
        # fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(pred_to_plot, cmap='jet')
        # axes[1].imshow(ct_to_plot, cmap='jet')
        # axes[2].imshow(dose_to_plot, cmap='jet')
        # plt.show()

    return


def save_checkpoint(model):
    torch.save(model.state_dict(), 'checkpoints/model.pt')
    return

