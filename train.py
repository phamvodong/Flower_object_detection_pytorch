from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

try:
    from config import (DEVICE, NUM_EPOCHS, OUT_DIR, TRAIN_DIR, VALID_DIR)
    from dataset import create_train_dataset, create_val_dataset
    from evaluate import learning_curve
    from model import get_model
except Exception:
    from Object_detection.config import (DEVICE, NUM_EPOCHS, OUT_DIR, TRAIN_DIR, VALID_DIR)
    from Object_detection.dataset import create_train_dataset, create_val_dataset
    from Object_detection.evaluate import learning_curve
    from Object_detection.model import get_model


def training(train_dataloader, model):
    """
    Trains a PyTorch model on a training dataset.
    :param train_dataloader:
    :param model:
    :return:
    """
    loss = 0

    # Train the model on the training dataset
    for images, targets in tqdm(train_dataloader):
        # Move the data to the device.
        images = list(image.to(DEVICE) for image in images)
        images = [image.float() / 255.0 for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Zero the gradients and compute the loss.
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = {k: v.mean() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        losses_reduced.backward()
        optimizer.step()

        # Update the parameters and print the loss.
        loss += losses.item()

    return loss


if __name__ == '__main__':
    train_dl = create_train_dataset(TRAIN_DIR)
    val_dl = create_val_dataset(VALID_DIR)

    _model = get_model(9)

    _model.to(DEVICE)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    CHECKPOINT_INTERVAL = 1

    print("Start training - - -")

    for epoch in range(NUM_EPOCHS):
        correct = 0

        timestamp_start = datetime.now()

        running_loss = training(train_dl, _model)

        lr_scheduler.step()

        # Save the checkpoint after each epoch
        if epoch % CHECKPOINT_INTERVAL == 0:
            try:
                arr = np.load(OUT_DIR + 'variable/model_paths.npy')
                last_epochs = arr[-1]
            except FileNotFoundError:
                last_epochs = 0

            print(last_epochs + epoch)
            checkpoint_path = OUT_DIR + f"model_epoch{last_epochs + epoch}.pt"
            torch.save(_model.state_dict(), checkpoint_path)

        timestamp_end = datetime.now()

        print(
            f'Epoch: {epoch}, Train Loss: {running_loss / len(train_dl)}, Time: {(timestamp_end - timestamp_start).total_seconds()}s')

    print('Evaluating ---')
    try:
        _model_paths = np.load(OUT_DIR + 'variable/model_paths.npy')
        last_epochs = _model_paths[-1]
    except FileNotFoundError:
        last_epochs = 0
        _model_paths = []

    try:
        _accuracies_train = np.load(OUT_DIR + 'variable/accuracies_train.npy')
    except FileNotFoundError:
        _accuracies_train = []

    try:
        _accuracies_val = np.load(OUT_DIR + 'variable/accuracies_val.npy')
    except FileNotFoundError:
        _accuracies_val = []

    learning_curve(last_epochs, last_epochs + NUM_EPOCHS, _model_paths, _accuracies_train, _accuracies_val, train_dl,
                   val_dl)
