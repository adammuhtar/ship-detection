"""
File: model.py
Author: Adam Muhtar <adam.muhtar23@imperial.ac.uk>
Description: Training function for ShipClassifierConvNet models.
"""

import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    device: torch.device,
    checkpoint_path: Path,
    num_epochs: int = 30,
    return_losses: bool = True,
) -> tuple[list[float], list[float], list[float]] | None:
    """Train ShipClassifierConvNet models using the specified data loaders, loss
    function, optimiser, and learning rate scheduler.

    Args:
        model (`nn.Module`): PyTorch model to train.
        train_loader (`DataLoader`): DataLoader for the training set.
        val_loader (`DataLoader`): DataLoader for the validation set.
        criterion (`nn.Module`): Loss function.
        optimiser (`torch.optim.Optimizer`): Optimizer for training the model.
        scheduler (`torch.optim.lr_scheduler.StepLR`): Learning rate scheduler.
        device (`torch.device`): Device to run the training on.
        checkpoint_path (`Path`): Path to save model checkpoints.
        num_epochs (`int`): Number of epochs to train the model. Default is 30.
        return_losses (`bool`): Return training and validation losses and accuracies.
            Default is True.

    Returns:
        `tuple[list[float], list[float], list[float]]`: Training losses, validation
            losses, and validation accuracies for each epoch if `return_losses`
            is set to True.

    Raises:
        `Exception`: If an error occurs during training.
    """
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        try:
            model.train()
            running_loss = 0.0
            start_time = perf_counter()

            # Training loop
            for inputs, labels in tqdm(
                iterable=train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="batch",
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero gradients, forward pass, calculate loss, backward pass, optimiser step
                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()

                running_loss += loss.item()

            # Average training loss for the epoch
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)

            # Validation loop
            model.eval()
            val_running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Disable gradient calculations for validation
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    # Compute accuracy
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices
                    correct_predictions += torch.sum(preds == labels).item()
                    total_samples += labels.size(0)

            epoch_val_loss = val_running_loss / len(val_loader)
            epoch_val_accuracy = correct_predictions / total_samples
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)

            # Print stats
            elapsed_time = perf_counter() - start_time
            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}] - "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}, "
                f"Val Accuracy: {epoch_val_accuracy:.4f}, "
                f"Time: {elapsed_time:.2f}s"
            )

            # Step the scheduler
            scheduler.step()

            # Save checkpoint
            torch.save(
                obj=model.state_dict(),
                f=checkpoint_path / f"ship_classifier_epoch_{epoch + 1}.pt",
            )
        except Exception as e:
            logging.error(f"Error during epoch {epoch + 1}: {e}")
            raise e

    # Plot training and validation loss
    plt.figure(figsize=(16, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, num_epochs + 1),
        val_accuracies,
        label="Validation Accuracy",
        color="green",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if return_losses:
        return train_losses, val_losses, val_accuracies
