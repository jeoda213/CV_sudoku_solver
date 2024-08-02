import time
import torch
import os
import ml_model
import constant_variables
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

def plot_metrics(train_metric: list, val_metric: list, metric_name: str) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        train_metric (list): List of training metric values for each epoch.
        val_metric (list): List of validation metric values for each epoch.
        metric_name (str): Name of the metric being plotted (e.g., 'Loss' or 'Accuracy').

    Returns:
        None

    This function creates and displays a plot showing the training and validation
    metrics over the course of training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_metric, label=f'Training {metric_name}')
    plt.plot(val_metric, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.show()

def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                device: torch.device, num_epochs: int = 5, learning_rate: float = 0.001) -> nn.Module:
    """
    Train a neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device (CPU or CUDA) to use for training.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 5.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        nn.Module: The trained model.

    This function trains the model on the provided training data, validates it on the
    validation data, and prints progress and metrics throughout the training process.
    It also plots the training and validation loss and accuracy after training.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Get the total number of batches
    total_batches = len(train_loader)

    # Loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Start time for the epoch
        epoch_start_time = time.time()

        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move inputs and targets to the device (CPU or GPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Reshape inputs to (batch_size, channels, height, width)
            inputs = inputs.view(-1, 1, 64, 64)  # Assuming 64x64 images

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed_time = time.time() - epoch_start_time
                progress = (batch_idx + 1) / total_batches
                eta = elapsed_time / progress - elapsed_time
                print(f'\rEpoch [{epoch+1}/{num_epochs}] '
                      f'Batch [{batch_idx+1}/{total_batches}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}% '
                      f'ETA: {eta:.0f}s', end='')

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.view(-1, 1, 64, 64)  # Reshape inputs

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
              f'Time: {epoch_time:.0f}s')

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, 'Loss')
    plot_metrics(train_accuracies, val_accuracies, 'Accuracy')

    return model