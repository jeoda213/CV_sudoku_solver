import os
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import constant_variables as constant

def split_data(data_dir: str, test_size: float = 0.3, val_test_size: float = 0.5, random_state: int = 42) -> tuple:
    """
    Load data from a CSV file and split it into training, validation, and test sets.

    Args:
        data_dir (str): Path to the CSV file containing the data.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.3.
        val_test_size (float, optional): Proportion of the test set to use for validation. Defaults to 0.5.
        random_state (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        tuple: Six NumPy arrays in the following order:
            - features_train: Training features
            - targets_train: Training targets
            - features_val: Validation features
            - targets_val: Validation targets
            - features_test: Test features
            - targets_test: Test targets

    Notes:
        - Labels are assumed to start from 1 and are adjusted to start from 0.
        - Feature values are normalized to the range [0, 1].
    """
    # Load data
    train = pd.read_csv(data_dir, dtype=np.float32)
    
    # Extract targets and normalize features
    targets_numpy = train.label.values - 1  # Subtract 1 from labels here
    features_numpy = train.loc[:, train.columns != "label"].values / 255  # normalization

    # Train-validation-test split
    features_train, features_temp, targets_train, targets_temp = train_test_split(
        features_numpy, targets_numpy, test_size=test_size, random_state=random_state
    )
    features_val, features_test, targets_val, targets_test = train_test_split(
        features_temp, targets_temp, test_size=val_test_size, random_state=random_state
    )
    
    return features_train, targets_train, features_val, targets_val, features_test, targets_test

def data_process(features_train: np.ndarray, targets_train: np.ndarray, 
                 features_val: np.ndarray, targets_val: np.ndarray, 
                 features_test: np.ndarray, targets_test: np.ndarray) -> tuple:
    """
    Process numpy arrays into PyTorch DataLoaders for model training and evaluation.

    Args:
        features_train (np.ndarray): Training features.
        targets_train (np.ndarray): Training targets.
        features_val (np.ndarray): Validation features.
        targets_val (np.ndarray): Validation targets.
        features_test (np.ndarray): Test features.
        targets_test (np.ndarray): Test targets.

    Returns:
        tuple: Four DataLoader objects in the following order:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - val_loader: DataLoader for validation data (repeated)
            - test_loader: DataLoader for test data

    Notes:
        - The batch size for DataLoaders is set in constant.batch_size.
        - Training data is shuffled, while validation and test data are not.
        - All DataLoaders drop the last batch if it's smaller than the batch size.
    """
    # Convert training features and targets to PyTorch tensors
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
    # Convert validation features and targets to PyTorch tensors
    featuresVal = torch.from_numpy(features_val)
    targetsVal = torch.from_numpy(targets_val).type(torch.LongTensor)
    # Convert test features and targets to PyTorch tensors
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

    ## DataLoader Configuration
    # Create TensorDataset for training data (features and targets)
    train_tensor = TensorDataset(featuresTrain, targetsTrain)
    # Create TensorDataset for validation data (features and targets)
    val_tensor = TensorDataset(featuresVal, targetsVal)
    # Create TensorDataset for testing data (features and targets)
    test_tensor = TensorDataset(featuresTest, targetsTest)
    
    # DataLoader for training data with batch_size and shuffling enabled
    train_loader = DataLoader(train_tensor, batch_size=constant.batch_size, shuffle=True, drop_last=True)
    # DataLoader for validation data with batch_size and no shuffling
    val_loader = DataLoader(val_tensor, batch_size=constant.batch_size, shuffle=False, drop_last=True)
    # DataLoader for testing data with batch_size and no shuffling
    test_loader = DataLoader(test_tensor, batch_size=constant.batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, val_loader, test_loader
