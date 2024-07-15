import os
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import constant_variables as constant

def split_data(data_dir, test_size=0.3, val_test_size=0.5, random_state=42):
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

def data_process(features_train, targets_train, features_val, targets_val, features_test, targets_test):
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
