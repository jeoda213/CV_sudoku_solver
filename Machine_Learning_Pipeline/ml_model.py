import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        # Input: 1 channel (grayscale image)
        # Output: 16 feature maps
        # Kernel size: 3x3, Padding: 1 (to maintain spatial dimensions)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Batch normalization for the output of conv1
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional layer
        # Input: 16 feature maps (from conv1)
        # Output: 32 feature maps
        # Kernel size: 3x3, Padding: 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Batch normalization for the output of conv2
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third convolutional layer
        # Input: 32 feature maps (from conv2)
        # Output: 64 feature maps
        # Kernel size: 3x3, Padding: 1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Batch normalization for the output of conv3
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        # First FC layer
        # Input: 64 * 8 * 8 (flattened output from conv3 after max pooling)
        # Output: 256 neurons
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        # Batch normalization for the output of fc1
        self.bn4 = nn.BatchNorm1d(256)
        
        # Second FC layer
        # Input: 256 neurons (from fc1)
        # Output: 64 neurons
        self.fc2 = nn.Linear(256, 64)
        # Batch normalization for the output of fc2
        self.bn5 = nn.BatchNorm1d(64)
        
        # Output layer
        # Input: 64 neurons (from fc2)
        # Output: 9 neurons (one for each digit 1-9)
        self.fc3 = nn.Linear(64, 9)
        
        # Dropout layer for regularization
        # 30% of neurons will be randomly set to zero during training
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # ReLU activation
        x = F.max_pool2d(x, 2)  # Max pooling with 2x2 kernel
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # First fully connected layer
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second fully connected layer
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        # Note: no activation function here as we typically use CrossEntropyLoss
        # which applies softmax internally
        
        return x

    def l2_regularization(self):
        # Initialize the regularization term
        l2_reg = torch.tensor(0., requires_grad=True)
        
        # Iterate over all parameters
        for param in self.parameters():
            # Add the L2 norm of each parameter tensor to the regularization term
            l2_reg = l2_reg + torch.norm(param, 2)
        
        return l2_reg