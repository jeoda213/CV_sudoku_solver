import time
import os
import torch
import constant_variables as constant
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, 1, 64, 64)  # Reshape inputs, assuming 64x64 images

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    # Print test results
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Generate classification report
    class_names = [str(i) for i in range(1, 10)]  # Classes 1-9
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return accuracy, avg_loss

import os
import torch
import constant_variables as constant

def save_model(test_accuracy, trained_model):
    # Save the model if accuracy is higher than 98%
    if test_accuracy > 98.0:
        # Ensure the directory exists
        os.makedirs(constant.machine_learning_folder_path, exist_ok=True)
        
        # Generate a filename with the accuracy
        base_name, extension = os.path.splitext(constant.model_name)
        model_filename = f'{base_name}_acc_{test_accuracy:.2f}{extension}'
        
        # Construct the full save path
        save_path = os.path.join(constant.machine_learning_folder_path, model_filename)
        
        # Save the model
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        print(f"Model accuracy: {test_accuracy:.2f}%")
    else:
        print(f"Model not saved. Accuracy ({test_accuracy:.2f}%) below 98%")