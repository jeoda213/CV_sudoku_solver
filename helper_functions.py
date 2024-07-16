import os
import glob
from Machine_Learning_Pipeline import constant_variables as constant

def find_best_model():
    model_pattern = os.path.join(constant.machine_learning_folder_path, 'digit_recognition_model_acc_*.pth')
    model_files = glob.glob(model_pattern)
    if not model_files:
        return None
    
    def extract_accuracy(filename):
        return float(filename.split('_acc_')[-1].split('.pth')[0])
    
    best_model = max(model_files, key=extract_accuracy)
    return best_model

# Add other utility functions here as needed