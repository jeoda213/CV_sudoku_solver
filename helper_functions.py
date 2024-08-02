import os
import glob
import logging
from Machine_Learning_Pipeline import constant_variables as constant

def find_best_model() -> str | None:
    """
    Find the best trained model based on accuracy.

    This function searches for model files in the specified machine learning folder,
    following the pattern 'digit_recognition_model_acc_*.pth'. It selects the model
    with the highest accuracy.

    Returns:
        str | None: The file path of the best model if found, None otherwise.
    """
    model_pattern = os.path.join(constant.machine_learning_folder_path, 'digit_recognition_model_acc_*.pth')
    model_files = glob.glob(model_pattern)
    if not model_files:
        return None
    
    def extract_accuracy(filename: str) -> float:
        """
        Extract the accuracy value from the model filename.

        Args:
            filename (str): The filename of the model.

        Returns:
            float: The accuracy value extracted from the filename.
        """
        return float(filename.split('_acc_')[-1].split('.pth')[0])
    
    best_model = max(model_files, key=extract_accuracy)
    return best_model

def setup_logging() -> logging.Logger:
    """
    Set up and configure logging for the application.

    This function configures the basic logging settings and creates a logger
    for the current module.

    Returns:
        logging.Logger: A configured logger object.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)