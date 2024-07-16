# import torch
# import torch.nn as nn
# import os
# import time
# import traceback
# # Necessary modules
# import constant_variables as constant
# import data_preprocessing as pre_process
# import data_processing as process
# import ml_model
# import model_training as training
# import model_evaluation as eval

# def main():
#     model_path = constant.model_directory
    
#     if not os.path.exists(model_path):
#         try:
#             ## checking if processed csv file already exists 
#             if not os.path.exists(constant.processed_data_directory):
#                 try:
#                     ########## Data preprocessing ##########
#                     ## turning the images to grey-style and resize them
#                     processed_images = pre_process.process_images(constant.original_image_folder_path)
#                     print('step 2 complete')
#                     ## save image dataset as csv file
#                     pre_process.save_data_matrix_as_csv(processed_images, constant.csv_data_folder_path, constant.processed_image_file_name)
#                     print('step 3 complete')

#                 except Exception as e:
#                     print(f"Error occurred: {str(e)}")
#                     print(traceback.format_exc())

#             ########## Data processsing ##########
#             ## Spliting data for training, validation, and testing
#             features_train, targets_train, features_val, targets_val, features_test, targets_test = process.split_data(os.path.join(constant.csv_data_folder_path, constant.processed_image_file_name))
#             print('step 4 complete')
#             ## convert the dataset to DataLoader for ease of use during training and testing
#             train_loader, val_loader, val_loader, test_loader = process.data_process(features_train, targets_train, features_val, targets_val, features_test, targets_test)
#             print('step 5 complete')

#             ########## Model ##########
#             # Create an instance of the SimpleCNN model
#             cnn_model = ml_model.SimpleCNN()
#             # Define the loss function (cross-entropy loss) for classification tasks
#             criterion = nn.CrossEntropyLoss()
#             # Define the optimizer (Adam optimizer) for updating the model parameters during training
#             optimizer = torch.optim.Adam(cnn_model.parameters(), lr=constant.learning_rate, weight_decay=constant.weight_decay)
#             print('step 6 complete')

#             ########## Model Training ##########
#             # Assuming you've defined device in ML_pipeline.py as well
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             trained_model = training.train_model(cnn_model, train_loader, val_loader, optimizer, device)
#             print('step 7 complete')

#             ########## model evaluation ##########
#             acc, ave_loss = eval.test_model(trained_model, test_loader, device)
#             print('step 8 complete')
#             ######### save the trained and evaluated model ##########
#             eval.save_model(acc, trained_model)

#         except Exception as e:
#             print(f"Error occurred: {str(e)}")
#             print(traceback.format_exc())
#     else:
#         print(f"Model already exists at {model_path}. Skipping pipeline execution.")

# if __name__ == "__main__":
#     main()


import os
import logging
import traceback
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import constant_variables as const
import data_preprocessing as preprocess
import data_processing as process
import ml_model
import model_training as training
import model_evaluation as eval
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_functions import find_best_model, setup_logging

class MLPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logging()

    def check_and_preprocess_data(self) -> None:
        if not os.path.exists(const.processed_data_directory):
            try:
                self.logger.info("Starting data preprocessing...")
                processed_images = preprocess.process_images(const.original_image_folder_path)
                preprocess.save_data_matrix_as_csv(processed_images, const.csv_data_folder_path, const.processed_image_file_name)
                self.logger.info("Data preprocessing completed successfully.")
            except Exception as e:
                self.logger.error(f"Error in data preprocessing: {str(e)}")
                raise

    def process_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        try:
            self.logger.info("Starting data processing...")
            features_train, targets_train, features_val, targets_val, features_test, targets_test = process.split_data(
                os.path.join(const.csv_data_folder_path, const.processed_image_file_name)
            )
            train_loader, val_loader, _, test_loader = process.data_process(
                features_train, targets_train, features_val, targets_val, features_test, targets_test
            )
            self.logger.info("Data processing completed successfully.")
            return train_loader, val_loader, test_loader
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            raise

    def create_and_train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        try:
            self.logger.info("Creating and training model...")
            model = ml_model.SimpleCNN().to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=const.learning_rate, weight_decay=const.weight_decay)
            trained_model = training.train_model(model, train_loader, val_loader, optimizer, self.device)
            self.logger.info("Model training completed successfully.")
            return trained_model
        except Exception as e:
            self.logger.error(f"Error in model creation or training: {str(e)}")
            raise

    def evaluate_and_save_model(self, model: nn.Module, test_loader: DataLoader) -> None:
        try:
            self.logger.info("Evaluating model...")
            accuracy, avg_loss = eval.test_model(model, test_loader, self.device)
            self.logger.info(f"Model evaluation completed. Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")
            eval.save_model(accuracy, model)
        except Exception as e:
            self.logger.error(f"Error in model evaluation or saving: {str(e)}")
            raise

    def run(self):
        try:
            model_path = find_best_model()
            if model_path and os.path.exists(model_path):
                self.logger.info(f"Existing model found: {model_path}. Skipping pipeline execution.")
                return

            self.check_and_preprocess_data()
            train_loader, val_loader, test_loader = self.process_data()
            trained_model = self.create_and_train_model(train_loader, val_loader)
            self.evaluate_and_save_model(trained_model, test_loader)
            self.logger.info("Pipeline execution completed successfully.")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.debug(traceback.format_exc())

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run()