import torch
import torch.nn as nn
import os
import time
import traceback
# Necessary modules
import constant_variables as constant
import data_preprocessing as pre_process
import data_processing as process
import ml_model
import model_training as training
import model_evaluation as eval

def main():
    model_path = constant.model_directory
    
    if not os.path.exists(model_path):
        try:
            ## checking if processed csv file already exists 
            if not os.path.exists(constant.processed_data_directory):
                try:
                    ########## Data preprocessing ##########
                    ## turning the images to grey-style and resize them
                    processed_images = pre_process.process_images(constant.original_image_folder_path)
                    print('step 2 complete')
                    ## save image dataset as csv file
                    pre_process.save_data_matrix_as_csv(processed_images, constant.csv_data_folder_path, constant.processed_image_file_name)
                    print('step 3 complete')

                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print(traceback.format_exc())

            ########## Data processsing ##########
            ## Spliting data for training, validation, and testing
            features_train, targets_train, features_val, targets_val, features_test, targets_test = process.split_data(os.path.join(constant.csv_data_folder_path, constant.processed_image_file_name))
            print('step 4 complete')
            ## convert the dataset to DataLoader for ease of use during training and testing
            train_loader, val_loader, val_loader, test_loader = process.data_process(features_train, targets_train, features_val, targets_val, features_test, targets_test)
            print('step 5 complete')

            ########## Model ##########
            # Create an instance of the SimpleCNN model
            cnn_model = ml_model.SimpleCNN()
            print('step 6 complete')
            # Define the loss function (cross-entropy loss) for classification tasks
            criterion = nn.CrossEntropyLoss()
            print('step 7 complete')
            # Define the optimizer (Adam optimizer) for updating the model parameters during training
            optimizer = torch.optim.Adam(cnn_model.parameters(), lr=constant.learning_rate, weight_decay=constant.weight_decay)
            print('step 8 complete')

            ########## Model Training ##########
            # Assuming you've defined device in ML_pipeline.py as well
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trained_model = training.train_model(cnn_model, train_loader, val_loader, optimizer, device)
            print('step 9 complete')

            ########## model evaluation ##########
            eval.test_model(trained_model, test_loader, device)
            print('step 10 complete')

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
    else:
        print(f"Model already exists at {model_path}. Skipping pipeline execution.")

if __name__ == "__main__":
    main()