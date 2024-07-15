import os

#folder path
original_image_folder_path = '/Users/danieljeong/Python/CV_sudoku_solver/original_training_data'
csv_data_folder_path ='/Users/danieljeong/Python/CV_sudoku_solver/processed_training_data'
main_project_folder_path = '/Users/danieljeong/Python/CV_sudoku_solver'

#file names 
processed_image_file_name = 'images_data.csv'
model_name = 'digit_recognition_model.pth'

# parameters for fine-tuning
num_epochs = 5
batch_size = 36
resize_length = 64
learning_rate = 0.001
weight_decay = 1e-5

model_directory = os.path.join(original_image_folder_path, model_name)
processed_data_directory = os.path.join(csv_data_folder_path, processed_image_file_name)