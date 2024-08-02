from PIL import Image
import os
import csv
import numpy as np
import constant_variables as constant

# Function to convert image to grayscale and resize to 64x64
def preprocess_image(image_path: str) -> Image.Image:
    """
    Convert an image to grayscale and resize it to a specified dimension.

    Args:
        image_path (str): The file path of the input image.

    Returns:
        Image.Image: A PIL Image object that has been resized to the dimensions
                     specified in constant.resize_length.
    """
    img = Image.open(image_path) # Open image
    img_resized = img.resize((constant.resize_length, constant.resize_length)) # Resize to 64x64
    
    return img_resized

# Function to convert resized image to flattened NumPy array
def image_to_array(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image object to a flattened NumPy array.

    Args:
        img (Image.Image): The input PIL Image object.

    Returns:
        np.ndarray: A 1D NumPy array containing the flattened image data.
    """
    img_array = np.array(img) # Convert image to NumPy array
    flattened_array = img_array.flatten() # Flatten the array
    
    return flattened_array

# Function to process images and return the data matrix
def process_images(img_dir: str) -> np.ndarray:
    """
    Process all PNG images in a directory and create a data matrix.

    This function reads all PNG images in the specified directory, preprocesses them,
    extracts labels from filenames, and combines all data into a single matrix.

    Args:
        img_dir (str): The directory containing the input images.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents an image.
                    The first column is the label, followed by pixel values.
    """
    data = [] # Initialize an empty list to store flattened arrays
    # Loop through each image in the input directory
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):  # Assuming all images are PNG format
            # Extract label from filename
            label = int(filename.split('-')[0])  # Extract the number from the filename
            img = preprocess_image(os.path.join(img_dir, filename)) # Preprocess image
            img_array = image_to_array(img) # Convert image to flattened NumPy array
            img_array_with_label = np.insert(img_array, 0, label) # Append label to the beginning of the flattened array
            data.append(img_array_with_label) # Append array with label to the list
    data_matrix = np.stack(data) # Convert list of arrays to a single NumPy matrix

    return data_matrix

# Function to save the data matrix as a CSV file
def save_data_matrix_as_csv(data_matrix: np.ndarray, output_dir: str, csv_filename: str) -> None:
    """
    Save a data matrix as a CSV file with appropriate column names.

    Args:
        data_matrix (np.ndarray): The 2D NumPy array containing the data to be saved.
        output_dir (str): The directory where the CSV file will be saved.
        csv_filename (str): The name of the output CSV file.

    Returns:
        None

    Notes:
        This function prints information about the save operation to the console.
    """
    csv_filepath = os.path.join(output_dir, csv_filename) # Specify the full path for the CSV file
    # Save the data matrix as a CSV file with column names
    print(f"Saving to directory: {constant.csv_data_folder_path}")
    print(f"File name: {constant.processed_image_file_name}")
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Define column names
        column_names = ['label'] + [f'pixel_{i}' for i in range(1, (constant.resize_length ** 2)  + 1)]  # Assuming 64x64 images
        writer.writerow(column_names) # Write column names to the CSV file
        writer.writerows(data_matrix) # Write data matrix to the CSV file
    print(f"Processed data saved to {csv_filepath}")