# Computer Vision Sudoku Solver

This project is a real-time Sudoku solver using computer vision and machine learning techniques. It captures Sudoku puzzles through a camera, processes the image, recognizes the digits, solves the puzzle, and overlays the solution onto the original image.

## Features

- Real-time video capture and processing
- Sudoku grid detection and extraction
- Digit recognition using a Convolutional Neural Network (CNN)
- Efficient Sudoku solving algorithm
- Augmented reality display of the solution

## Prerequisites

- Python 3.7+
- OpenCV
- PyTorch
- NumPy

## Project Structure

### CV Sudoku
This section contains the core logic for Sudoku processing and solving.

- `main.py`: The main script that runs the Sudoku solver
- `CV_process.py`: Contains functions for image processing and grid extraction
- `CV_process_support.py`: Helper functions for image processing
- `Sudoku_algorithm.py`: Implements the Sudoku solving algorithm

### Machine Learning
This section includes all components related to the digit recognition model.

- `ML_pipeline.py`: Consolidates the entire machine learning workflow
- `ml_model.py`: Defines the CNN model for digit recognition
- `model_training.py`: Script for training the digit recognition model
- `model_evaluation.py`: Functions for evaluating the trained model
- `constant_variables.py`: Contains constant values and configurations
- `data_preprocessing.py`: Functions for preprocessing training data
- `data_processing.py`: Functions for processing data during runtime


## Usage

1. Train the digit recognition model (if not already trained):

3. Point your camera at a Sudoku puzzle. The solution will be overlaid on the image in real-time.

4. Press 'q' to quit the application.

## Model Training

If you want to train the digit recognition model on your own dataset:

1. Place your training images in the `original_training_data` folder.
2. Run the `ML_pipeline.py` script to preprocess the data and train the model.

## Credits

This project includes code adapted from the following sources:

- [sudoku-solver-opencv](https://github.com/dharm1k987/sudoku-solver-opencv) by Dharmik Shah: Used for training dataset and studying the use of cv2 library.
- The Sudoku solving algorithm is based on the work by Ali Assaf (https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt)
Thank you to these developers for their contributions to the open-source community.

### Modifications
The following modifications have been made to the original code:
- Image resizing
- New convolutional neural network
- Model training
- Model evaluation

These changes can be found in the relevant Python files of this project.

