import time as t
import cv2
import os
import sys
import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Optional, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the Machine_Learning_Pipeline directory to sys.path for the ML packages to run
sys.path.append(os.path.abspath('Machine_Learning_Pipeline'))

try:
    from Machine_Learning_Pipeline import constant_variables as constant
    from Machine_Learning_Pipeline import ml_model
    import CV_process
    import CV_process_support
    import Sudoku_algorithm
    from helper_functions import find_best_model
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def initialize_camera() -> Optional[cv2.VideoCapture]:
    """
    Initialize and configure the camera for capturing video.

    Returns:
        Optional[cv2.VideoCapture]: A VideoCapture object if the camera is successfully
        initialized, or None if an error occurs.

    Raises:
        IOError: If the camera fails to open.
    """
    try:
        cap = cv2.VideoCapture(0)  # Open the camera (index 0, adjust if necessary)
        cap.set(3, 960)  # Width
        cap.set(4, 720)  # Height
        cap.set(10, 150)  # Brightness
        if not cap.isOpened():
            raise IOError("Failed to open camera")
        return cap
    except Exception as e:
        logging.error(f"Error initializing camera: {e}")
        return None

def load_model() -> Optional[nn.Module]:
    """
    Load the trained CNN model for digit recognition.

    Returns:
        Optional[nn.Module]: The loaded PyTorch model if successful, or None if an error occurs.

    Raises:
        FileNotFoundError: If no suitable model file is found.
    """
    try:
        model = ml_model.SimpleCNN()
        best_model_path = find_best_model()
        if best_model_path and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            accuracy = float(best_model_path.split('_acc_')[-1].split('.pth')[0])
            logging.info(f'Model loaded successfully: {os.path.basename(best_model_path)}')
            logging.info(f'Model accuracy: {accuracy}%')
            return model
        else:
            raise FileNotFoundError('No suitable model file found.')
    except FileNotFoundError as e:
        logging.error(str(e))
        logging.error("Please train the model first.")
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {str(e)}")
    return None

def process_frame(img: np.ndarray, model: nn.Module) -> Tuple[np.ndarray, Optional[Tuple[int, ...]], Optional[List[np.ndarray]], Optional[List[List[int]]]]:
    """
    Process a single frame of video to detect and recognize a Sudoku puzzle.

    Args:
        img (np.ndarray): The input image frame.
        model (nn.Module): The loaded CNN model for digit recognition.

    Returns:
        Tuple containing:
        - np.ndarray: The processed image result.
        - Optional[Tuple[int, ...]]: Recognized digits from the Sudoku grid, or None if not detected.
        - Optional[List[np.ndarray]]: Processed squares of the Sudoku grid, or None if not detected.
        - Optional[List[List[int]]]: Corners of the detected Sudoku grid, or None if not detected.

    Raises:
        Exception: If an error occurs during frame processing.
    """
    try:
        img_result = img.copy()
        img_corners = img.copy()
        processed_img = CV_process_support.preprocess(img)
        corners = CV_process.find_contours(processed_img, img_corners)
        
        if not corners:
            return img_result, None, None, None
        
        warped, matrix = CV_process.warp_image(corners, img)
        warped_processed = CV_process_support.preprocess(warped)
        vertical_lines, horizontal_lines = CV_process.get_grid_lines(warped_processed)
        mask = CV_process.create_grid_mask(vertical_lines, horizontal_lines)
        numbers = cv2.bitwise_and(warped_processed, mask)
        squares = CV_process.split_into_squares(numbers)
        squares_processed = CV_process.clean_squares(squares)
        resized_squares = CV_process.resize_to_64x64(squares_processed)
        squares_guesses = CV_process.recognize_digits(resized_squares, model)
        
        return img_result, tuple(squares_guesses), squares_processed, corners
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return img, None, None, None

def solve_sudoku(squares_guesses: Tuple[int, ...]) -> Optional[Tuple[List[int], float]]:
    """
    Attempt to solve the detected Sudoku puzzle.

    Args:
        squares_guesses (Tuple[int, ...]): The recognized digits from the Sudoku grid.

    Returns:
        Optional[Tuple[List[int], float]]: A tuple containing the solved puzzle and solving time
        if successful, or None if the puzzle couldn't be solved.

    Raises:
        Exception: If an error occurs during the solving process.
    """
    try:
        solved_puzzle, time = Sudoku_algorithm.solve_wrapper(squares_guesses)
        return (solved_puzzle, time) if solved_puzzle is not None else None
    except Exception as e:
        logging.error(f"Error solving Sudoku: {e}")
        return None