import time as t
import cv2
import os
import sys
import torch
import torch.nn as nn

# Add the Machine_Learning_Pipeline directory to sys.path for the ML packages to run
sys.path.append(os.path.abspath('Machine_Learning_Pipeline'))
import CV_process
import CV_process_support
import Sudoku_algorithm
import Machine_Learning_Pipeline.ML_pipeline
from Machine_Learning_Pipeline import constant_variables as constant
from Machine_Learning_Pipeline import ml_model
import Sudoku_algorithm
from helper_functions import find_best_model  # Import the function from utils.py

frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(2)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

# load the model with weights
model = ml_model.SimpleCNN()
best_model_path = find_best_model()

try:
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        accuracy = float(best_model_path.split('_acc_')[-1].split('.pth')[0])
        print(f'Model loaded successfully: {os.path.basename(best_model_path)}')
        print(f'Model accuracy: {accuracy}%')
    else:
        raise FileNotFoundError('No suitable model file found.')
except FileNotFoundError as e:
    print(str(e))
    print("Please train the model first.")
except Exception as e:
    print(f"An error occurred: {str(e)}")


prev = 0

seen = dict()

while True:
    time_elapsed = t.time() - prev

    success, img = cap.read()

    if time_elapsed > 1. / frame_rate:
        prev = t.time()

        img_result = img.copy()
        img_corners = img.copy()

        processed_img = CV_process_support.preprocess(img)
        corners = CV_process.find_contours(processed_img, img_corners)

        if corners:
            warped, matrix = CV_process.warp_image(corners, img)
            warped_processed = CV_process_support.preprocess(warped)

            vertical_lines, horizontal_lines = CV_process.get_grid_lines(warped_processed)
            mask = CV_process.create_grid_mask(vertical_lines, horizontal_lines)
            numbers = cv2.bitwise_and(warped_processed, mask)

            squares = CV_process.split_into_squares(numbers)
            squares_processed = CV_process.clean_squares(squares)
            resized_squares = CV_process.resize_to_64x64(squares_processed)
            squares_guesses = CV_process.recognize_digits(resized_squares, model)

            # Convert squares_guesses to a tuple to use as a dictionary key
            squares_guesses = tuple(squares_guesses)

            # if it is impossible, continue
            if squares_guesses in seen and seen[squares_guesses] is False:
                continue

            # if we already solved this puzzle, just fetch the solution
            if squares_guesses in seen:
                CV_process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                img_result = CV_process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])

            else:
                solved_puzzle, time = Sudoku_algorithm.solve_wrapper(squares_guesses)
                if solved_puzzle is not None:
                    CV_process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                    img_result = CV_process.unwarp_image(warped, img_result, corners, time)
                    seen[squares_guesses] = [solved_puzzle, time]

                else:
                    seen[squares_guesses] = False

    cv2.imshow('window', img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()