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

# Import modularized functions
from main_script_modularisation import initialize_camera, process_frame, load_model, solve_sudoku

def main():
    cap = initialize_camera()
    if cap is None:
        return

    model = load_model()
    if model is None:
        return

    frame_rate = 30
    prev = 0
    seen = dict()

    try:
        while True:
            time_elapsed = t.time() - prev
            success, img = cap.read()
            if not success:
                logging.warning("Failed to capture frame")
                continue

            if time_elapsed > 1. / frame_rate:
                prev = t.time()
                
                img_result, squares_guesses, squares_processed, corners = process_frame(img, model)
                if squares_guesses is None:
                    cv2.imshow('window', img_result)
                    continue

                if squares_guesses in seen:
                    if seen[squares_guesses] is False:
                        continue
                    solved_puzzle, time = seen[squares_guesses]
                else:
                    solution = solve_sudoku(squares_guesses)
                    if solution:
                        solved_puzzle, time = solution
                        seen[squares_guesses] = solution
                    else:
                        seen[squares_guesses] = False
                        continue

                try:
                    warped = CV_process.warp_image(corners, img)[0]  # Get the warped image
                    CV_process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                    img_result = CV_process.unwarp_image(warped, img_result, corners, time)
                except Exception as e:
                    logging.error(f"Error drawing solution: {e}")

                cv2.imshow('window', img_result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    main()