import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

import CV_process_support as process_helpers
# from Machine_Learning_Pipeline.constant_variables import resize_length as img_size

def create_grid_mask(vertical, horizontal):
    """
    Creates a grid mask by combining vertical and horizontal lines, then processing the result to isolate the grid structure.

    This function takes binary images of vertical and horizontal lines, combines them to form a grid, and then processes
    the combined image to enhance the grid structure. It uses adaptive thresholding and dilation to cover more area,
    followed by Hough line transformation to detect lines. The final output is a mask that isolates the grid lines, making
    it easier to extract the numbers within the grid.

    Args:
    vertical (numpy.ndarray): Binary image containing detected vertical lines.
    horizontal (numpy.ndarray): Binary image containing detected horizontal lines.

    Returns:
    numpy.ndarray: A binary mask image where the grid lines are black, and the background is white.
    """

    # combine the vertical and horizontal lines to make a grid
    grid = cv2.add(horizontal, vertical)
    # threshold and dilate the grid to cover more area
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # find the list of where the lines are, this is an array of (rho, theta in radians)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    lines = process_helpers.draw_lines(grid, pts)
    # extract the lines so only the numbers remain
    mask = cv2.bitwise_not(lines)
    return mask

def get_grid_lines(img, length=10):
    """
    Extracts vertical and horizontal grid lines from the input image.

    This function uses a helper function to identify and extract vertical and horizontal lines
    from a preprocessed binary image. It is particularly useful for detecting grid lines in
    images such as Sudoku puzzles.

    Args:
    img (numpy.ndarray): The preprocessed binary image from which grid lines are to be extracted.
    length (int, optional): The length parameter for the helper function that defines the minimum line length.
                            This parameter can be adjusted based on the size of the grid lines in the image.
                            Default is 10.

    Returns:
    tuple: A tuple containing two numpy.ndarrays:
           - vertical (numpy.ndarray): The binary image with only the vertical grid lines.
           - horizontal (numpy.ndarray): The binary image with only the horizontal grid lines.

    """
    horizontal = process_helpers.grid_line_helper(img, 1, length)
    vertical = process_helpers.grid_line_helper(img, 0, length)
    return vertical, horizontal


def find_contours(img, original):
    """
    Finds and identifies the largest quadrilateral contour in a binary image.

    This function processes a binary (thresholded) image to identify the largest contour with four corners,
    which is assumed to be a grid or a similar rectangular object. It sorts the contours by area and 
    selects the largest one that meets the criteria. The extreme corners of this contour are then found 
    and marked on the original image.

    Args:
    img (numpy.ndarray): The binary image (thresholded) in which to find contours.
    original (numpy.ndarray): The original image on which the detected contour and its corners will be drawn.

    Returns:
    list: A list of tuples representing the coordinates of the four corners of the identified quadrilateral
          contour in the order [top_left, top_right, bottom_right, bottom_left]. Returns an empty list if no
          valid contour is found.
    """
    # find contours on thresholded image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    # make sure this is the one we are looking for
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = cnt
            break

    if polygon is not None:
        # find its extreme corners
        top_left = process_helpers.find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = process_helpers.find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = process_helpers.find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = process_helpers.find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        # if its not a square, we don't want it
        if bot_right[1] - top_right[1] == 0:
            return []
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        # draw corresponding circles
        [process_helpers.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []


def warp_image(corners, original):
    """    
    Warps the original image to a top-down view of the region defined by the given corners.

    This function takes the coordinates of four corners defining a quadrilateral in the original image and
    warps the image such that the quadrilateral is transformed into a square. This is typically used to 
    obtain a bird's-eye view of a region of interest, such as a Sudoku grid.

    Args:
    corners (list of tuples): A list of four (x, y) coordinates representing the corners of the quadrilateral 
                              in the order [top_left, top_right, bottom_right, bottom_left].
    original (numpy.ndarray): The original image from which the region will be warped.
    
    Returns:
    tuple:
        numpy.ndarray: The warped image transformed into a square.
        numpy.ndarray: The transformation matrix used for the perspective warp.

    """
    # we will be warping these points
    corners = np.array(corners, dtype='float32')
    top_left, top_right, bot_right, bot_left = corners

    # find the best side width, since we will be warping into a square, height = length
    width = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))

    # create an array with shows top_left, top_right, bot_left, bot_right
    mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(original, matrix, (width, width)), matrix


def split_into_squares(warped_img):
    """
    Splits a warped image of a Sudoku puzzle into 81 individual squares.

    This function assumes that the warped image is a perfect square divided into a 9x9 grid. Each cell of the grid
    is extracted and returned as an individual image.

    Args:
    warped_img (numpy.ndarray): The warped image of the Sudoku puzzle.

    Returns:
    list of numpy.ndarray: A list of 81 images, each representing a square of the Sudoku grid.
    """
    squares = []

    width = warped_img.shape[0] // 9

    # find each square assuming they are of the same side
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return squares


def clean_squares(squares):
    """
    Cleans each square in the list by centering the largest contour and determining if it contains a number.

    This function processes each square image to center the largest contour, which helps in isolating the number
    in each cell. It returns a list of cleaned images and replaces squares without numbers with zero.

    Args:
    squares (list of numpy.ndarray): A list of images, each representing a square of the Sudoku grid.

    Returns:
    list: A list of cleaned images. Squares without numbers are replaced with zero.
    """
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = process_helpers.clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares

def resize_to_64x64(squares):
    """
    Resize images to 64x64 pixels.
    
    Args:
    squares (list or numpy.ndarray): A list of square images or a single square image.
    
    Returns:
    list or numpy.ndarray: Resized square images.
    """
    if isinstance(squares, list):
        resized_squares = []
        for square in squares:
            if isinstance(square, np.ndarray):
                # Resize the image to 64x64
                resized = cv2.resize(square, (64, 64), interpolation=cv2.INTER_CUBIC)
                resized_squares.append(resized)
            else:
                # If it's not an image (e.g., 0 for empty cell), keep it as is
                resized_squares.append(square)
        return resized_squares
    elif isinstance(squares, np.ndarray):
        # If it's a single image
        return cv2.resize(squares, (64, 64), interpolation=cv2.INTER_CUBIC)
    else:
        raise TypeError("Input must be a list of images or a single image.")

def recognize_digits(squares_processed, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    s = ""
    batch_size = 16  # Process images in batches for efficiency

    with torch.no_grad():  # Disable gradient calculation for inference
        for i in range(0, len(squares_processed), batch_size):
            batch = squares_processed[i:i+batch_size]
            
            # Prepare the batch
            processed_batch = []
            for square in batch:
                if isinstance(square, np.ndarray):
                    # Resize to 64x64 (assuming this is what your model expects)
                    img = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
                    # Normalize the image
                    img = img.astype(np.float32) / 255.0
                    # Add channel dimension
                    img = np.expand_dims(img, axis=0)
                    processed_batch.append(img)
                else:
                    # For non-image squares (empty cells), use a blank image
                    processed_batch.append(np.zeros((1, 64, 64), dtype=np.float32))

            # Convert to PyTorch tensor
            tensor_batch = torch.from_numpy(np.array(processed_batch)).to(device)
            
            # Get predictions
            outputs = model(tensor_batch)
            _, predicted = torch.max(outputs, 1)

            # Convert predictions to string
            for j, pred in enumerate(predicted):
                if isinstance(squares_processed[i+j], np.ndarray):
                    s += str(pred.item() + 1)  # +1 because model output is 0-8 for digits 1-9
                else:
                    s += "0"  # For empty cells

    return s

def draw_digits_on_warped(warped_img, solved_puzzle, squares_processed):
    """
    Draws the solved digits onto the warped image of the Sudoku grid.

    This function takes the solved Sudoku puzzle and draws the digits onto the corresponding squares in the warped image.
    Only the squares that were originally blank are updated with the solved digits.

    Args:
    warped_img (numpy.ndarray): The warped image of the Sudoku grid.
    solved_puzzle (str): A string representing the solved Sudoku puzzle.
    squares_processed (list): A list of cleaned images of the Sudoku grid squares.

    Returns:
    numpy.ndarray: The image with the solved digits drawn onto it.
    """
    width = warped_img.shape[0] // 9

    img_w_text = np.zeros_like(warped_img)

    # find each square assuming they are of the same side
    index = 0
    for j in range(9):
        for i in range(9):
            if type(squares_processed[index]) == int:
                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_w_text


def unwarp_image(img_src, img_dest, pts, time):
    """
    Unwarps a source image onto a destination image based on specified points.

    This function performs perspective transformation (warping) of the source image `img_src` onto the
    destination image `img_dest`, using the perspective transformation matrix derived from the four
    specified points `pts`. The transformation rectifies the perspective of the source image as seen
    from a particular viewpoint, aligning it with the destination image.

    Args:
    img_src (numpy.ndarray): The source image to be unwarped.
    img_dest (numpy.ndarray): The destination image onto which the source image will be unwarped.
    pts (list of tuples): Four points specifying the perspective in the source image,
                          provided as a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    time (str): Text to be displayed on the unwarped image, typically indicating the time taken for
                solving or processing.

    Returns:
    numpy.ndarray: The destination image (`img_dest`) with the source image (`img_src`) unwarped and
                   superimposed. The text `time` is also overlaid on the image.

    """
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img
