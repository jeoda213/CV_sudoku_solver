import operator
import cv2
import numpy as np


def preprocess(img):
    """
    Preprocesses the input image to enhance grid lines and text, preparing it for further analysis such as
    contour detection or character recognition in a Sudoku puzzle.

    The preprocessing steps include converting the image to grayscale, applying Gaussian blur to reduce noise,
    using adaptive thresholding for binarization, inverting the binary image, performing morphological operations
    to clean up noise, and dilating the image to enhance the grid lines and text.

    Args:
    img (numpy.ndarray): The input image in BGR format.

    Returns:
    numpy.ndarray: The preprocessed binary image ready for further analysis.

    Detailed Explanation:

    1. Convert the image to grayscale:
       The function starts by converting the input image from BGR color space to grayscale.
       This reduces the complexity of the image and prepares it for further processing.
       img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    2. Apply Gaussian blur:
       Gaussian blur is applied to the grayscale image to smooth it and reduce noise.
       This helps in making the thresholding step more effective by reducing variations in pixel intensities.
       blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    3. Apply adaptive thresholding:
       Adaptive thresholding is used to convert the blurred image to a binary image.
       This method is adaptive to local changes in illumination, making it suitable for images with varying lighting conditions.
       thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    4. Invert the binary image:
       The binary image is inverted so that the grid lines and text become white and the background becomes black.
       This inversion is necessary because some morphological operations work better with white foregrounds.
       inverted = cv2.bitwise_not(thresh, 0)

    5. Create a rectangular structuring element:
       A rectangular kernel is created to be used in morphological operations.
       The size of the kernel is (2, 2), which is suitable for removing small noise elements.
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    6. Apply morphological opening:
       Morphological opening is applied to the inverted image using the rectangular kernel.
       This operation helps remove small noise elements like random dots, which are not part of the grid or text.
       morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    7. Dilate the image:
       Dilation is applied to the morphologically opened image to increase the size of the grid lines and text.
       This step enhances the borders and makes the grid lines more pronounced.
       result = cv2.dilate(morph, kernel, iterations=1)

    8. Return the processed image:
       Finally, the preprocessed image is returned. This image is ready for further analysis, such as contour detection
       or character recognition in a Sudoku puzzle.
       return result
    """
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # Apply adaptive thresholding to obtain a binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the binary image to make the grid lines and text white
    inverted = cv2.bitwise_not(thresh, 0)

    # Create a rectangular structuring element (kernel) for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Apply morphological opening to remove small noise elements
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    # Apply dilation to enhance the grid lines and text
    result = cv2.dilate(morph, kernel, iterations=1)

    return result

def find_extreme_corners(polygon: np.ndarray, limit_fn, compare_fn):
    """
    Finds the extreme corner of a polygon based on specified functions for comparison and limiting.

    This function is used to determine a specific corner (e.g., the top-left, top-right, bottom-left, or bottom-right)
    of a polygon by comparing the coordinates of its points. It employs two functions: one for establishing limits
    (such as finding the minimum or maximum) and another for comparing points (such as addition or subtraction of coordinates).

    Args:
    polygon (numpy.ndarray): The polygon represented as an array of points. Each point is typically in the format [[x, y]].
    limit_fn (function): A function to determine the limiting criterion, such as min or max, applied to the comparison results.
                         Example: min, max.
    compare_fn (function): A function to compare points, typically involving operations on the coordinates.
                           Example: np.add, np.subtract.

    Returns:
    tuple: Coordinates of the extreme corner in the form (x, y).

    Detailed Steps:

    1. Calculate Comparison Values:
       The function starts by iterating over each point in the polygon. For each point, it applies the compare_fn
       to the x and y coordinates. This produces a list of comparison values.
       For example, if compare_fn is np.add, it adds the x and y coordinates of each point.

       comparison_values = [compare_fn(pt[0][0], pt[0][1]) for pt in polygon]

    2. Find the Limit Index:
       The limit_fn is then used to find the index of the point that corresponds to the limit (e.g., minimum or maximum)
       value in the list of comparison values. The key parameter of limit_fn is set to operator.itemgetter(1),
       which ensures the comparison is based on the values rather than the indices.

       section, _ = limit_fn(enumerate(comparison_values), key=operator.itemgetter(1))

    3. Return the Extreme Corner Coordinates:
       Using the index obtained in the previous step, the function retrieves the corresponding point from the original polygon array
       and returns its coordinates.

       return polygon[section][0][0], polygon[section][0][1]

    Example Usage:
    - To find the top-left corner of a polygon, you might use np.add as the compare_fn and min as the limit_fn.
    - To find the bottom-right corner of a polygon, you might use np.add as the compare_fn and max as the limit_fn.

    Example:
    ```python
    import numpy as np
    polygon = np.array([[[1, 2]], [[4, 6]], [[7, 3]], [[2, 8]]])
    top_left_corner = find_extreme_corners(polygon, min, np.add)
    print(top_left_corner)  # Output will be the coordinates of the top-left corner based on the compare_fn and limit_fn
    ```

    Note:
    - The choice of compare_fn and limit_fn will significantly affect the resulting extreme corner.
    - This function assumes that the input polygon is a numpy array where each point is formatted as [[x, y]].

    """
    # Use the limit_fn to find the index of the corner based on the compare_fn
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    # Return the coordinates of the extreme corner
    return polygon[section][0][0], polygon[section][0][1]



def draw_extreme_corners(pts, original):
    """
    Draws a circle at the given points on the original image.

    Args:
    pts (tuple): Coordinates of the point where the circle is to be drawn.
    original (numpy.ndarray): The image on which to draw the circle.
    """
    # Draw a filled circle at the given points with a radius of 7 and color (0, 255, 0)
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)


def clean_helper(img):
    # Check if the image is almost entirely black
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.98:
        return np.zeros_like(img), False

    # Check if there is very little white in the region around the center
    height, width = img.shape
    mid = width // 2
    if np.isclose(img[:, int(mid - width * 0.3):int(mid + width * 0.3)], 0).sum() / (2 * width * 0.3 * height) >= 0.95:
        return np.zeros_like(img), False

    # Find the largest contour and center it in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(img), False
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Check if the contour is too small (likely noise)
    min_area = 0.01 * img.shape[0] * img.shape[1]  # 1% of image area
    if cv2.contourArea(contours[0]) < min_area:
        return np.zeros_like(img), False

    # Check aspect ratio
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.2 or aspect_ratio > 5:
        return np.zeros_like(img), False

    # Create a new image with the contour centered
    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    return new_img, True


def grid_line_helper(img, shape_location, length=10):
    """
    Enhances grid lines in the given image by applying morphological operations.

    Args:
    img (numpy.ndarray): The input binary image containing grid lines.
    shape_location (int): The orientation of the lines to process. 
                          Use 0 for vertical lines and 1 for horizontal lines.
    length (int, optional): The number of grid lines expected in the image. 
                            This determines the size of the structuring element. 
                            Default is 10.

    Returns:
    numpy.ndarray: The processed image with enhanced grid lines.
    
    Detailed Explanation:
    
    1. Clone the input image:
       The function starts by creating a copy of the input image to avoid altering the original image.
       This is done using the `copy` method:
       clone = img.copy()
    
    2. Determine the orientation and size of the grid lines:
       The shape_location parameter indicates the orientation of the lines to be enhanced.
       - If shape_location is 0, the function will process vertical lines.
       - If shape_location is 1, the function will process horizontal lines.
       
       The variable row_or_col is set to the width (number of columns) if processing vertical lines, 
       or to the height (number of rows) if processing horizontal lines:
       row_or_col = clone.shape[shape_location]
    
    3. Calculate the distance between grid lines:
       The size variable is determined by dividing the dimension (width or height) by the number of 
       expected grid lines (length):
       size = row_or_col // length
    
    4. Create a structuring element (kernel):
       A structuring element is a matrix used for morphological operations. The size and shape of this 
       element determine the operation's effect on the image.
       
       - For vertical lines (shape_location == 0), the kernel is a vertical rectangle:
         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
         This creates a kernel with a width of 1 pixel and a height of 'size' pixels, suitable for processing vertical lines.
       
       - For horizontal lines (shape_location == 1), the kernel is a horizontal rectangle:
         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
         This creates a kernel with a width of 'size' pixels and a height of 1 pixel, suitable for processing horizontal lines.
       
       The cv2.getStructuringElement function generates the structuring element based on the specified shape 
       and size. The cv2.MORPH_RECT parameter indicates that the kernel is rectangular.
       
    5. Apply morphological operations:
       Morphological operations are used to process binary images, primarily for removing noise and 
       enhancing structures.
       
       - Erosion:
         The clone image is eroded using the created kernel:
         clone = cv2.erode(clone, kernel)
         Erosion shrinks white regions and can help remove small noise points.
       
       - Dilation:
         The eroded image is then dilated using the same kernel:
         clone = cv2.dilate(clone, kernel)
         Dilation enlarges white regions and can help connect broken grid lines.
    
    6. Return the processed image:
       Finally, the function returns the processed image with enhanced grid lines:
       return clone
    """
    clone = img.copy()
    # if its horizontal lines then it is shape_location 1, for vertical it is 0
    row_or_col = clone.shape[shape_location]
    # find out the distance the lines are placed
    size = row_or_col // length

    # find out an appropriate kernel
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    # erode and dilate the lines
    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone

def draw_lines(img, lines):
    """
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    Draws lines on the given image based on the provided lines array.

    This function takes an image and an array of lines described by the Hough Line Transform parameters (rho and theta).
    It draws each line on a copy of the image.

    Args:
    img (numpy.ndarray): The input image on which lines will be drawn.
    lines (numpy.ndarray): Array of lines, where each line is represented by (rho, theta).
                           - rho: Distance from the origin to the line.
                           - theta: Angle from the x-axis to the line.

    Returns:
    numpy.ndarray: The image with lines drawn on it.

    Detailed Explanation:

    1. Clone the input image:
       The function starts by creating a copy of the input image to avoid altering the original image.
       This is done using the `copy` method:
       clone = img.copy()

    2. Squeeze the lines array:
       The `np.squeeze` function is used to remove single-dimensional entries from the shape of the lines array.
       This ensures that the lines array is in the correct format for processing:
       lines = np.squeeze(lines)

    3. Iterate over each line:
       The function iterates over each line in the lines array. Each line is represented by the parameters (rho, theta).

    4. Calculate the endpoints of each line:
       Using the rho and theta values, the function calculates the endpoints of each line.
       - `a = np.cos(theta)`: Calculates the cosine of the angle theta.
       - `b = np.sin(theta)`: Calculates the sine of the angle theta.
       - `x0 = a * rho`: Calculates the x-coordinate of the point closest to the origin.
       - `y0 = b * rho`: Calculates the y-coordinate of the point closest to the origin.
       - The line is extended in both directions from this point by multiplying by a large value (1000) and adjusting for the angle.

       The endpoints are calculated as:
       - `x1 = int(x0 + 1000 * (-b))`
       - `y1 = int(y0 + 1000 * a)`
       - `x2 = int(x0 - 1000 * (-b))`
       - `y2 = int(y0 - 1000 * a)`

    5. Draw the line on the image:
       The `cv2.line` function is used to draw the line on the image:
       cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)
       - `clone`: The image on which the line will be drawn.
       - `(x1, y1)`: The starting point of the line.
       - `(x2, y2)`: The ending point of the line.
       - `(255, 255, 255)`: The color of the line (white in this case).
       - `4`: The thickness of the line.

    6. Return the processed image:
       Finally, the function returns the processed image with the lines drawn on it:
       return clone
    """
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        # Calculate the cosine and sine of the angle theta
        a = np.cos(theta)
        b = np.sin(theta)
        # Calculate the x and y coordinates of the point closest to the origin
        x0 = a * rho
        y0 = b * rho
        # Calculate the endpoints of the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        # Draw the line on the image
        cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)

    return clone