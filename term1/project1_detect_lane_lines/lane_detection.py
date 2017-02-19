# Importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import math
import os

global y_bottom
global y_top

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def line_slope(line):
    """
    Given a line passing through two points computes the slope of the line
    """
    x1, y1, x2, y2 = line
    return (y2 - y1) / (x2 - x1)

def line_intercept(line):
    """
    Given a line passing through two points computes the intercept of the line
    """
    x1, y1, x2, y2 = line
    m = line_slope(line)
    return (y1 - m * x1)

def filter_out_horizontal_lines(lines, min_slope=0.2):
    """
    Remove lines whose slope is less than 'min_slope'
    """
    _lines = list()
    for line in lines:
        for l in line:
            if math.fabs(line_slope(l)) > min_slope:
                _lines.append(line)
    return _lines

def filter_left_right_lines(lines):
    left_lines = list()
    right_lines = list()
    for line in lines:
        for l in line:
            if line_slope(l) > 0:
                right_lines.append(line)
            else:
                left_lines.append(line)
    return left_lines, right_lines

def extract_line(slope, intercept):
    """
    Given the slope and intercept of a line
    compute two points at (x_bottom, y_bottom), (x_top, y_top)
    """
    x_bottom = (y_bottom - intercept) / slope
    x_top = (y_top - intercept) / slope
    return [int(x_bottom), int(y_bottom), int(x_top), int(y_top)]

def line_average(lines):
    """
    Given a set of lines computes a single line with coordinates:
    x1 = average(all x1's)
    y1 = average(all y1's)
    x2 = average(all x2's)
    y2 = average(all y2's)
    """
    if not lines:
        return []

    sum_x1 = sum_y1 = sum_x2 = sum_y2 = 0
    n_lines = len(lines)

    for line in lines:
        for x1, y1, x2, y2 in line:
            sum_x1 += x1
            sum_y1 += y1
            sum_x2 += x2
            sum_y2 += y2
    return [int(sum_x1/n_lines), int(sum_y1/n_lines), int(sum_x2/n_lines), int(sum_y2/n_lines)]

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global y_top
    global y_bottom

    # Filter out horizontal lines
    lines = filter_out_horizontal_lines(lines)

    # Separate 'left' lines from 'right' lines
    left_lines, right_lines = filter_left_right_lines(lines)

    # Average all left lines in a single left line and all right lines in a single right line
    left_line = line_average(left_lines)
    right_line = line_average(right_lines)

    # Find the slope and intercept of both the left and right line
    if left_line:
        left_line_slope = line_slope(left_line)
        left_line_intercept = line_intercept(left_line)
    if right_line:
        right_line_slope= line_slope(right_line)
        right_line_intercept = line_intercept(right_line)

    # Compute the left and right line going from bottom to top of the region of interest
    if left_line:
        left_line = extract_line(left_line_slope, left_line_intercept)

    if right_line:
        right_line = extract_line(right_line_slope, right_line_intercept)

    lines = list()
    if right_line:
        lines.append([right_line])
    if left_line:
        lines.append([left_line])
    # Draw the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=10)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_test_images(directory='test_images/'):
	return os.listdir(directory)

def process_image(image):
    global y_top
    global y_bottom
    # Pipeline step 1
    # Convert the image to greyscale
    gray_image = grayscale(image)
    # Pipeline step 2
    # Apply Gaussian smoothing / blurring
    blur_image = gaussian_blur(gray_image, kernel_size=5)
    # Pipeline step 3
    # Apply Canny edge detection
    canny_image = canny(blur_image, low_threshold=50, high_threshold=150)
    # Pipeline step 4
    # Select region of interest
    ysize = image.shape[0]
    xsize = image.shape[1]
    y_bottom = ysize
    y_top = ysize - 0.38 * ysize
    left_bottom = [xsize - 0.89 * xsize, ysize]
    right_bottom = [xsize - 0.06 * xsize, ysize]
    left_top = [xsize - 0.55 * xsize, y_top]
    right_top = [xsize - 0.45 * xsize, y_top]
    vertices = np.array([left_bottom, left_top, right_top, right_bottom])
    masked_image = region_of_interest(canny_image, np.int32([vertices]))
    # Piepline step 5
    # Apply Hough Trasform
    # Define Hough transform parameters
    rho = 2
    theta = np.pi/180
    threshold = 20
    min_line_length = 35
    max_line_gap = 20
    line_image = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
    # Pipeline step 6
    # Put the original image and the lines one on top of the other
    weighted_image = weighted_img(line_image, image)
    return weighted_image

def image_lanes(test_image):
    print("Processing image: {0}".format(test_image))
    image = mpimg.imread("test_images/" + test_image)
    result_image = process_image(image)
    result_filename = "output_images/" + test_image
    print("Saving image: {0}".format(result_filename))
    mpimg.imsave(result_filename, result_image)

def video_lanes(input_file, output_file):
    print("Processing video: {0}".format(input_file))
    clip1 = VideoFileClip(input_file)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_file, audio=False)
    print("{0} video done".format(output_file))

if __name__ == '__main__':
    test_images = get_test_images()
    for test_image in test_images:
        image_lanes(test_image)

    video_lanes("solidWhiteRight.mp4", "white.mp4")
    video_lanes("solidYellowLeft.mp4", "yellow.mp4")
    video_lanes("challenge.mp4", "extra.mp4")