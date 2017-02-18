# Importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import math
import os

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
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, filter_horizontal_lines=True):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # Filter out horizontal lines
    if filter_horizontal_lines:
    	non_horizontal_lines = list()
    	for line in lines:
    		for x1, y1, x2, y2 in line:
    			if math.fabs((y2 - y1) / (x2 - x1)) >= 0.2:
    				non_horizontal_lines.append(line)
    else:
    	non_horizontal_lines = lines

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, non_horizontal_lines)
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
	# Pipeline step 1
	# Convert the image to greyscale
	gray_image = grayscale(image)
	# Pipeline step 2
	# Apply Gaussian smoothing / blurring
	blur_image = gaussian_blur(gray_image, kernel_size=5)
	# Pipeline step 3
	# Apply Canny edge detection
	canny_image = canny(gray_image, low_threshold=1, high_threshold=150)
	# Pipeline step 4
	# Select region of interest
	ysize = image.shape[0]
	xsize = image.shape[1]
	left_bottom = [0, ysize]
	right_bottom = [xsize, ysize]
	left_top = [xsize - 0.55 * xsize, ysize - 0.40 * ysize]
	right_top = [xsize - 0.45 * xsize, ysize - 0.40 * ysize]
	vertices = np.array([left_bottom, left_top, right_top, right_bottom])
	masked_image = region_of_interest(canny_image, np.int32([vertices]))
	# Piepline step 5
	# Apply Hough Trasform
	# Define Hough transform parameters
	rho = 1
	theta = np.pi/180
	threshold = 3
	min_line_length = 40
	max_line_gap = 5
	line_image = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
	# Pipeline step 6
	# Put the original image and the lines one on top of the other
	weighted_image = weighted_img(line_image, image)
	return weighted_image

if __name__ == '__main__':
	test_images = get_test_images()
	for test_image in test_images:
		print("Processing image: {0}".format(test_image))
		image = mpimg.imread("test_images/" + test_image)
		result_image = process_image(image)
		result_filename = "output_images/" + test_image
		print("Saving image: {0}".format(result_filename))
		mpimg.imsave(result_filename, result_image)

	white_output = 'white.mp4'
	print("Processing video: {0}".format(white_output))
	clip1 = VideoFileClip("solidWhiteRight.mp4")
	white_clip = clip1.fl_image(process_image)
	white_clip.write_videofile(white_output, audio=False)
	print("{0} video done".format(white_output))
	yellow_output = 'yellow.mp4'
	print("Processing video: {0}".format(yellow_output))
	clip2 = VideoFileClip('solidYellowLeft.mp4')
	yellow_clip = clip2.fl_image(process_image)
	yellow_clip.write_videofile(yellow_output, audio=False)
	print("{0} video done".format(yellow_output))