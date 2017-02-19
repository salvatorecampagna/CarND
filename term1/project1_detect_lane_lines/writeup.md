#**Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following: 
* Make a pipeline that finds lane lines on the road 
* Reflect on your work in a written report 


[//]: # (Image References)

[greyscale_image]: ./writeup_images/grayscale_image.jpg "Grayscale"
[blur_image]: ./writeup_images/blur_image.jpg "Gaussian smoothing"
[canny_image]: ./writeup_images/canny_image.jpg "Canny Edge Detection"
[masked_image]: ./writeup_images/masked_image.jpg "Region of Interest"
[line_image]: ./writeup_iamges/line_image.jpg "Hough Transform"
[weighted_image]: ./writeup_images/weighted_image.jpg "Final image"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 
First, I converted the images to grayscale. 
![alt text][grayscale_image]
Then I used Gaussian smoohing to reduce noise in the images. 
![alt text][blur_image]
The third step is about using the Canny image detection algorithm to find 'object' boundaries in the images. 
![alt text][canny_image]
In the fourth step I used a polyline to isolate the region of interest (trapezoid). I had to do some fine tuning to find a good compromise here. 
![alt text][masked_image]
The fourth step uses the Hough Trasform to identify the lines in the Region of Interest. 
![alt text][line_image]
In the last step I stacked the image with lines on top of the original image. 
![alt text][weighted_image]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function for doing the following:
* Filter out horizontal lines 
* Separate left and right lines based on the slope 
* Average all the left lines to a single left line 
* Average all the right lines to a single right line 
* Extend both the left and right line so to have a line going from bottom to about half of the image 

###2. Identify potential shortcomings with your current pipeline

While evaluating the result of my algorithm I identified the following issues: 
* Handling dashed lines: I had to do some fine tuning to get something acceptable 
* Handling light changes in the image and color changes in the road (issue visible by looking at the third video) 
* Hadling slope changes and curves in the road 

###3. Suggest possible improvements to your pipeline

A possible improvement would be to process the image in order to have better robustness agains light changes in the image, for instance increasing the contrast. Another potential improvement could be to process the image in such a way to isolate and extract yellow and white lines. 