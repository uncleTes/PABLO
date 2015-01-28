import numpy as np
from math import *
import sys
import os
import glob
import argparse
import cv2
import cv
import random

# OpenCV docs link:
# http://docs.opencv.org/master/modules/refman.html

# Global variables
marker_mask = None
img = None
prev_pt = (-1, -1)

# Mouse callback
def on_mouse(event, x, y, flags, param):
	# Call global variables used
	global marker_mask, img, prev_pt
	# Get rows and columns (in pixels) of the image passed to
	rows, cols = img.shape[:2]
	# Yeah, if you are not inside the image, don't do nothing
    	if(x < 0 or x >= cols or y < 0 or y >= rows):
    		return
	# If the left button of the mouse is released, reset "prev_pt"
    	if(event == cv2.EVENT_LBUTTONUP or 
	   not (flags and cv2.EVENT_FLAG_LBUTTON)):
    		prev_pt = (-1, -1)
	# When clicking the left button of the mouse, "prev_pt" become the 
	# current position
    	elif(event == cv2.EVENT_LBUTTONDOWN):
    		prev_pt = (x, y)
	# Move the mouse with the flag "EVENT_FLAG_LBUTTON", will draw a line
	# on "img" and on "marker_mask"
    	elif(event == cv2.EVENT_MOUSEMOVE and 
	     (flags and cv2.EVENT_FLAG_LBUTTON)):
    		pt = (x, y)
    		if(prev_pt[0] < 0):
    			prev_pt = pt
    		cv2.line(marker_mask,       # image where draw a line
			 prev_pt,           # first point of the line
			 pt,                # second point of the line
			 cv.ScalarAll(255), # line color
			 5,                 # line thickness
			 8,                 # line type 
			 0)                 # shift of the line (number of fractionalbits)
    		cv2.line(img, 
			 prev_pt, 
			 pt, 
			 cv.ScalarAll(255), 
			 5, 
			 8, 
			 0)
    		prev_pt = pt
    		cv2.imshow("image", 
			   img)
	
# Main function
def main():
	# Call global variables used
	global marker_mask, img
	# Check if the script is called with the right number of arguments
	if len(sys.argv) != 2:
		print("Usage: water_shed <image_name>")
		sys.exit(1)
	# Get the name of the image passed (absolute path)
	file_name = sys.argv[1]
	# Load the image. Passing "1" as flag, means that the image will
	# be loaded with 3-channel colors
	img0 = cv2.imread(file_name, # name of the file to be loaded
			  1)         # flag specifying the color type of the image
	# If the image cannot be read, advise and exit
	if img0 is None:
		print("Could not open image " + str(file_name))
		sys.exit(1)
	# Get height and width of the image passed to
	h, w  = img0.shape[:2]
	# Create a window with  name "image" and with fixed dimensions
	# (that's why "1" is passed as flag)
	cv2.namedWindow("image", # name of the window in the window caption
			1)       # flag for the window creation (normal, auto-size...)
	# Copy the image passed to
	img = img0.copy()
	# Transform image from BGR scale to GRAY scale...the function
	# returns the transformed image
	marker_mask = cv2.cvtColor(img,                # source image
				   cv2.COLOR_BGR2GRAY) # color space conversion code
	# Get a 3-channel GRAY image
	img_gray = cv2.cvtColor(marker_mask, 
				cv2.COLOR_GRAY2BGR)
	# Set "marker_mask" to all zeros
	marker_mask = np.zeros((h, w),   # matrix size 
			       np.uint8) # matrix type
	# Show "img" in the previously created window
	cv2.imshow("image", # name of the window
		   img)     # image to be displayed
	# Set "on_mouse" callback, with optional "param" seto to 0
	cv2.setMouseCallback("image",  # window's name
			     on_mouse, # mouse callback
			     0)        # optional parameters
	# Looping infinitely (pressing "esc" will cause exit the program)
	while True:
		# Delay setted to 0
		c = cv2.waitKey(0)
		# That's the famous "esc" pressure
		if c == 27:
			break
		# Given a string of length one, "ord(...)" return an integer
		# representing the Unicode code
		if c == ord('r'):
			# Resetting "marker_mask" to all zeros
			marker_mask = np.zeros((h, w), 
					       np.uint8)
			# Recopy of the image passed to 
			img = img0.copy()
			# Show the new copy of the image passed to
			cv2.imshow("image", 
				   img)
		# That's the cool part
		if c == ord('w') or c == ord(' '):
           		comp_count = 0
			# Find contours...the function returns a vector of 
			# point and a vector of vector composed by 4 elements
			# each
           		contours, hierarchy = cv2.findContours(marker_mask,             # input 8 bit single channel image
							       cv2.RETR_CCOMP,          # contour's retrieval mode
							       cv2.CHAIN_APPROX_NONE) # contours' approximation method

           		if len(contours) == 0:
           			continue
			# Create a "markers" matrix of int32, with elements 0
           		markers = np.zeros((h, w),
					   np.int32)
           		idx = 0
			# Draw contours
			while idx >= 0:
				cv2.drawContours(markers,                      # destination image
						 contours,                     # contours' vector
						 idx,                          # index of the contour to be drawn (-1 for all)
						 cv.ScalarAll(comp_count + 1), # color of the contour
						 -1,                           # thickness of the line. If negative, the interiors are drawn
						 8,                            # type of the line connectivity
						 hierarchy,                    # optional information needed to draw only some contours
						 np.iinfo(np.int16).max)       # maximum level to draw contours
				# Previously said that "hierarchy" is a vector
				# of vector of four elements each. Numpy stores
				# it as a vector (with one components) of 
				# vector of vectors of four elements. That's
				# why there are three indeces
				idx = hierarchy[0][idx][0]
				comp_count += 1

           		if comp_count == 0:
           			continue
			# Create some random colors
           		color_tab = []
           		for i in xrange(0, comp_count):
           			b = random.randint(0, 255)
           			g = random.randint(0, 255)
           			r = random.randint(0, 255)
				color_tab.append((b, g, r))
			# Watershed is an algorithm in image processing for 
			# isolating objects in the image from the background
           		cv2.watershed(img0,    # input 8-bit 3-channel image
				      markers) # input/output 32-bit single-channel image (map) of markers. 
					       # It should have the same size of the input image
			# Low-level method to instantiate an array (multi-
			# dimensional also) 
			wshed = np.ndarray((h, w),         # array size
					   dtype='3uint8') # array type

			# Give a value to the elements of "wshed"
			for i in xrange(0, h):
				for j in xrange(0, w):
					index = int(markers[i][j])
					if index == -1 :
						wshed[i][j] = (255, 255, 255)
					elif (index <= 0 or index > comp_count):
						wshed[i][j] = (0, 0, 0)
					else:
						# passing the tuples as numpy array
						wshed[i][j] = np.asarray(color_tab[index - 1], 
									 dtype='uint8')

			wshed = ((wshed * 0.5) + (img_gray * 0.5)).astype('uint8')
			cv2.imshow("watershed transform", wshed)
		if c == ord('o'):
			prova = cv2.cvtColor(wshed, cv2.COLOR_BGR2GRAY)
			cv2.imshow("prova", prova)

if __name__ == "__main__":
	main()
