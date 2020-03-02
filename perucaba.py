import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image 'perucaba'
img = cv2.imread('perucaba.jpg', cv2.IMREAD_COLOR)
# Making a copy of the original image
img_gray = img.copy()
# Converting the original image from (Blue, Green, Red) to (Red, Green, Blue)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Converting the copied image to grayscale
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
# Setting the size of the plot image to 16x9 proportion
plt.rcParams['figure.figsize'] = (16, 9)

# Showing the original and the grayscale images
plt.imshow(img)
plt.show()
plt.imshow(img_gray, cmap='gray')
plt.show()

# Making the threshold image using the 'thershold' function
# It takes 4 parameters, which are:
#       The grayscale image which will be analized
#       The threshold limit for the pixel value being read
#       The value that will be assigned to the pixel that is greater than the limit
#       The method of analisis used by the function
#           In this case, using THRESH_BINARY_INV, if the value of the pixel
#           is greater than the treshold limit, it will be the inverse of the assigned
#           value, so the pixel, instead of receaving 255, will receive 0, in other words,
#           instead of being represented as white, it will be represented as black.
# It returns two items, the threshold value and the threshold image
ret, thresh = cv2.threshold(img_gray,11,255,cv2.THRESH_BINARY_INV)
# Printing the threshold image
plt.imshow(thresh, cmap='gray')
plt.show()

# Creating the list with all the contours found in the image
# The 'findContours' function has three arguments:
#       The threshold image
#       The contour retrieval mode (in this case RETR_TREE which
#           retrieves all of the contours and reconstructs a full hierarchy of nested contours.)
#       The method of contour approximation (in this case CHAIN_APPROX_SIMPLE
#           which compresses horizontal, vertical, and diagonal segments and leaves only their end points.)
# It returns a python list with all contours in the image. Each individual contour
# is an nparray of (x, y) coordinates of boundary points of the object
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Drawing the contours found in the original image
# The 'drawContours' function has 5 arguments:
#       The image in which the contours will be drawn
#       The list of contours
#       The index of the contours that will be drawn (using '-1' it prints all the contours)
#       The RGB code for the contour color
#       The thickness of the contour line
cv2.drawContours(img, contours, -1, (0,255,0), 3)
# Printing the image with the contours drawn
plt.imshow(img)
plt.show()
