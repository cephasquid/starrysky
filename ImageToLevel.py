__author__ = 'me'


import cv2
import sys
import numpy as np
import math

import findbits
import starrysky



try:
    fn = sys.argv[1]
except:
    fn = "test.jpg"


src = cv2.imread(fn)
src = cv2.resize(src,(800,600))
h,w = src.shape[:2]
new_image = findbits.find_edges(src)
squares = findbits.find_squares(src)
blank_image = np.ones((h, w,3), np.uint8)

starrysky.create_star_background(blank_image,.2)
contours = findbits.find_contours(src)

#cv2.drawContours(blank_image,contours,-1, (0,255,0),3)
starrysky.create_contour_stars(blank_image,contours)

shapes = findbits.find_closed_shapes(src)

tofollow = findbits.find_good_features(src)

starrysky.create_good_feature_stars(blank_image,tofollow)

cv2.imshow("starry sky", blank_image)
cv2.waitKey(0)


