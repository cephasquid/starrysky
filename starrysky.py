__author__ = 'me'

import cv2
import math
import random
import numpy as np

def create_star_background(image,density):
    h,w = image.shape[:2]
    base_color = 20
    for i in range(0,h,4):
        for j in range(0,w,4):
            rand = random.random()
            if rand < density:
                intensity = np.random.normal(0,.5,100)
                intensity = intensity[random.randrange(0,99)] * 50
                cv2.circle(image,(j,i),1,(base_color + intensity,)*3,1)



def create_contour_stars(image, contours):
    for contour in contours:
        center, radius = cv2.minEnclosingCircle(contour)
        rad = 100 + int((math.log(radius)+1) * 25)
        r = random.randrange(-15,15)
        g = random.randrange(-15,15)
        b = random.randrange(-15,15)
        center = (int(center[0]), int(center[1]))
        cv2.circle(image,center,2,(rad+ r -10, rad+g+-10,rad+b+-1), 0)
        cv2.circle(image,center,1, (rad + r, rad+g, rad+b), -1)


def create_good_feature_stars(image, features):
    for feature in features:
        x,y = feature.ravel()

        cv2.circle(image,(x,y),3,(0,)*3,-1)


