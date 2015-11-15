__author__ = 'me'
import cv2

import math
import numpy as np
import sys

def find_edges(img):
    image = img.copy()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 11, 17, 17)

    dst = cv2.Canny(grey, 50, 200)
    return dst


def find_circles(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    return circles


def find_lines(img):
    dst = cv2.Canny(img, 50, 200)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(cdst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
    return lines

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def find_contours(img):
    img = img.copy();
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    done = cv2.Canny(img, 100,200,3)
    _, contours0, hierarchy = cv2.findContours( done, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

    return contours


def find_closed_shapes(img):
    img = img.copy()
    tmp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    threshold = cv2.Canny(tmp, 50, 200, 5)


    (_,contours, _) = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    closed = []
    for this_contour in contours:
        arc_length = cv2.arcLength(this_contour,True) * 0.02
        if abs(cv2.contourArea(this_contour)) < 50 or not cv2.isContourConvex(this_contour):
            pass

        approx = cv2.approxPolyDP(this_contour,arc_length, True)

        if approx.size < 10:
            closed.append(approx)

    return closed



def find_good_features(img):
    img = img.copy()
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(grey,25,0.01,10)

    return corners
