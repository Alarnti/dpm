from skimage.feature import hog
from skimage import data, color, exposure, transform

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from skimage import io

from dpm import DPM

import cv2

import os

im = cv2.imread("side_col.jpg",0)

dpm = DPM()

adaclfs = dpm.init_part_filters()


im = cv2.resize(im,None,fx=0.5, fy=.5, interpolation = cv2.INTER_AREA)
dpm.process_image(im, 'side_col')

# result = dpm.new_test_train(adaclfs)

# print result

dpm.get_new_test(adaclfs)


# print '--cars--'
# print '128.jpg', dpm.process_image(cv2.imread('128.jpg',0),adaclfs)
# print '112.jpg', dpm.process_image(cv2.imread('112.jpg',0),adaclfs)
# print '118.jpg', dpm.process_image(cv2.imread('118.jpg',0),adaclfs)
# print '150.jpg', dpm.process_image(cv2.imread('150.jpg',0),adaclfs)

# print '--false-cars--'
# for el in os.listdir('negative_tests'):
# 	print el, dpm.process_image(cv2.imread('negative_tests/' + el,0),adaclfs)



