from skimage.feature import hog
from skimage import data, color, exposure, transform

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from skimage import io

from dpm import DPM

import cv2

im = cv2.imread("car.pgm",0)

dpm = DPM()

adaclfs = dpm.init_part_filters()

result = dpm.get_new_test(adaclfs)

print result

result = dpm.new_test_train(adaclfs)

print result

print dpm.process_image(cv2.imread('128.jpg',0),adaclfs)



