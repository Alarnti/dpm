from skimage.feature import hog
from skimage import data, color, exposure, transform
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io
from dpm import DPM
import cv2
import os

# Our test image
im = cv2.imread("side_col.jpg",0)

dpm = DPM()
adaclfs = dpm.init_part_filters()

im = cv2.resize(im,None,fx=0.5, fy=.5, interpolation = cv2.INTER_AREA)
dpm.process_image(im, 'side_col')

dpm.get_new_test(adaclfs)
