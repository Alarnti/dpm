from skimage.feature import hog
from skimage import data, color, exposure

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from skimage import io

#from Car_Detection place
def get_train():
	train_images_pos = []
	train_images_neg = []

	#GETTING TRAINIG IMAGES
	#this is bad, rewrite
	#-->

	less_10 = True
	additional_symbol = '0'
	for count in range(0,50):
		if less_10:
			if count >= 10:
				additional_symbol = ''
				less_10 = False

		input_image = io.imread('cars/images/trainpos00' + additional_symbol + str(count) + '.jpg')
		train_images_pos.append(color.rgb2gray(input_image))

	additional_symbol = '0'
	less_10 = True

	for count in range(0,50):
		if less_10:
			if count >= 10:
				additional_symbol = ''
				less_10 = False		
		
		input_image = io.imread('cars/images/trainneg00' + additional_symbol + str(count) + '.jpg')
		train_images_neg.append(color.rgb2gray(input_image))
	

	return train_images_pos, train_images_neg
	#--<

	#TRAIN ROOT-FILTER 
	#scikit-learn svm with --sliding windows

#converting to [x,y,width,heigh]
def convert_pos_obj_cord(pos_str):
	coordinates = pos_str[8:].replace('\n','').split('\t')
	return coordinates

#[x,y,width,heigh] - bounding box
def get_positive_pos():
	positions = []

	less_10 = True
	additional_symbol = '0'
	for count in range(0,50):
		if less_10:
			if count >= 10:
				additional_symbol = ''
				less_10 = False
	
		with open('cars/objects/trainpos00' + additional_symbol + str(count) + '.pgm.objects') as f:
			image_pos = [convert_pos_obj_cord(line) for line in f.readlines()]
        		positions.append(image_pos)

	return positions

def get_training_XY(positive_im, positions, negative_im):
	X = []
	Y = []
	
	H = 185
	W = 300

	from PIL import Image
	from skimage.transform import resize
	import numpy as np

	#Positive
	for i in range(0,50):
		im = Image.fromarray(positive_im[i])
		for el in positions[i]:
			x = int(el[0])
			y = int(el[1])
			dx = int(el[2])
			dy = int(el[3])

			crop = resize(np.asarray(im.crop((x,y,x+dx,y+dy))),(H,W))
			hog_feature = hog(crop, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
			X.append(hog_feature)
			Y.append('1')

	#Negative
	import random
	for i in range(0,50):
		im = Image.fromarray(negative_im[i])
		for count in range(0,50):

			x = int(random.random()*(im.size[0] - W - 5)) + 1
			y = int(random.random()*(im.size[1] - H - 5)) + 1
			dx = W #int(random.random()*(im.size[0] - x - 1)) + 1
			dy = H #int(random.random()*(im.size[1] - y - 1)) + 1

			crop = np.asarray(im.crop((x,y,x+dx,y+dy)))
			hog_feature = hog(crop, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)

			X.append(hog_feature)
			Y.append('0')

		
	return X,Y	
	

def detect(clf_root, image):

	x = 0
	y = 0

	H = 185
	W = 300

	from PIL import Image, ImageFont, ImageDraw
	import numpy as np

	dr = ImageDraw.Draw(image)

	while  y + H < image.size[0]:
		while x + W < image.size[1]:
			im = np.asarray(image.crop((x,y,x+H,y+W)))
			hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
			cl = clf_root.predict([hog_feature])
			if cl == 1:
				print cl
				dr.rectangle(((x,y),(x+H,y+W)), outline = "red")
		
			x += 1
		y += 1
	
	plt.imshow(np.asarray(image))
	plt.show()
	


def train_root(X,Y, C):
	from sklearn import svm
	clf_root = svm.LinearSVC(C=C)
	clf_root.fit(X,Y)
	return clf_root
	
	
	
maxX = 0
maxY = 0
maxS = 0
for ima in positions:
	for el in ima:
		x = int(el[0])
		y = int(el[1])
		dx = int(el[2])
		dy = int(el[3])
		S = dx*dy
		if S > maxS:
			maxX = max(maxX,dx)
			maxY = max(maxY,dy)
			maxS = S



#fd0001 = hog(test0001, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),visualise=False)
"""
im = Image.fromarray(test_image)

	fd0001, hog_image = hog(test0001, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))


	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')
	ax1.set_adjustable('box-forced')

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	ax1.set_adjustable('box-forced')
	plt.show()
"""
