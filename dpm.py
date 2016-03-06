from skimage.feature import hog
from skimage import data, color, exposure, transform

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

	#less_10 = True
	#additional_symbol = '0'
	for count in range(0,500):
		#if less_10:
		#	if count >= 10:
		#		additional_symbol = ''
		#		less_10 = False

		input_image = io.imread('CarData/TrainImages/pos-' + str(count) + '.pgm')
		train_images_pos.append(color.rgb2gray(input_image))

	#additional_symbol = '0'
	#less_10 = True

	for count in range(0,500):
		#if less_10:
		#	if count >= 10:
		#		additional_symbol = ''
		#		less_10 = False		
		
		input_image = io.imread('CarData/TrainImages/neg-' + str(count) + '.pgm')
		train_images_neg.append(color.rgb2gray(input_image))
	

	return train_images_pos, train_images_neg
	#--<

	#TRAIN ROOT-FILTER 
	#scikit-learn svm with --sliding windows

#converting to [x,y,width,heigh]
# DEPRECATED
def convert_pos_obj_cord(pos_str):
	coordinates = pos_str[8:].replace('\n','').split('\t')
	return coordinates

#[x,y,width,heigh] - bounding box
#DEPRECATED
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

def get_training_XY(positive_im, negative_im):#positions,
	X = []
	Y = []
	
	H = 100
	W = 40

	from PIL import Image
	from skimage.transform import resize
	import numpy as np

	#Positive
	for i in range(0,500):
		im = Image.fromarray(positive_im[i])
		#for el in positions[i]:
		#	x = int(el[0])
		#	y = int(el[1])
		#	dx = int(el[2])
		#	dy = int(el[3])

		#	crop = resize(np.asarray(im.crop((x,y,x+dx,y+dy))),(H,W))
		#	hog_feature = hog(crop, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
		hog_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
		X.append(hog_feature)
		Y.append('1')

	#Negative
	import random
	for i in range(0,500):
		im = Image.fromarray(negative_im[i])
		#for count in range(0,50):

			# x = int(random.random()*(im.size[0] - W - 5)) + 1
			# y = int(random.random()*(im.size[1] - H - 5)) + 1
			# dx = W #int(random.random()*(im.size[0] - x - 1)) + 1
			# dy = H #int(random.random()*(im.size[1] - y - 1)) + 1

			# crop = np.asarray(im.crop((x,y,x+dx,y+dy)))
			# hog_feature = hog(crop, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)

		hog_feature = hog_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
		X.append(hog_feature)
		Y.append('0')

		
	return X,Y	
	

def test_with_show(clf_root, image):

	H = 40
	W = 100

	from PIL import Image, ImageFont, ImageDraw
	import numpy as np
	import pylab
	from skimage import transform
	
	rescale_coeff = 1

	while int(image.size[1]*rescale_coeff) >= H and int(image.size[0]*rescale_coeff) >= W:
		
		image = Image.fromarray(transform.rescale(np.asarray(image),rescale_coeff))
		dr = ImageDraw.Draw(image)

		x = 0
		y = 0

		while  y + H <= image.size[1]:
			x = 0
			while x + W <= image.size[0]:
				im = np.asarray(image.crop((x,y,x+W,y+H)))					    
				hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
				cl = clf_root.predict([hog_feature])

				#print x,y, image.size[0],image.size[1], y + H, cl
				if cl[0] == '1':
					print cl
					dr.rectangle(((x,y),(x+W,y+H)), fill = None, outline = None)
		
				x += 1
			y = y + 1
	
		plt.imshow(np.asarray(image))
		plt.show()
		rescale_coeff *= 2.0/3

def test(clf_root, image):

	H = 40
	W = 100

	boxes = []

	from PIL import Image, ImageFont, ImageDraw
	import numpy as np
	import pylab
	from skimage import transform
	
	rescale_coeff = 1

	while int(image.size[1]*rescale_coeff) >= H and int(image.size[0]*rescale_coeff) >= W:
		
		image = Image.fromarray(transform.rescale(np.asarray(image),rescale_coeff))

		x = 0
		y = 0

		while  y + H <= image.size[1]:
			x = 0
			while x + W <= image.size[0]:
				im = np.asarray(image.crop((x,y,x+W,y+H)))					    
				hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
				cl = clf_root.predict([hog_feature])

				if cl[0] == '1':
					#print cl
					boxes.append((x,y))
		
				x += 1
			y = y + 1

		rescale_coeff *= 2.0/3
	return boxes

def testing(clf_root):
	#TestImages
	annotations = open('CarData/trueLocations.txt','r')

	from PIL import Image

	truepos = 0
	falsepos = 0

	delta = 10 #pixels

	founded_obj = 0

	for count in range(0,170):
		print count
		objects_raw = annotations.readline().replace('\n','').split(' ')[1:]
		objects = []
		for el in objects_raw:
			height = int(el.replace('(','').replace(')','').split(',')[0])
			width = int(el.replace('(','').replace(')','').split(',')[1])
			objects.append([height,width])
			print el

		input_image = io.imread('CarData/TestImages/test-' + str(count) + '.pgm')
		im_boxes = test(clf_root, Image.fromarray(input_image))
		
		for box in im_boxes:
			
			fp = True
			for obj in objects:
				if  obj[0]-delta <= box[0] <= obj[0]+delta and obj[1]-delta <= box[1] <= obj[1]+delta:
					fp = False
					break
			if fp:
				falsepos += 1
			else:
				truepos += 1
	annotations.close()
	return truepos, falsepos


def train_root(X,Y, C):
	from sklearn import svm
	clf_root = svm.LinearSVC(C=C)
	clf_root.fit(X,Y)
	return clf_root
	
	
	




#fd0001 = hog(test0001, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),visualise=False)
"""

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
