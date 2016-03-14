from skimage.feature import hog
from skimage import data, color, exposure, transform

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from skimage import io

class DPM:

	#from Car_Detection place
	def get_train():
		train_images_pos = []
		train_images_neg = []

		#train images -> gray
	
		for count in range(0,550):
	
			input_image = io.imread('CarData/TrainImages/pos-' + str(count) + '.pgm')
			train_images_pos.append(color.rgb2gray(input_image))

		for count in range(0,500):
	
			input_image = io.imread('CarData/TrainImages/neg-' + str(count) + '.pgm')
			train_images_neg.append(color.rgb2gray(input_image))
	

		return train_images_pos, train_images_neg
		

		#TRAIN ROOT-FILTER 
		#scikit-learn svm with --sliding windows

	def get_training_XY(positive_im, negative_im):
		X = []
		Y = []
	
		#window is strictly given for now
		H = 100
		W = 40

		from PIL import Image
		from skimage.transform import resize
		import numpy as np

		#Positive
		for i in range(0,500):
			im = Image.fromarray(positive_im[i])

			hog_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

			X.append(hog_feature)
			Y.append('1')

		#Negative
		import random
		for i in range(0,500):
			im = Image.fromarray(negative_im[i])

			hog_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

			X.append(hog_feature)
			Y.append('0')
		
		return X,Y	


	

	def test_with_show(clf_root, image):

		import time

		H = 40
		W = 100

		from PIL import Image, ImageFont, ImageDraw
		import numpy as np
		import pylab
		from skimage import transform
	
		# try to detect in different scales
		rescale_coeff = 1

		coeff = 1.0/2

		origin_image = Image.fromarray(np.copy(np.asarray(image)))

		images_scales = []
	
		while int(origin_image.size[1]*rescale_coeff) >= H and int(origin_image.size[0]*rescale_coeff) >= W:

			start = time.time()	
			origin_image_rescaled = Image.fromarray(transform.rescale(np.asarray(origin_image),rescale_coeff))
			dr = ImageDraw.Draw(origin_image_rescaled)
		
		

			x = 0
			y = 0

			while  y + H <= origin_image_rescaled.size[1]:
				x = 0
				while x + W <= origin_image_rescaled.size[0]:
					im = np.asarray(origin_image_rescaled.crop((y,x,y+H,x+W)))					    
					hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
					cl = clf_root.predict([hog_feature])

					if cl[0] == '1':
						dr.rectangle(((x,y),(x+W,y+H)), fill = None, outline = None)
		
					x += 1
				y = y + 1


			images_scales.append(np.asarray(origin_image_rescaled))
	
			end = time.time()
			print end - start
		
			rescale_coeff *= coeff
			del dr

		show_images(images_scales)
		
	

	#returns detcted boxes
	def test_boxes(clf_root, image):

		H = 40
		W = 100

		boxes = []

		from PIL import Image, ImageFont, ImageDraw
		import numpy as np
		import pylab
		from skimage import transform
	
		rescale_coeff = 1

		coeff = 2.0/3

		while int(image.size[1]*rescale_coeff) >= H and int(image.size[0]*rescale_coeff) >= W:
		
			image = Image.fromarray(transform.rescale(np.asarray(image),rescale_coeff))

			x = 0
			y = 0

			while  y + H <= image.size[1]:
				x = 0
				while x + W <= image.size[0]:
					im = np.asarray(image.crop((y,x,y+H,x+W)))					    
					hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
					cl = clf_root.predict([hog_feature])

					if cl[0] == '1':
						boxes.append((x,y))
		
					x += 1
				y = y + 1

			rescale_coeff *= coeff
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
			im_boxes = test_boxes(clf_root, Image.fromarray(input_image))
		
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


	def testing_train(clf_root, X, Y):
		#TestImages
		#annotations = open('CarData/trueLocations.txt','r')

		from PIL import Image

		truepos = 0
		falsepos = 0

		founded_obj = 0

		count = 0
		for window in X:
			if clf_root.predict([window])[0] == Y[count]:
				truepos += 1
			else: 
				falsepos += 1
			count += 1

		return truepos, falsepos

	def train_root(X,Y, C):
		from sklearn import svm
		clf_root = svm.LinearSVC(C=C)
		clf_root.fit(X,Y)
		return clf_root
	
	
	def get_XY_test():
		annotations = open('CarData/trueLocations.txt','r')

		from PIL import Image

		X_test = []
		Y_test = []

		truepos = 0
		falsepos = 0

		for count in range(0,170):
			objects_raw = annotations.readline().replace('\n','').split(' ')[1:]
			objects = []
			for el in objects_raw:
				height = int(el.replace('(','').replace(')','').split(',')[0])
				width = int(el.replace('(','').replace(')','').split(',')[1])
				objects.append([height,width])

			input_image = io.imread('CarData/TestImages/test-' + str(count) + '.pgm')

			for obj in objects:
				w = obj[0] # height
				h = obj[1] # width
				if obj[1] < 0:
					h = 0
				if obj[0] < 0:
					w = 0
				im = np.asarray(Image.fromarray(input_image).crop((h,w,h+40,w+100)))					    
				X_test.append(hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2)))
				Y_test.append('1')
				
		annotations.close()
		return X_test, Y_test



	def show_images(images,titles=None):
	    #Display a list of images
	    n_ims = len(images)
	    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
	    fig = plt.figure()
	    n = 1
	    for image,title in zip(images,titles):
		a = fig.add_subplot(1,n_ims,n) # Make subplot
		#if image.ndim == 2: # Is image grayscale?
		#    plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
		plt.imshow(image)
		a.set_title(title)
		n += 1
	    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
	    plt.show()
