from skimage.feature import hog
from skimage import data, color, exposure, transform

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from skimage import io

class PartFilter:
	"""Here F is a part filter(HOG), v is a two-dimensional vector specifying the center for
a box of possible positions for part relative to the root position,
s gives the size of this box, while ai and bi are twodimensional
vectors specifying coefficients of a quadratic
function measuring a score for each possible placement of
the part.   """
	def __init__(self,F,v,s,a,b):
		self.F = F
		self.v = v
		self.s = s
		self.a = a
		self.b = b
	

class DPM:
	
	step_x = 4
	step_y = 2
	#from Car_Detection place
	def get_train(self):
		train_images_pos = []
		train_images_neg = []

		#train images -> gray
	
		for count in range(0,550):
	
			input_image = io.imread('CarData/TrainImages/pos-' + str(count) + '.pgm')
			train_images_pos.append(color.rgb2gray(input_image))

		for count in range(0,500):
	
			input_image = io.imread('CarData/TrainImages/neg-' + str(count) + '.pgm')
			train_images_neg.append(color.rgb2gray(input_image))
	

		self.train_images_pos = train_images_pos
		self.train_images_neg = train_images_neg

		return train_images_pos, train_images_neg
		

		#TRAIN ROOT-FILTER 
		#scikit-learn svm with --sliding windows

	def get_training_XY(self, positive_im, negative_im):
		X = []
		Y = []
	
		#window is strictly given for now
		H = 100
		W = 40

		from PIL import Image
		#from skimage.transform import resize
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
		
		self.X = X
		self.Y = Y
		self.X_norm = []
		for el in X:
			self.X_norm.append(el/(0.0 + np.sqrt(np.dot(el,el))))
		return X,Y	


	def train_root(self, X,Y, C):
		from sklearn import svm
		clf_root = svm.LinearSVC(C=C)
		clf_root.fit(X,Y)
		self.clf_root = clf_root
		return clf_root


	def test_with_show(self, clf_root, image):

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
					im = np.asarray(origin_image_rescaled.crop((x,y,x+W,y+H)))#(y,x,y+H,x+W)))# X -> Y AAAAA					    
					hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
					cl = clf_root.predict([hog_feature])

					if cl[0] == '1':
						dr.rectangle(((x,y),(x+W,y+H)), fill = None, outline = None)
		
					x += step_x
				y = y + step_y


			images_scales.append(np.asarray(origin_image_rescaled))
	
			end = time.time()
			print end - start
		
			rescale_coeff *= coeff
			del dr

		self.show_images(images_scales)
		
	

	#returns detected boxes
	def test_boxes(self, clf_root, image):

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
					im = np.asarray(image.crop((x,y,x+W,y+H)))#(y,x,y+H,x+W)))# X -> Y AAAAA							    
					hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
					cl = clf_root.predict([hog_feature])

					if cl[0] == '1':
						boxes.append((x,y))
		
					x += step_x
				y = y + step_y

			rescale_coeff *= coeff
		return boxes



	#DEPRECATED
	def testing(self, clf_root):
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


	def testing_train(self, clf_root, X, Y):
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

	
	
	
	def get_XY_test(self):
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



	def show_images(self, images,titles=None):
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

	#def score(self,filters 


	def init_part_filters(self):
		part_w = 20
		part_h = 10

		parts = []
		#[w,h] point on image 
		parts.append([10,35])
		parts.append([12,55])
		parts.append([20,75])
		parts.append([25,60])
		parts.append([25,15])
		parts.append([20,5])

		filters_F = []

		im = io.imread('CarData/TrainImages/pos-0.pgm')
		#im = np.asarray(origin_image_rescaled.crop((y,x,y+H,x+W)))					    
		#hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
		
		X_with_filters = []
		Y_with_filters = []

		for el in range(0,550):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/pos-' + str(el) + '.pgm'))
			#filters_el = []

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2))

				#filters_el.append(part_hog)
				feature_vec += part_hog.tolist()
			

			for i in range(0,6):	
				x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
				y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
				feature_vec += [x_hat]
		 		feature_vec += [y_hat]
				feature_vec += [x_hat**2]
				feature_vec += [y_hat**2]
				
				
				
			feature_vec_new = []
			for el in feature_vec:
				feature_vec_new += (el + 0.0)/np.sqrt(np.dot(el,el))


			X_with_filters.append(feature_vec_new)
			Y_with_filters.append('1')
		
		for el in range(0,500):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/neg-' + str(el) + '.pgm'))
			#filters_el = []

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2))

				#filters_el.append(part_hog)
				feature_vec += part_hog.tolist()
			

			for i in range(0,6):	
				x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
				y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
				feature_vec += [x_hat]
				feature_vec += [y_hat]
				feature_vec += [x_hat**2]
				feature_vec += [y_hat**2]
				
			feature_vec_new = []
			for el in feature_vec:
				feature_vec_new += (el + 0.0)/np.sqrt(np.dot(el,el))


			X_with_filters.append(feature_vec_new)
			Y_with_filters.append('0')

			




		return self.train_root(X_with_filters,Y_with_filters,1)
			
			
	def find_car_filters(self,clf,im, parts):
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

		origin_image = Image.fromarray(np.copy(np.asarray(im)))

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
					im = np.asarray(origin_image_rescaled.crop((x,y,x+W,y+H)))#(y,x,y+H,x+W)))# X -> Y AAAAA
					feature_vec = []					    
					hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

					feature_vec = hog_feature.tolist()
					for i in range(0,6):
						part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 				part_hog = hog(part_F, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2))
						feature_vec += part_hog.tolist()
					
					for i in range(0,6):	
						x_hat = (parts[i][1] - 2 * x + (50 + x))/100.0
						y_hat = (parts[i][0] - 2 * y + (20 + y))/40.0
				
						feature_vec += [x_hat]
						feature_vec += [y_hat]
						feature_vec += [x_hat**2]
						feature_vec += [y_hat**2]
	

					cl = clf.predict([feature_vec])

					if cl[0] == '1':
						dr.rectangle(((x,y),(x+W,y+H)), fill = None, outline = None)
		
					x += step_x
				y = y + step_y


			images_scales.append(np.asarray(origin_image_rescaled))
	
			end = time.time()
			print end - start
		
			rescale_coeff *= coeff
			del dr
		#correct please
		dpm.show_images(images_scales)

		
	#test trash 
	def new_test_train(self, clf):
		tp = 0;

		scores_pos = []
		for el in range(0,550):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/pos-' + str(el) + '.pgm'))

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2))

				#filters_el.append(part_hog)
				feature_vec += part_hog.tolist()
			

			for i in range(0,6):	
				x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
				y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
				feature_vec += [x_hat]
		 		feature_vec += [y_hat]
				feature_vec += [x_hat**2]
				feature_vec += [y_hat**2]
				
				
				
			score = np.dot(clf.coef_,feature_vec)
			scores_pos.append(score)
			if clf.predict([feature_vec]) == '1':
				tp += 1
		

		scores_neg = []
		for el in range(0,500):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/neg-' + str(el) + '.pgm'))
			#filters_el = []

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2))

				#filters_el.append(part_hog)
				feature_vec += part_hog.tolist()
			

			for i in range(0,6):	
				x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
				y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
				feature_vec += [x_hat]
				feature_vec += [y_hat]
				feature_vec += [x_hat**2]
				feature_vec += [y_hat**2]
				
			score = np.dot(clf.coef_,feature_vec)
			scores_neg.append(score)
			if clf.predict([feature_vec]) == '0':
				tp += 1
		return scores_pos,scores_neg#,tp, tp/(0.0 + 1050)








	#another trash, delete
	def get_new_test(self,clf,parts):
		annotations = open('CarData/trueLocations.txt','r')

		from PIL import Image

		X_test = []
		Y_test = []

		tp = 0
		all_obj = 0
		falsepos = 0

		#scores = []

		for count in range(0,170):
			objects_raw = annotations.readline().replace('\n','').split(' ')[1:]
			objects = []
			for el in objects_raw:
				all_obj += 1
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
				im = np.asarray(Image.fromarray(input_image).crop((w,h,w+100,h+40)))					    
				root_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

				feature_vec = []

				#root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))

				feature_vec += root_feature.tolist()


				for i in range(0,6):
					#parts in clojure
					part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
			 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2))

					#filters_el.append(part_hog)
					feature_vec += part_hog.tolist()
			

				for i in range(0,6):	
					x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
					y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
					feature_vec += [x_hat]
			 		feature_vec += [y_hat]
					feature_vec += [x_hat**2]
					feature_vec += [y_hat**2]
				
				
				
				#score = np.dot(clf.coef_,feature_vec)
				#scores_pos.append(score)
				if clf.predict([feature_vec]) == '1':
					tp += 1
				
		annotations.close()
		return (tp+0.0)/all_obj


	#def coord_transform(self, x0,y0, xi,yi, vi, si):# vi =  smth (_,_)
	#	return (xi,yi) - 2*(x0,y0) + vi)/si

