from skimage.feature import hog
from skimage import data, color, exposure, transform

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from skimage import io

from PIL import Image



'''
plans -> 7 adaboosts with changed distance of partfilters
'''

# class PartFilter:
# 	"""
# 	CHANGE OR DELETE

# 	Here F is a part filter(HOG), v is a two-dimensional vector specifying the center for
# a box of possible positions for part relative to the root position,
# s gives the size of this box, while ai and bi are twodimensional
# vectors specifying coefficients of a quadratic
# function measuring a score for each possible placement of
# the part.   """
# 	def __init__(self, x_coord, y_coord, width, height):
# 		self.x_coord = x_coord
# 		self.y_coord = y_coord
# 		self.width = width
# 		self.height = height


# 	def train_part(dpm, trees_count):
# 		from sklearn.ensemble import AdaBoostClassifier
# 		from sklearn.tree import DecisionTreeClassifier

# 		X_part = []
# 		Y_part = []

# 		for im in self.train_images_pos:

# 			im = np.asarray(im.crop((self.x_coord,self.y_coord,self.x_coord+self.width,self.y_coord+self.height)))					    
# 			X_part.append(hog(im, orientations=9, pixels_per_cell=(dpm.pix_per_cell, dpm.pix_per_cell),cells_per_block=(dpm.cells_per_bl, dpm.cells_per_bl)))
# 			Y_part.append('1')

# 		for im in self.train_images_neg:

# 			im = np.asarray(im.crop((self.x_coord,self.y_coord,self.x_coord+self.width,self.y_coord+self.height)))					    
# 			X_test.append(hog(im, orientations=9, pixels_per_cell=(dpm.pix_per_cell, dpm.pix_per_cell),cells_per_block=(dpm.cells_per_bl, dpm.cells_per_bl)))
# 			Y_test.append('0')


# 		adaboostclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm='SAMME', n_estimators = trees_count)
# 		adaboostclf.fit(X_part,Y_part)
		
# 		self.adaclf = adaboostclf

class DPM:
	
	image_h = 40
	image_w = 100

	# step_x = 4
	# step_y = 2
	
	# #HOG configuration
	# pix_per_cell_root = 8
	# cells_per_block_root = 2
	
	# pix_per_cell_part_part = 8
	# cells_per_block_part = 2

	# # our partfilters
	# parts = []

	
	# # partfilter size
	# part_w = 30
	# part_h = 20

	# # train hog features
	# X = []
	# Y = []

	# # train images
	# train_images_pos = []
	# train_images_neg = [] 


	def __init__(
		self, 
		step_x = 4, 
		step_y = 2, 
		pix_per_cell_root = 8, 
		cells_per_block_root = 2, 
		pix_per_cell_part = 4, 
		cells_per_block_part = 2, 
		parts = [], 
		part_w = 30, 
		part_h = 20):

		self.step_x = step_x
		self.step_y = step_y

		#HOG configuration
		self.pix_per_cell_root = pix_per_cell_root
		self.cells_per_block_root = cells_per_block_root
	
		self.pix_per_cell_part = pix_per_cell_part
		self.cells_per_block_part = cells_per_block_part

		# our partfilters 
		#ADD more Documentation
		self.parts = parts
	
		# partfilter size
		self.part_w = part_w
		self.part_h = part_h



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

	def get_training_XY(self, positive_im, negative_im, pix_per_cell = 8, cells_per_bl = 2, is_svm = False):
		X = []
		Y = []

		#Positive
		for i in range(0,500):
			im = Image.fromarray(positive_im[i])

			hog_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			X.append(hog_feature)
			Y.append('1')

		#Negative
		import random
		for i in range(0,500):
			im = Image.fromarray(negative_im[i])

			hog_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			X.append(hog_feature)
			Y.append('0')
		
		self.X = X
		self.Y = Y
		# self.X_norm = []
		# if is_svm:
		# 	for el in X:
		# 		self.X_norm.append(el/(0.0 + np.sqrt(np.dot(el,el))))
		# 	return X_norm, Y
		return X,Y	




	def train_root(self, X,Y, C,kernel='linear',degree=3, gamma='auto', coef0=0.0, trees_count = 200):
		#from sklearn import svm
		#clf_root = svm.LinearSVC(C=C)#OneClassSVM(C=C,kernel=kernel,degree=degree, gamma=gamma, coef0=coef0)
		#clf_root.fit(X,Y)
		#self.clf_root = clf_root
		#from sklearn.tree import DecisionTreeClassifier
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier
		
		adaboostclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm='SAMME',n_estimators = trees_count)
		adaboostclf.fit(X,Y)
		return adaboostclf



	def image_mag(self, image):

		import numpy
		import scipy
		from scipy import ndimage

		img = numpy.asarray(image)

		im = img.astype('int32')
		dx = ndimage.sobel(im, 1)  # horizontal derivative
		dy = ndimage.sobel(im, 0)  # vertical derivative
		mag = numpy.hypot(dx, dy)  # magnitude
		#mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
		
		return sum(sum(mag))
	

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


	def testing_train(self, clf_root, X, Y):#error_rate_train

		from PIL import Image

		truepos = 0
		falsepos = 0

		founded_obj = 0

		count = 0
		for window_feature in X:
			if clf_root.predict([window_feature])[0] == Y[count]:
				truepos += 1
			else: 
				falsepos += 1
			count += 1

		return 'truepos:',truepos,'falsepos:',falsepos

	
	
	
	def get_XY_test(self, pix_per_cell = 8, cells_per_bl = 2):
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
				X_test.append(hog(im, orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl)))
				Y_test.append('1')
				
		annotations.close()
		return X_test, Y_test

	def show_images(self, images,titles=None):
		import matplotlib.pyplot as plt
	
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


	# image as 2d-array
	# def ave_image_entropy(self, image, disk_radius = 2):
	# 	from skimage.filters.rank import entropy
	# 	from skimage.morphology import disk
	# 	return sum(map(sum,entropy(image,disk(disk_radius))))/(0.0 + len(image)*len(image[0]))


	def set_black_rectangle(self, image, max_w, max_h, part_w, part_h):
		im = image.copy()
		im = np.asarray(im)
		im.flags.writeable = True
		for j in range(max_w, max_w + part_w):
			for i in range(max_h, max_h + part_h):
				im[i][j] = 0
		return Image.fromarray(im)


	# image as PIL Image
	def parts_of_image(self, image,parts_count = 6):

		from PIL import ImageDraw

		if not isinstance(image, Image.Image):
			raise Exception('Image should be of type PIL.Image')

		res_parts = []

		i = 0

		while i < parts_count:

			#self.show_images([image])
			max_mag = 0
			max_w = 0
			max_h = 0


			# TODO REFACTOR height and width ->> y and x (point)
			for height in range(0, self.image_h - self.part_h, 1):
				for width in range(0, self.image_w - self.part_w, 2):
					im_crop = np.array(image.crop((width,height,width+self.part_w,height+self.part_h)))
					im_mag = self.image_mag(im_crop)
					
					if [width,height] in res_parts :
						continue
					if max_mag < im_mag:
						max_mag = im_mag
						max_w = width
						max_h = height

			draw = ImageDraw.Draw(image)
			draw.rectangle([max_w, max_h,self.part_w,self.part_h], fill=0)
			del draw

			#self.show_images([image])
			res_parts.append([max_w,max_h])
			print '[',max_w,',',max_h,'] - ',max_mag



			#image = self.set_black_rectangle(image, max_w, max_h, self.part_w, self.part_h)
			i += 1

		self.show_images([image])
		return res_parts




	def init_part_filters(
		self, 
		C = 1, 
		kernel = 'linear',
		degree=3, 
		gamma='auto', 
		coef0=0.0, 
		pix_per_cell = 8, 
		cells_per_bl = 2,
		pix_per_cell_1 = 4, 
		cells_per_bl_1 = 2, 
		trees = 200):
	
		part_w = self.part_w #20
		part_h = self.part_h #10

		parts = []
		#[w,h] point on image 
		parts.append([10,35])
		parts.append([12,55])
		parts.append([20,65])
		parts.append([25,60])
		parts.append([25,15])
		parts.append([20,5])
		
		self.parts = parts

		filters_F = []

		# im = io.imread('CarData/TrainImages/pos-0.pgm')
		
		X_with_filters = []
		Y_with_filters = []

		for el in range(0,550):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/pos-' + str(el) + '.pgm'))
			#filters_el = []

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

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
			norm = np.sqrt(np.dot(feature_vec,feature_vec))
			for elem in feature_vec:
				feature_vec_new += [(elem + 0.0)/norm]


			#print '+++', el, len(feature_vec_new)
			X_with_filters.append(feature_vec_new)
			Y_with_filters.append('1')
		
		for el in range(0,500):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/neg-' + str(el) + '.pgm'))
			#filters_el = []

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+part_w,parts[i][0]+part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

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

			norm = np.sqrt(np.dot(feature_vec,feature_vec))
			for elem in feature_vec:
				
				feature_vec_new += [(elem + 0.0)/norm]


			#print el, '---', len(feature_vec_new)
			X_with_filters.append(feature_vec_new)
			Y_with_filters.append('0')

			




		return self.train_root(X_with_filters,Y_with_filters,C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, trees_count = trees)
			
			
	def find_car_filters(
		self, 
		clf,
		im, 
		parts,
		pix_per_cell = 8, 
		cells_per_bl = 2,
		pix_per_cell_1 = 8, 
		cells_per_bl_1 = 2, 
		):
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
					hog_feature = hog(im, orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

					feature_vec = hog_feature.tolist()
					for i in range(0,6):
						part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
		 				part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))
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
		
					x += self.step_x
				y = y + self.step_y


			images_scales.append(np.asarray(origin_image_rescaled))
	
			end = time.time()
			print end - start
		
			rescale_coeff *= coeff
			del dr
		#correct please
		dpm.show_images(images_scales)

		
	#test trash 
	def new_test_train(self, clf, pix_per_cell = 8, cells_per_bl = 2,pix_per_cell_1 = 4, cells_per_bl_1 = 2):
		tp = 0;

		#refactor
		parts = self.parts

		scores_pos = []
		for el in range(0,550):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/pos-' + str(el) + '.pgm'))

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell,pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

				#filters_el.append(part_hog)
				feature_vec += part_hog.tolist()
			

			for i in range(0,6):	
				x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
				y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
				feature_vec += [x_hat]
		 		feature_vec += [y_hat]
				feature_vec += [x_hat**2]
				feature_vec += [y_hat**2]
				
			#commented for AdaBoost
			#score = np.dot(clf.coef_[0],feature_vec)
			#scores_pos.append(score)
			if clf.predict([feature_vec]) == '1':
				tp += 1
		

		scores_neg = []
		for el in range(0,500):
	
			im = color.rgb2gray(io.imread('CarData/TrainImages/neg-' + str(el) + '.pgm'))
			#filters_el = []

			feature_vec = []

			root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			feature_vec += root_feature.tolist()


			for i in range(0,6):
				part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

				feature_vec += part_hog.tolist()
			

			for i in range(0,6):	
				x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
				y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
				feature_vec += [x_hat]
				feature_vec += [y_hat]
				feature_vec += [x_hat**2]
				feature_vec += [y_hat**2]

			#score = np.dot(clf.coef_[0],feature_vec)
			#scores_neg.append(score)
			if clf.predict([feature_vec]) == '0':
				tp += 1
		return tp, tp/(0.0 + 1050)#scores_pos,scores_neg,








	#another trash, delete
	def get_new_test(self,clf,pix_per_cell = 8, cells_per_bl = 2,pix_per_cell_1 = 4, cells_per_bl_1 = 2):
		annotations = open('CarData/trueLocations.txt','r')

		from PIL import Image

		parts = self.parts

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
				root_feature = hog(im, orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

				feature_vec = []
				feature_vec += root_feature.tolist()


				for i in range(0,6):
					#parts in clojure
					part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
			 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

					#filters_el.append(part_hog)
					feature_vec += part_hog.tolist()
			

				for i in range(0,6):	
					x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
					y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
					feature_vec += [x_hat]
			 		feature_vec += [y_hat]
					feature_vec += [x_hat**2]
					feature_vec += [y_hat**2]
				
				
				
				#REMOVE NORMALIZATION
				feature_vec_new = []
				norm = np.sqrt(np.dot(feature_vec,feature_vec))
				for elem in feature_vec:
					feature_vec_new += [(elem + 0.0/norm)]
				
				#score = np.dot(clf.coef_,feature_vec)
				#scores_pos.append(score)
				if clf.predict([feature_vec]) == '1':
					tp += 1
				
		annotations.close()
		return (tp+0.0)/all_obj


	def configurate(self):
		for pix_per_c in [8,16]:
			for cell_p_block in [2,4]:
				for pix_per_c_1 in [8]:#[4,8]:
					for cell_p_block_1 in [2]:
						test_score_prev = 2
						test_score_curr = 0

						for trees_count in range(300,801, 100):
							if cell_p_block == 2 and pix_per_c==4 :
								continue

							print '# Test', 'pix_per_cell:', pix_per_c, 'cell_per_block:', cell_p_block, 'pix_per_cell_1:', pix_per_c_1, 'cell_per_block_1:', cell_p_block_1, 'trees_count:', trees_count

							adaboost = self.init_part_filters(pix_per_cell = pix_per_c, cells_per_bl = cell_p_block,pix_per_cell_1 = pix_per_c_1, cells_per_bl_1 = cell_p_block_1, trees = trees_count)

							print 'train errors:', self.new_test_train(adaboost, pix_per_cell = pix_per_c, cells_per_bl = cell_p_block,pix_per_cell_1 = pix_per_c_1, cells_per_bl_1 = cell_p_block_1)

							test_score_curr = self.get_new_test(adaboost,pix_per_cell = pix_per_c, cells_per_bl = cell_p_block,pix_per_cell_1 = pix_per_c_1, cells_per_bl_1 = cell_p_block_1)

							#if test_score_curr < test_score_prev:
							print 'test errors:', test_score_curr
							#test_score_prev = test_score_curr
							#else:
							#	break


