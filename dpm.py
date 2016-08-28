from skimage.feature import hog
from skimage import data, color, exposure, transform

import matplotlib.pyplot as plt

import cv2

import numpy as np

from PIL import Image

from skimage import io

from PIL import Image

import random


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


		self.filters = []

	def train_root(self, X,Y, C,kernel='linear',degree=3, gamma='auto', coef0=0.0, trees_count = 200):
		from sklearn import svm
		# clf_root = svm.SVC(kernel='rbf')#svm.LinearSVC(C=C)#OneClassSVM(C=C,kernel=kernel,degree=degree, gamma=gamma, coef0=coef0)
		# clf_root.fit(X,Y)
		# return clf_root	
		#self.clf_root = clf_root
		from sklearn.tree import DecisionTreeClassifier
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier
		
		adaboostclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm='SAMME.R',n_estimators = trees_count)
		adaboostclf.fit(X,Y)
		return adaboostclf
	
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
	
	def image_mag(self, img):

		import numpy
		import scipy
		from scipy import ndimage

		im = img.astype('int32')
		dx = ndimage.sobel(im, 1)  # horizontal derivative
		dy = ndimage.sobel(im, 0)  # vertical derivative
		mag = numpy.hypot(dx, dy)  # magnitude
		#mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
		
		return mag

	def image_fragment_mag(self,mag_map,rect):
		im_mag = mag_map[rect[0]: rect[0] + rect[2],rect[1]:rect[1] + rect[3]].copy()

		return sum(sum(im_mag))

	def zero_mag_map(self,mag_map,rect):
		mag_map[rect[0]: rect[0] + rect[2],rect[1]:rect[1] + rect[3]] = 0


	def parts_of_image(self, im,parts_count = 6):

		# if not isinstance(image, Image.Image):
			# raise Exception('Image should be of type PIL.Image')

		res_parts = []

		i = 0


		mag_map = self.image_mag(im)
		while i < parts_count:

			#self.show_images([image])
			max_mag = 0
			max_x = 0
			max_y = 0

			#self.part_h = 20
			#self.part_w = 30 
			for y in range(0, im.shape[0] - self.part_h, 1):
				for x in range(0, im.shape[1] - self.part_w, 2):
					#im_crop = im[y:y+self.part_h,x:x+self.part_w]
					im_mag = self.image_fragment_mag(mag_map,(y,x,self.part_h,self.part_w))#self.image_mag(im_crop,res_parts)
					
					#print 'im_mag ', im_mag
					#print 'mag_map ', sum(sum(mag_map))

					if [y,x] in res_parts :
						continue
					else:
						if max_mag < im_mag:
							max_mag = im_mag
							max_x = x
							max_y = y

			# draw = ImageDraw.Draw(Image.fromarray(im))
			# draw.rectangle([max_x, max_y,self.part_w,self.part_h], fill=0)
			# del draw


			#self.show_images([image])
			self.zero_mag_map(mag_map,(max_y,max_x,self.part_h,self.part_w))
			res_parts.append([max_y,max_x])
			cv2.imwrite('parts/' + str(i) + '/' + str(random.random()) + '.jpg',im[max_y:max_y+self.part_h,max_x:max_x + self.part_w])
			#if i == 2:
			#im[res_parts[i][0] : res_parts[i][0] + self.part_h, res_parts[i][1] : res_parts[i][1] + self.part_w] = 255
			#cv2.imshow(str(i),im)
			#cv2.waitKey(0)
			#print '[',max_y,',',max_x,'] - ', max_mag



			#cv2.GaussianBlur(im[res_parts[i][0]:res_parts[i][0] + self.part_w,res_parts[i][1]:res_parts[i][0] + self.part_h],(5,5),5,5)
			#cv2.imshow('aa',im)
			#cv2.waitKey(0)
			i += 1

		#self.show_images([im])
		return res_parts



	def collect_pathes_from_train(self, path):
		import os

		path = 'CarData/TrainImages'
		
		im_names = os.listdir(path)

		#im_names.sort()

		im_parts = []

		for name in im_names:
			#print name

			image =  cv2.imread(path + '/' + name,0)
			if name.find('pos') == -1:
				continue
			im_parts.append(self.parts_of_image(image))

		return im_parts

	def compute_average(self,parts_):

		result_parts = []
		for part_i in range(0,6):
			summ_x = 0.0
			summ_y = 0.0

			length = 0

			for el in parts_:
				x = el[part_i][1]
				y = el[part_i][0]

				if x != 0 and y != 0:

					summ_x += x
					summ_y += y
					length += 1

			result_parts.append([summ_y/len(parts_),summ_x/len(parts_)])


		print 'result parts:'
		print result_parts
		return result_parts





	def compute_part_filters(self, path):
		computed_parts = self.collect_pathes_from_train(path)

		res =  computed_parts[0]#self.compute_average(computed_parts)
		print res
		return res


	def init_part_filters(
		self, 
		C = 1, 
		kernel = 'rbf',
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

		parts = self.compute_part_filters('CarData/TrainImages')
		
		self.parts = parts

		filters_F = []

		
		X_with_filters = []
		Y_with_filters = []

		print 'positive'

		filters7 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}

		answers7 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
		


		# REFACTOOOR!
		for el in range(0,550):
			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/pos-' + str(el) + '.pgm',0)
			#filters_el = []

			#feature_vec = []

			root_feature = hog(im, orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			#normed_root_feature = [ float(el*el)/np.sqrt(np.dot(root_feature,root_feature))  for el in root_feature]
			#feature_vec += root_feature.tolist()

			filters7[nmb].append(root_feature)
			answers7[nmb].append('1')

			nmb += 1


			for i in range(0,6):
				part_F = im[parts[i][0]: parts[i][0] + part_h, parts[i][1]: parts[i][1] + part_w]
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

		 		cv2.imwrite('pos/' + str(nmb) + '/' + str(random.random()) + '.jpg',part_F)
		 		#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]
		 		filters7[nmb].append(part_hog)
		 		answers7[nmb].append('1')
		 		nmb += 1
				#filters_el.append(part_hog)
				#feature_vec += part_hog.tolist()
			

			# for i in range(0,6):	
			# 	x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
			# 	y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
			# 	feature_vec += [x_hat]
		 # 		feature_vec += [y_hat]
			# 	feature_vec += [x_hat**2]
			# 	feature_vec += [y_hat**2]
				
				
				
			# feature_vec_new = []
			# norm = np.sqrt(np.dot(feature_vec,feature_vec))
			# for elem in feature_vec:
			# 	feature_vec_new += [(elem + 0.0)/norm]


			#print '+++', el, len(feature_vec_new)
			#X_with_filters.append(feature_vec_new)
			#Y_with_filters.append('1')

		
		print 'negative'

		for el in range(0,500):
			nmb = 0
			im = cv2.imread('CarData/TrainImages/neg-' + str(el) + '.pgm',0)
			#filters_el = []

			#feature_vec = []

			root_feature = hog(im, orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			#normed_root_feature = [ float(el*el)/np.sqrt(np.dot(root_feature,root_feature))  for el in root_feature]

			#feature_vec += root_feature.tolist()

			filters7[nmb].append(root_feature)#root_feature.tolist())
			answers7[nmb].append('0')

			nmb += 1

			for i in range(0,6):
				part_F = im[parts[i][0]:parts[i][0] + part_h,parts[i][1]: parts[i][1] + part_w]
		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

		 		cv2.imwrite('neg/' + str(nmb) + '/' + str(random.random()) +  '.jpg',part_F)
		 		#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]
		 		filters7[nmb].append(part_hog)
				answers7[nmb].append('0')

				nmb += 1

				#filters_el.append(part_hog)
				#feature_vec += part_hog.tolist()
			

			# for i in range(0,6):	
			# 	x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
			# 	y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
			# 	feature_vec += [x_hat]
			# 	feature_vec += [y_hat]
			# 	feature_vec += [x_hat**2]
			# 	feature_vec += [y_hat**2]
				
			# feature_vec_new = []

			# norm = np.sqrt(np.dot(feature_vec,feature_vec))
			# for elem in feature_vec:
				
			# 	feature_vec_new += [(elem + 0.0)/norm]


			# #print el, '---', len(feature_vec_new)
			# X_with_filters.append(feature_vec_new)
			# Y_with_filters.append('0')

		#print [len(filters7[nmb][0]) for nmb in range(0,7)]
		#print answers7

		print 'train time'

		adaboosts7 = {}
		for key in filters7.keys():
			adaboosts7[key] = self.train_root(filters7[key],answers7[key],C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, trees_count = trees)
			

		self.clfs = adaboosts7

		print 'train is over'

		return adaboosts7

		#return self.train_root(X_with_filters,Y_with_filters,C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, trees_count = trees)


	def process_filter_image(self,im):
		y = 0
		x = 0


		pix_per_cell = 8 
		cells_per_bl = 2 
		pix_per_cell_1 = 4 
		cells_per_bl_1 = 2

		height = len(im)
		width = len(im[0])

		max_prob = {1:0,2:0,3:0,4:0,5:0,6:0}
		max_coord_point = {1:(0,0),2:(0,0),3:(0,0),4:(0,0),5:(0,0),6:(0,0)}

		while y in range(0,height - self.part_h):
			x = 0
			while x in range(0,width - self.part_w):

				patch_F = im[y:y+self.part_h,x:x+self.part_w]
				patch_hog = hog(patch_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

				for j in range(1,7):

					probs = self.clfs[j].predict_proba([patch_hog])

					#print probs[0][1]
					#print clf.classes_
					if probs[0][1] > max_prob[j]:
						cv2.imwrite('parts/' + str(j) + '/' + str(random.random()) + '.jpg', patch_F)
						max_coord_point[j] = (y,x)
						max_prob[j] = probs[0][1]



				x += self.step_x

			y += self.step_y


		return max_coord_point
	

	def get_filters_cost(self, best_coord, filters_nmb):

		#Euclidean
		return np.sqrt((self.parts[filters_nmb - 1][0] - best_coord[0])**2 + (self.parts[filters_nmb - 1][1] - best_coord[1])**2)





	def process_frame(self,im):
		nmb = 0

		summ_cost = 0

		pix_per_cell = 8 
		cells_per_bl = 2 
		pix_per_cell_1 = 4 
		cells_per_bl_1 = 2

		root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))



		if self.clfs[nmb].predict([root_feature]) == '1':
			summ_cost += 0.5	

		nmb += 1

		best_coord = self.process_filter_image(im)

		for key in best_coord.keys():

			filter_cost = self.get_filters_cost(best_coord[key],key)

			summ_cost += 1.0/(1 + filter_cost)

			#part_F = np.asarray(Image.fromarray(im).crop((self.parts[i][1],self.parts[i][0],self.parts[i][1]+self.part_w,self.parts[i][0]+self.part_h)))

		 # 	part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

		 	#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]
		 # 	answers += [clfs[nmb].predict([part_hog]) == '1']
	 	#nmb += 1


	 	print 'summ_cost ', summ_cost
		if summ_cost > 1:
			print 'car here'
			return True
		else:
			return False
		# cnt = 0
		# for el in answers:
		# 	if el:
		# 		cnt += 1

		#return cnt



	def process_image(self, image):	

		print 'process image'

		height = len(image)
		width = len(image[0])

		im_result = image.copy()

		#TODO resize

		y = 0
		x = 0
		while y in range(0,height - self.image_h):
			x = 0
			while x in range(0,width - self.image_w):

				print y,' ', x
				res = self.process_frame(image[y : y + self.image_h,x : x + self.image_w])
				#print res
				if res:
					cv2.rectangle(im_result,(x,y),(x + self.image_w,y + self.image_h),(255),2)
				x += self.step_x + 25


			y += self.step_y + 10

		cv2.imwrite('results_test/' + str(random.random()) + 'res.jpg',im_result)



	def new_test_train(self, clfs, pix_per_cell = 8, cells_per_bl = 2,pix_per_cell_1 = 4, cells_per_bl_1 = 2):
		print 'testing train'

		tp = 0
		tf = 0

		#refactor
		parts = self.parts

		for el in range(0,550):
			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/pos-' + str(el) + '.pgm',0)

			if self.process_frame(im):
				tp += 1

			# root_feature = hog(im, orientations=9, pixels_per_cell=(pix_per_cell,pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			# #normed_root_feature = [ float(el*el)/np.sqrt(np.dot(root_feature,root_feature))  for el in root_feature]
			# #feature_vec += root_feature.tolist()

			# answers = []

			# answers += [clfs[nmb].predict([root_feature]) == '1']
			# # print clfs[nmb].predict([root_feature.tolist()])
			# # print clfs[nmb].predict([root_feature.tolist()]) == '1'

			# nmb += 1

			# for i in range(0,6):
			# 	part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
		 # 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

		 # 		#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]
		 # 		answers += [clfs[nmb].predict([part_hog]) == '1']
		 # 		# print clfs[nmb].predict([part_hog.tolist()])
		 # 		# print clfs[nmb].predict([part_hog.tolist()]) == '1'

		 # 		nmb += 1
			# 	#filters_el.append(part_hog)
			# 	#feature_vec += part_hog.tolist()
			

			# # for i in range(0,6):	
			# # 	x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
			# # 	y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
			# # 	feature_vec += [x_hat]
		 # # 		feature_vec += [y_hat]
			# # 	feature_vec += [x_hat**2]
			# # 	feature_vec += [y_hat**2]
				
			# #commented for AdaBoost
			# #score = np.dot(clf.coef_[0],feature_vec)
			# #scores_pos.append(score)
			# # if clf.predict([feature_vec]) == '1':
			# # 	tp += 1

			# #print answers
			# cnt = 0
			# for el in answers:
			# 	if el:
			# 		cnt += 1
			# if cnt == 7:
			# 	tp += 1
		

		scores_neg = []
		for el in range(0,500):
			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/neg-' + str(el) + '.pgm',0)
			#filters_el = []

			if not self.process_frame(im):
				tf += 1

			#feature_vec = []

			# root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			# #feature_vec += root_feature.tolist()

			# #normed_root_feature = [ float(el*el)/np.sqrt(np.dot(root_feature,root_feature))  for el in root_feature]

			# answers = []

			# answers += [clfs[nmb].predict([root_feature]) == '0']

			# nmb += 1

			# for i in range(0,6):
			# 	part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
		 # 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

	 	# 		#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]
		 # 		answers += [clfs[nmb].predict([part_hog]) == '0']
		 # 		nmb += 1

			# 	#feature_vec += part_hog.tolist()
			

			# # for i in range(0,6):	
			# # 	x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
			# # 	y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
			# # 	feature_vec += [x_hat]
			# # 	feature_vec += [y_hat]
			# # 	feature_vec += [x_hat**2]
			# # 	feature_vec += [y_hat**2]


			# # feature_vec_new = []
			# # norm = np.sqrt(np.dot(feature_vec,feature_vec))
			# # for elem in feature_vec:
			# # 	feature_vec_new += [(elem + 0.0/norm)]

			# #score = np.dot(clf.coef_[0],feature_vec)
			# #scores_neg.append(score)
			# # if clf.predict([feature_vec_new]) == '0':
			# # 	tp += 1

			# cnt = 0
			# for el in answers:
			# 	if el:
			# 		cnt += 1
			# if cnt == 7:
			# 	tf += 1

		return tp, tf, (tp + tf)/(0.0 + 1050)#scores_pos,scores_neg,


		#another trash, delete
	def get_new_test(self,clfs,pix_per_cell = 8, cells_per_bl = 2,pix_per_cell_1 = 4, cells_per_bl_1 = 2):
		print 'testing'
		annotations = open('CarData/trueLocations.txt','r')

		parts = self.parts

		tp = 0
		all_obj = 0
		falsepos = 0

		#scores = []

		for count in range(0,170):
			input_image = cv2.imread('CarData/TestImages/test-' + str(count) + '.pgm',0)

			self.process_image(input_image)

		# 		answers += [clfs[nmb].predict([root_feature]) == '1']

		# 		nmb += 1


		# 		for i in range(0,6):
		# 			#parts in clojure
		# 			part_F = np.asarray(Image.fromarray(im).crop((parts[i][1],parts[i][0],parts[i][1]+self.part_w,parts[i][0]+self.part_h)))
		# 	 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

	 # 				#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]

		# 	 		answers += [clfs[nmb].predict([part_hog]) == '1']

		# 			nmb += 1

		# 			#filters_el.append(part_hog)
		# 			#feature_vec += part_hog.tolist()
			

		# 		# for i in range(0,6):	
		# 		# 	x_hat = (parts[i][1] - 2 * 0 + (50 + 0))/100.0
		# 		# 	y_hat = (parts[i][0] - 2 * 0 + (20 + 0))/40.0
				
		# 		# 	feature_vec += [x_hat]
		# 	 # 		feature_vec += [y_hat]
		# 		# 	feature_vec += [x_hat**2]
		# 		# 	feature_vec += [y_hat**2]
				
				
		# 		cnt = 0
		# 		for el in answers:
		# 			if el:
		# 				cnt += 1
		# 		if cnt > 0:
		# 			tp += 1
				
		# 		#REMOVE NORMALIZATION
		# 		# feature_vec_new = []
		# 		# norm = np.sqrt(np.dot(feature_vec,feature_vec))
		# 		# for elem in feature_vec:
		# 		# 	feature_vec_new += [(elem + 0.0/norm)]
				
		# 		#score = np.dot(clf.coef_,feature_vec)
		# 		#scores_pos.append(score)
		# 		# if clf.predict([feature_vec_new]) == '1':
		# 		# 	tp += 1
				
		# annotations.close()
		# return (tp+0.0)/all_obj


