from skimage.feature import hog
from skimage import data, color, exposure, transform
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
import random
import time
import math
from itertools import compress

def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def normalize(v):
    vmag = magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]


class DPM:
	
	image_h = 22 # 14 - 36
	image_w = 90 # 5 - 95

	def __init__(
		self, 
		step_x = 4, 
		step_y = 2, 
		pix_per_cell_root = 8, 
		cells_per_block_root = 2, 
		pix_per_cell_part = 4, 
		cells_per_block_part = 2, 
		parts_count = 3, 
		part_w = 30, 
		part_h = 20):

		self.step_x = step_x
		self.step_y = step_y

		#HOG configuration
		self.pix_per_cell_root = pix_per_cell_root
		self.cells_per_block_root = cells_per_block_root
	
		self.pix_per_cell_part = pix_per_cell_part
		self.cells_per_block_part = cells_per_block_part

		

		self.parts_count = parts_count
	
		# partfilter size
		self.part_w = part_w
		self.part_h = part_h



		self.filters = []

		self.importance_treshold = 1e-3
		self.importance_features = []
		self.importance_features_bool = []

	def train_clf(self, X, Y, C, count, kernel='linear', degree=3, gamma='auto', coef0=0.0, trees_count = 1000):
		"""
		Train adaclf of specific filter
		"""

		from sklearn.tree import DecisionTreeClassifier
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier
		


		adaboostclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm='SAMME.R',n_estimators = trees_count)

		adaboostclf.fit(X,Y)

		print len(adaboostclf.feature_importances_)

		important_features = adaboostclf.feature_importances_ > self.importance_treshold

		self.importance_features.append(adaboostclf.feature_importances_)
		np.save('dump_clfs4/importance_' + str(count),adaboostclf.feature_importances_)
		self.importance_features_bool.append(important_features)


		X_new = []

		for x in X:
			X_new.append(list(compress(x,important_features)))

		adaboostclf.fit(X_new,Y)	

		print len(adaboostclf.feature_importances_)	

		return adaboostclf
	
	def show_images(self, images, titles=None):
		"""
		Typical function which show us images. It uses pyplot
		"""
		
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

		#img = cv2.GaussianBlur(img,(7,7),0)
		#print img
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


	def parts_of_image(self, im, parts_count):#6):

		# if not isinstance(image, Image.Image):
			# raise Exception('Image should be of type PIL.Image')

		res_parts = []

		i = 0


		mag_map = self.image_mag(im)

		cv2.imwrite('mag.jpg',mag_map)

		#print mag_map
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

			self.zero_mag_map(mag_map,(max_y,max_x,self.part_h,self.part_w))
			cv2.imwrite('mag' + str(i) + '.jpg',mag_map)
			res_parts.append([max_y,max_x])
			#cv2.imwrite('parts/' + str(i) + '/' + str(random.random()) + '.jpg',im[max_y:max_y+self.part_h,max_x:max_x + self.part_w])
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
		
		#im_names = os.listdir(path)

		#im_names.sort()
		im_parts = []

		for el in range(0,550):

			image = cv2.imread('CarData/TrainImages/pos-' + str(el) + '.pgm',0)

			image = image[14:36,5:95]

			#image  = cv2.GaussianBlur(image,(7,7),0)
			im_parts.append(self.parts_of_image(image,parts_count=self.parts_count))

		return im_parts

	def compute_average(self,parts_):

		result_parts = []

		median = np.median(list(parts_),axis=1)

		length = 0

		#print median

		summ_x = 0.0
		summ_y = 0.0

		for el in median:
			x = el[1]
			y = el[0]

			summ_x += x
			summ_y += y
			length += 1
		# for part_i in range(0,self.parts_count):
		# 	summ_x = 0.0
		# 	summ_y = 0.0

		# 	length = 0

		# 	for el in parts_:
		# 		x = el[part_i][1]
		# 		y = el[part_i][0]

		# 		# if x != 0 and y != 0:

		# 		summ_x += x
		# 		summ_y += y
		# 		length += 1

			result_parts.append([summ_y/length,summ_x/length])


		print 'result parts:'
		#print result_parts
		return result_parts





	def compute_part_filters(self, path):
		computed_parts = self.collect_pathes_from_train(path)

		#print computed_parts
		res =  self.compute_average(computed_parts) #computed_parts[0] 
		print res
		return res, computed_parts

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
		trees = 1000):
	
		part_w = self.part_w #20
		part_h = self.part_h #10

		parts, parts_per_image = self.compute_part_filters('CarData/TrainImages')
		self.parts = parts

		filters_F = []

		
		X_with_filters = []
		Y_with_filters = []

		print 'positive'

		#filters7 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
		filters7 = {}
		answers7 = {}
		for part_num in range(0,self.parts_count + 1):
			filters7[part_num] = []
			answers7[part_num] = []	

		
		


		# REFACTOOOR!
		for el in range(0,500):
			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/pos-' + str(el) + '.pgm',0)
			#filters_el = []

			#feature_vec = []

			#im = cv2.GaussianBlur(im,(7,7),0)

			im = im[14:36,5:95]

			root_feature = hog(im, orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

			#root_feature = normalize(root_feature)
			#normed_root_feature = [ float(el*el)/np.sqrt(np.dot(root_feature,root_feature))  for el in root_feature]
			#feature_vec += root_feature.tolist()

			filters7[nmb].append(root_feature)
			answers7[nmb].append('1')

			nmb += 1


			for i in range(0,self.parts_count):
				part_F = im[parts_per_image[el][i][0]: parts_per_image[el][i][0] + part_h, parts_per_image[el][i][1]: parts_per_image[el][i][1] + part_w]

		 		part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

		 		filters7[nmb].append(part_hog)
		 		answers7[nmb].append('1')
		 		nmb += 1
		
		print 'negative'

		for el in range(0,450):
			nmb = 0
			im = cv2.imread('CarData/TrainImages/neg-' + str(el) + '.pgm',0)

			root_feature = hog(im[14:36,5:95], orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))


			filters7[nmb].append(root_feature)
			answers7[nmb].append('0')

			nmb += 1

			height = len(im)
			width = len(im[0])

			for y in range(0, height - self.part_h,20):
				for x in range(0, width - self.part_w,20):
					nmb = 1

					part_F = im[y:y + self.part_h,x: x + self.part_w]

		 			part_hog = hog(part_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))

		 			#part_hog = normalize(part_hog)

					for i in range(1,self.parts_count + 1):
		 				#cv2.imwrite('neg/' + str(nmb) + '/' + str(random.random()) +  '.jpg',part_F)
		 				#normed_feature = [ float(el*el)/np.sqrt(np.dot(part_hog,part_hog))  for el in part_hog]
		 				filters7[nmb].append(part_hog)
						answers7[nmb].append('0')

						nmb += 1

		print 'train time'

		from sklearn.externals import joblib

		adaboosts7 = {}

		j = 0
		for j in range(0,self.parts_count + 1):
			adaboosts7[j] = joblib.load('dump_clfs4/clf_' + str(j) + '.pkl')
			self.importance_features.append(np.load('dump_clfs4/importance_' + str(j) + '.npy'))
			important_features = self.importance_features[j] > self.importance_treshold
			self.importance_features_bool.append(important_features)



		# for key in filters7.keys():
		# 	adaboosts7[key] = self.train_clf(filters7[key],answers7[key],C=C, count=key, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, trees_count = trees)
		# 	print adaboosts7[key].classes_
		# 	print'estimators len', len(adaboosts7[key].estimators_)
		# 	#print 'estimators ', key, ' ', adaboosts7[key].estimators_
		# 	# print 'estimator_weight ', adaboosts7[key].estimator_weights_




		self.clfs = adaboosts7



		# j = 0
		# for key in adaboosts7.keys():
		# 	joblib.dump(adaboosts7[key], 'dump_clfs4/clf_' + str(j) + '.pkl')
		# 	j += 1

		# print 'saved'

		print 'train is over'

		return adaboosts7

		#return self.train_clf(X_with_filters,Y_with_filters,C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, trees_count = trees)


	def clf_predict(self, pred_vec, clf, clf_count):

		start = time.time()
		features = list(compress(pred_vec,self.importance_features_bool[clf_count]))
		stop = time.time()
		print 'compress_time ', stop - start
		return clf.predict_proba([list(compress(pred_vec,self.importance_features_bool[clf_count]))])

	def process_filter_image(self,im):
		# start = time.time()

		y = 0
		x = 0


		pix_per_cell = 8 
		cells_per_bl = 2 
		pix_per_cell_1 = 4 
		cells_per_bl_1 = 2

		height = len(im)
		width = len(im[0])

		max_prob = {}
		max_coord_point = {}

		for part_num in range(1,self.parts_count + 1):
			max_prob[part_num] = 0
			max_coord_point[part_num] = (500,500)

		#max_prob = {1:0,2:0,3:0,4:0,5:0,6:0}
		#max_coord_point = {1:(0,0),2:(0,0),3:(0,0),4:(0,0),5:(0,0),6:(0,0)}

		start_all_hogs = time.time()
		while y in range(0,height - self.part_h,2):
			x = 0
			while x in range(0,width - self.part_w,5):

				patch_F = im[y:y+self.part_h,x:x+self.part_w]

				start_hog = time.time()
				patch_hog = hog(patch_F, orientations=9, pixels_per_cell=(pix_per_cell_1, pix_per_cell_1),cells_per_block=(cells_per_bl_1, cells_per_bl_1))
				stop_hog = time.time()
				print 'time hog little: ', stop_hog - start_hog

				for j in range(1,self.parts_count + 1):

					start_predict = time.time()
					probs = self.clf_predict(patch_hog,self.clfs[j],j)#self.clfs[j].predict_proba([patch_hog])
					#is_part = self.clfs[j].predict([patch_hog]) == '1'
					stop_predict = time.time()
					print 'time predict: ', stop_predict - start_predict

					if probs[0][1] > 0.5 and probs[0][1] > max_prob[j]:# and is_part:# and self.clfs[j].predict([patch_hog]):
						#cv2.imwrite('parts/' + str(j) + '/' + str(random.random()) + '.jpg', patch_F)
						max_coord_point[j] = (y,x)
						max_prob[j] = probs[0][1]



				x += self.step_x

			y += self.step_y

		stop = time.time()

		print 'time all hogs little: ', stop - start_all_hogs
		print max_prob
		# print 'time elaplsed all: ', stop - start
		return max_coord_point
	

	def get_filters_cost(self, best_coord, filters_nmb):

		#Euclidean
		return np.sqrt((self.parts[filters_nmb - 1][0] - best_coord[0])**2 + (self.parts[filters_nmb - 1][1] - best_coord[1])**2)/(self.parts_count)





	def process_frame(self,im):
		nmb = 0

		summ_cost = 0

		pix_per_cell = 8 
		cells_per_bl = 2 
		pix_per_cell_1 = 4 
		cells_per_bl_1 = 2

		#im = cv2.GaussianBlur(im,(7,7),0)

		root_feature = hog(np.asarray(im), orientations=9, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cells_per_bl, cells_per_bl))

		#root_feature= normalize(root_feature)


		main_filter = False
		if self.clf_predict(root_feature,self.clfs[nmb],nmb)[0][1] == '1':# self.clfs[nmb].predict([root_feature])[0][1] == '1':
			print 'main filter'
			summ_cost += 0.5
			main_filter = True

		nmb += 1

		best_coord = self.process_filter_image(im)

		for key in best_coord.keys():

			print best_coord[key], ' ', key
			filter_cost = self.get_filters_cost(best_coord[key],key)

			cost = 1.0/(1 + filter_cost)

			print 'key: ', key, ' cost: ', cost 

			summ_cost += cost

	 	print 'summ_cost ', summ_cost

	 	if main_filter:

	 		tresh = 1
	 	else:
	 		tresh = 1.4
		if summ_cost > tresh:
			print 'car here'
			return True
		else:
			return False



	def process_image(self, image, name = ''):	

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
					cv2.rectangle(im_result,(x,y),(x + self.image_w,y + self.image_h),(255),1)
				x += self.step_x + 6
				print 


			y += self.step_y +  3

		#cv2.imwrite('results_test/' + name + '_' + str(random.random()) + '.jpg',im_result)



	def new_test_train(self, clfs, pix_per_cell = 8, cells_per_bl = 2,pix_per_cell_1 = 4, cells_per_bl_1 = 2):
		print 'testing train'

		tp = 0
		fp = 0


		tn = 0
		fn = 0

		i = 0

		for el in range(0,550):
			print 'IMAGE --- ', i

			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/pos-' + str(el) + '.pgm',0)

			if self.process_frame(im[14:36,5:95]):
				tp += 1
			else:
				fp += 1

			i += 1
		
		for el in range(0,500):
			print 'IMAGE --- ', i

			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/neg-' + str(el) + '.pgm',0)
			#filters_el = []

			if not self.process_frame(im[14:36,5:95]):
				tn += 1
			else:
				fn += 1

			i += 1


		recall = tp/(0.0+tp+fn)
		precision = tp/(0.0+tp+fp)
		accuracy = (tp + tn)/(0.0 + tp+fn+fp+tn)
		f1 = 2*precision * recall/(precision + recall)

		print 'RESULT: ','tp=',tp, 'fp=',fp,"tn=",tn,"fn=",fn,"recall=",recall,"precision=",precision,"accuracy=",accuracy ,"f1=",f1

	def test_valid(self,clfs):
		print 'testing validation set'

		tp = 0
		fp = 0


		tn = 0
		fn = 0

		i = 0

		for el in range(500,550):
			print 'IMAGE --- ', i

			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/pos-' + str(el) + '.pgm',0)

			if self.process_frame(im[14:36,5:95]):
				tp += 1
			else:
				fp += 1

			i += 1
		
		for el in range(450,500):
			print 'IMAGE --- ', i

			nmb = 0
	
			im = cv2.imread('CarData/TrainImages/neg-' + str(el) + '.pgm',0)
			#filters_el = []

			if not self.process_frame(im[14:36,5:95]):
				tn += 1
			else:
				fn += 1

			i += 1


		recall = tp/(0.0+tp+fn)
		precision = tp/(0.0+tp+fp)
		accuracy = (tp + tn)/(0.0 + tp+fn+fp+tn)
		f1 = 2*precision * recall/(precision + recall)

		print 'RESULT_VALID: ','tp=',tp, 'fp=',fp,"tn=",tn,"fn=",fn,"recall=",recall,"precision=",precision,"accuracy=",accuracy ,"f1=",f1


		#another trash, delete
	def get_new_test(self,clfs,pix_per_cell = 8, cells_per_bl = 2,pix_per_cell_1 = 4, cells_per_bl_1 = 2):
		print 'testing'
		annotations = open('CarData/trueLocations.txt','r')

		tp = 0
		all_obj = 0
		falsepos = 0

		#scores = []

		tp = 0
		fp = 0

		for count in range(0,170):
			input_image = cv2.imread('CarData/TestImages/test-' + str(count) + '.pgm',0)

			inp = annotations.readline()
			inp = inp.split(' ')[1:]

			coords = []
			for el in inp:
				el = el.strip()
				coord = el[1:-1].split(',')
				print coord

				y = 0 if coord[0] == '' else int(coord[0])
				x = 0 if coord[1] == '' else int(coord[1])
				#coord = [int(c) for c in coord]


				if y < 0 or y == '':
					y = 0
				if x < 0 or x == '':
					x = 0
				coords.append((y,x))


			p = 0
			for car in coords:

				cv2.imwrite('CarData/test_little_images/pos-' + str(count) + '-' + str(p) + '.jpg',input_image[y:y+40,x:x+100])

				# cv2.imshow('a',input_image[y:y+40,x:x+100][14:36,5:95])
				# cv2.waitKey(0)
				answer = self.process_frame(input_image[y:y+40,x:x+100][14:36,5:95])#process_image(input_image, 'testing_' + str(count))

				if answer:
					tp += 1
				else:
					fp += 1
			p += 1

		print tp, ' ', fp
