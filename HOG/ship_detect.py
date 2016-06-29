from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

import csv
import re
import math
import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

import os.path
import sys
import glob

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

CLF_FILE = "genfiles/ships_clf.pkl"
TRAIN_LABEL_FILE = "genfiles/ships_label.csv"
TRAIN_FEATURES_FILE = "genfiles/ships_feature.dat"

FILEPATH_PREFIX = "../../../../temp/CT_OCEAN/"

def getAllFilesInDirectory(dir):
	return glob.glob(dir)

def train():
	trainPositiveFiles = getAllFilesInDirectory("train/ship/pos/*.png")
	trainNegativeFiles = getAllFilesInDirectory("train/ship/neg/*.png")

	labels = np.array([0 for i in range(len(trainNegativeFiles))] + \
						[1 for i in range(len(trainPositiveFiles))])

	list_hog_fd = []
	for filepath in (trainNegativeFiles + trainPositiveFiles):
		im = cv2.imread(filepath, 0)
		fd = hog(im, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')

	clf = LinearSVC()
	clf.fit(hog_features, labels)
	joblib.dump(clf, CLF_FILE, compress=3)

def calculateRXD(band, rows, cols):
	# calculate global covariance matrix
	# n/a
	# end calculate global covariance matrix
	
	rxd = np.zeros((rows, cols))
	tRXD = np.zeros((rows, cols))
	iRXD = np.zeros((rows, cols))

	sub_rect_cols = 128
	sub_rect_rows = 128
	window = 2
	
	x = 0
	while x < rows - 1:
		x1 = x + sub_rect_rows if x + sub_rect_rows <= rows - 1 else rows - 1

		y = 0
		while y < cols - 1:
			y1 = y + sub_rect_cols if y + sub_rect_cols <= cols - 1 else cols - 1

			brow = x - window if x - window >= 0 else 0
			trow = x1 + window if x1 + window < rows - 1 else rows - 1
			lcol = y - window if y - window >= 0 else 0
			rcol = y1 + window if y1 + window < cols - 1 else cols - 1

			drows = trow - brow + 1
			dcols = rcol - lcol + 1

			data = band.ReadAsArray(lcol, brow,
				dcols, drows).astype(np.int)

			dev = np.std(data)
			if dev >= 3:
				n = 4 * window * window
				sub_total = (x1 - x + 1) * (y1 - y + 1)

				# calculate histogram
				binwidth = 20
				hist, bin_edges = np.histogram(data, bins=np.arange(0, np.max(data) + 2 * binwidth, binwidth))

				# calculate integral image
				integral = np.zeros((drows, dcols))
				integral_sqr = np.zeros((drows, dcols))
				for i in range(drows):
					for j in range(dcols):
						integral[i, j] = data[i, j] + (integral[i - 1, j] if i - 1 >= 0 else 0)\
											+ (integral[i, j - 1] if j - 1 >= 0 else 0)\
											- (integral[i - 1, j - 1] if (j - 1 >= 0 and i - 1 >= 0) else 0)

						integral_sqr[i, j] = data[i, j] ** 2 + (integral_sqr[i - 1, j] if i - 1 >= 0 else 0)\
											+ (integral_sqr[i, j - 1] if j - 1 >= 0 else 0)\
											- (integral_sqr[i - 1, j - 1] if (j - 1 >= 0 and i - 1 >= 0) else 0)

				# calculate rxd
				for i in xrange(x, x1):
					for j in xrange(y, y1):
						signal = data[i - brow, j - lcol]
						if signal == 0:
							continue

						# get mean of neighborhood pixels data
						bnrow = i - window - brow
						bnrow = bnrow if bnrow > 0 else 0

						tnrow = (i + window if i + window <= trow else trow) - brow
						
						lncol = j - window - lcol
						lncol = lncol if lncol > 0 else 0

						rncol = (j + window if j + window <= rcol else rcol) - lcol

						if data[bnrow, lncol] == 0 or data[bnrow, rncol] == 0 or data[tnrow, lncol] == 0 or data[tnrow, rncol] == 0:
							continue

						S1 = integral[tnrow, rncol] + integral[bnrow, lncol]\
								- integral[bnrow, rncol] - integral[tnrow, lncol]

						S2 = integral_sqr[tnrow, rncol] + integral_sqr[bnrow, lncol]\
								- integral_sqr[bnrow, rncol] - integral_sqr[tnrow, lncol]

						tRXD[i, j] = (math.sqrt(n * S2 - (S1 ** 2)) / S1) if S1 > 0 else 0

						index = int(signal / binwidth)
						iRXD[i, j] = (sub_total / (hist[index] * 100)) if (hist[index]) > 0 else 1
			y = y1
		x = x1
	rxd = iRXD / np.max(iRXD) + tRXD / np.max(tRXD)
	return rxd

def thresholdAnomaly(rxd, rows, cols):
	# finding threshold
	threshold = 1

	binwidth = 0.01
	
	# minval = rxd[rxd > 0].min()
	hist, bin_edges = np.histogram(rxd, bins=np.arange(0, np.max(rxd) + binwidth, binwidth))

	pixels = sum(hist)
	total = 0
	for i in range(len(hist) - 1, 0, -1):
		total += hist[i]
		if float(total) / pixels * 100.00 >= 4:
			threshold = bin_edges[i]
			break
	print "RXD threshold:", threshold
	# end finding threshold
	
	anomaly = np.zeros((rows, cols), dtype=np.uint8)
	indices = rxd > threshold
	anomaly[indices] = 255
	
	return anomaly

def makeData():
	# list of train images
	train_images = []
	list_hog_fd = []
	for filename in train_images:
		filepath = FILEPATH_PREFIX + filename + ".tif"
		vispath = FILEPATH_PREFIX + filename + ".png"
		
		print filepath
		if os.path.isfile(filepath) and os.path.isfile(vispath):
			# read color visualizable image
			vis = cv2.imread(vispath)
			greyscale = cv2.imread(vispath, 0)

			# read pan image
			dataset = gdal.Open(filepath, GA_ReadOnly)
			cols = dataset.RasterXSize
			rows = dataset.RasterYSize
			num_bands = dataset.RasterCount
			
			band = dataset.GetRasterBand(1)

			# calculate rxd image
			print "Calculating RXD"
			rxd = calculateRXD(band, rows, cols)

			# threshold the image
			print "Thresholding image"
			anomaly = thresholdAnomaly(rxd, rows, cols)

			# finding ROI positions
			print "Finding potential candidates"

			pan = band.ReadAsArray().astype(np.int)

			# noise removal
			kernel = np.ones((5, 5), np.uint8)
			opening = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel, iterations = 2)

			# connected compponents
			kernel = np.ones((5, 5), np.uint8)
			im_th = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 2)

			ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			rects = [cv2.boundingRect(ctr) for ctr in ctrs]
			
			for rect in rects:
				leng = int(rect[3] * 1.6)
				pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
				pt1 = pt1 if pt1 > 0 else 0

				pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
				pt2 = pt2 if pt2 > 0 else 0

				roi = greyscale[pt1 : pt1 + leng if pt1 + leng < rows - 1 else rows - 1, \
							pt2 : pt2 + leng if pt2 + leng < cols - 1 else cols - 1]
				
				roi = cv2.resize(roi, (28, 28))
				roi = cv2.dilate(roi, (3, 3))

				fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

				# draw contour

				# save the vis
				cv2.imwrite("results/" + filename + ".png", vis)
		else:
			print "File Not Found"

def test():
	# load classifier
	clf = joblib.load(CLF_FILE)

	testFiles = getAllFilesInDirectory("test/ship/*.png")
	for filepath in testFiles:
		im = cv2.imread(filepath, 0)
		hog_fd = hog(im, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		nbr = clf.predict(np.array([hog_fd], 'float64'))

		print filepath, nbr[0]

if __name__ == '__main__':
	if not os.path.isfile(CLF_FILE):
		train()

	test()