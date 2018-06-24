## Load packages
import numpy as np
import glob
import time

from sklearn.cluster import MiniBatchKMeans
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from datetime import datetime

import argparse

## Function that clusters hue values
def cluster_hue(hue, n_clusters = 2):
	height, width = hue.shape
	hue_flat = hue.reshape(-1,1)
	clustering = MiniBatchKMeans(n_clusters = n_clusters).fit(hue_flat)
	labels = (clustering.labels_).reshape((height, width))

	return labels

## Function that calculates the ratio between clusters
# If the smallest class consists of less than min_ratio (default: 0.1), the ratio is set to min_ratio
# If the smallest class consists of more than max_ratio (default: 0.3), the ratio is set to max_ratio
def get_label_ratio(labels, min_ratio = 0.10, max_ratio = 0.30):
	labels_unique, labels_count = np.unique(labels, return_counts=True)

	min_class = np.min(labels_count)
	n_pixels = np.sum(labels_count)
	actual_ratio = min_class/n_pixels

	if actual_ratio < min_ratio:
		ratio = min_ratio
	elif actual_ratio > max_ratio:
		ratio = max_ratio
	else:
		ratio = actual_ratio

	return ratio

## Function that detects a cluster change in a row/column given the label_ratio
# If the ratio in the row/column is greater than the label_ratio the function returns True (and False otherwise)
def detect_cluster_changes(labels, label_ratio):
	l = len(labels)
	lab_unique, lab_count = np.unique(labels, return_counts = True)
	n_labs = len(lab_unique)

	if n_labs > 1:
		idx_sort = np.argsort(lab_count)[::-1]
		c1 = lab_count[idx_sort[1]]
		change = c1/l > label_ratio

		return change
	else:
		return False

## Function that returns the row/column to crop to in the specified range
# If no crop value is found the crop value is the initial row/column in the range
# i.e. no cropping in the current direction (left, right, top, bottom)
def crop_value(image, initial, crop_range, label_ratio, row = True):
	crop_value = initial
	
	for i in crop_range:
		if row:
			labels = image[i,:]
		else:
			labels = image[:,i]
		
		stop_search = detect_cluster_changes(labels, label_ratio)

		if stop_search:
			crop_value = i
			break

	return crop_value

## Function that returns all 4 crop values (left, right, top, bottom)
# upper_lower_bound specifies how far into the image (from top to bottom and from bottom to top)
# the function can search for a crop value
# Likewise left_right_bound specifies how far into the image (from left to right and from right to left)
# the function can search for a crop value
def get_crop_values(image, upper_lower_bound = 2.5, left_right_bound = 2.5, label_ratio = 0.3):
	height, width = image.shape
	left_crop = 0
	right_crop = width-1
	upper_crop = 0
	lower_crop = height-1

	stop_left = int(width/left_right_bound)
	stop_right = int((left_right_bound - 1)*width/left_right_bound)
	stop_upper = int(height/upper_lower_bound)
	stop_lower = int((upper_lower_bound - 1)*height/upper_lower_bound)

	range_left = range(0,stop_left)
	range_right = reversed(range(stop_right, width))
	range_upper = range(0, stop_upper)
	range_lower = reversed(range(stop_lower, height))

	left_crop = crop_value(image, left_crop, range_left, label_ratio, row = False)
	right_crop = crop_value(image, right_crop, range_right, label_ratio, row = False)
	upper_crop = crop_value(image, upper_crop, range_upper, label_ratio, row = True)
	lower_crop = crop_value(image, lower_crop, range_lower, label_ratio, row = True)

	return left_crop, right_crop, upper_crop, lower_crop

def crop_img(file):

	image = Image.open(file)
	width, height = image.size
	res_width = int(width/4)
	res_height = int(height/4)
	image_res = image.resize((res_width,res_height))
	image_array = np.array(image_res)

	hsv = rgb_to_hsv(image_array/255.0)
	hue = hsv[:,:,0]
	labels = cluster_hue(hue, n_clusters = 2)
	label_ratio = get_label_ratio(labels)
	left_crop, right_crop, upper_crop, lower_crop = get_crop_values(labels, upper_lower_bound = 2.3, left_right_bound = 2.1, label_ratio = label_ratio)

	image_crop = image.crop((left_crop*4, upper_crop*4, right_crop*4, lower_crop*4))

	return(np.array(image_crop))
