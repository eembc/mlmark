# 
# Copyright (C) 2019 EEMBC(R). All Rights Reserved
# 
# All EEMBC Benchmark Software are products of EEMBC and are provided under the
# terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
# are proprietary intellectual properties of EEMBC and its Members and is
# protected under all applicable laws, including all applicable copyright laws.  
# 
# If you received this EEMBC Benchmark Software without having a currently
# effective EEMBC Benchmark License Agreement, you must discontinue use.
# 

import os
import sys
import numpy as np
import logging as log
import eelib.models as models
import eelib.paths as paths
import cv2

def mobilenet(fn):
	'''Preprocess a MobileNet V1.0 / ILSVRC2012 image

	Convert to RGB
	Scale to 256x256 with bilinear filter
	Center crop a 224x224 region
	Convert to Numpy float array
	Mean subtract RGB dataset means
	Normalize to range of +/-1.0

	Args:
		fn: Fully qualified path to the image file

	Return:
		array: Numpy float array of RGB values
	'''
	img = cv2.imread(fn)
	if img is None:
		raise Exception("Failed to read file %s" % fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
	img = img[16:240, 16:240]
	rgb_means = (123.68, 116.78, 103.94)
	data = np.asarray(img)
	data = data - rgb_means
	data = data / 128.
	return data

def resnet50(fn):
	'''Preprocess a ResNet-50 V1.0 / ILSVRC2012 image

	Convert to RGB
	Scale to 256x256 with bilinear filter
	Center crop a 224x224 region
	Convert to Numpy float array
	Mean subtract RGB dataset means
	No normalization

	Args:
		fn: Fully qualified path to the image file

	Return:
		array: Numpy float array of RGB values
	'''
	img = cv2.imread(fn)
	if img is None:
		raise Exception("Failed to read file %s" % fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
	img = img[16:240, 16:240]
	rgb_means = (123.68, 116.78, 103.94)
	data = np.asarray(img)
	data = data - rgb_means
	return data

def ssdmobilenet(fn):
	'''Preprocess a SSD-MobileNet V1.0 / COCO2017 image

	Convert to RGB
	Scale to 300x300 with bilinear filter
	No crop
	Convert to Numpy float array
	No mean subtraction
	No normalization

	Args:
		fn: Fully qualified path to the image file

	Return:
		array: Numpy float array of RGB values
	'''
	img = cv2.imread(fn)
	if img is None:
		raise Exception("Failed to read file %s" % fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
	data = np.asarray(img)
	return data

def apiProcess(modelName, fn):
	'''Route the preprocessing to the correct model's preprocessor.'''
	log.debug("Preprocessing: %s" % fn)
	if modelName == models.MOBILENET:
		return mobilenet(fn)
	elif modelName == models.RESNET50:
		return resnet50(fn)
	elif modelName == models.SSDMOBILENET:
		return ssdmobilenet(fn)
	else:
		log.critical("Invalid model '%s' passed to preprocessor" % modelName)
		sys.exit(1)
