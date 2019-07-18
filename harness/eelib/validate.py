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
import json
import numpy as np
import logging as log
from collections import defaultdict
from operator import itemgetter
from eelib import models
from eelib import paths
from eelib import constants
from eelib.coco import COCO

def getIou(b1, b2):
	'''Compute the interscection over union area for two rectangles

	Args:
		b1 (list of floats): Bounding box in the format [x1, y1, x2, y2]
		b1 (list of floats): Bounding box in the format [x1, y1, x2, y2]

	Returns:
		(float): Intersection over union area
	'''
	# Areas
	b1a = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
	b2a = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
	# Intersection box
	x1 = max(b1[0], b2[0])
	y1 = max(b1[1], b2[1])
	x2 = min(b1[2], b2[2])
	y2 = min(b1[3], b2[3])
	# Intersection area
	i = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
	# Union area
	u = b1a + b2a - i
	iou = i / float(u)
	return iou

def calcCocoAp(recall, precision):
	'''Compute the COCO average precision.

	This is a naive implementation of AP calculation. Optimization suggestions
	are welcomed. the recall vs precision is accumulated based on the max 
	precision >= recall[i].

	Args:
		recall (list of floats): Recall values
		precision (list of floats): Precision values

	Returns:
		(float): The average precision
	'''
	assert(len(recall) == len(precision))
	n = len(recall)
	acc = 0
	# COCO requires 101 point interpolation
	for i in np.arange(0, 1.01, .01):
		mx = 0
		# Find max precision for recall >= i
		for j in range(0, n):
			if i < recall[j]:
				mx = max(mx, precision[j])
		acc = acc + mx
	return acc / 101.

def calcMap(mi, results):
	'''Compute the COCO mean-average-precision

	Given the results schema described in the documentation, and an annotation
	set associated with the input stimuli, run the AP against all classes for
	10 thresholds, then return the mean of these 800 APs.

	Args:
		mi (object): model-info structure
		results (list): the results from a detecton run (see report.py)

	Returns:
		(float): The mAP, multiplied by 100
	'''
	fn = os.path.join(paths.DATASETS, mi['dataset'], 'annotations', mi['groundtruth'])
	log.debug('Loading annotations...')
	coco = COCO(fn)
	# First, find all of the objects it found and sort by confidence
	detectedObjects = defaultdict(list)
	imgIds = []
	# Sort the detections in categories as a tuple of score, box and image ID
	log.debug('Coalescing detections...')
	for (fullpath, detections) in results:
		# Extract the image name from the filename
		path, file = os.path.split(fullpath)
		name, ext = os.path.splitext(file)
		# Remove leading text '0's
		imgId = int(name)
		imgIds.append(imgId)
		for det in detections:
			catId = int(det['class'])
			conf = det['score']
			bbox = det['box']
			# Enforce the >=30% confidence rule for mAP
			if conf >= 0.30:
				detectedObjects[catId].append((conf, bbox, imgId))

	# For each category of detections, order them by confidence score
	allAp = []
	log.debug('Iterating over %d detected categories' % len(detectedObjects))
	for (catId, detections) in detectedObjects.items():
		# Sort the detections so the AP computation starts with best guesses
		detections.sort(key=itemgetter(0), reverse=True)
		# Choose only the annotation ids for the given category and images
		annIds = coco.getAnnIds(imgIds=imgIds, catIds=catId)
		anns = coco.loadAnns(annIds)
		N = len(annIds)
		for thresh in np.arange(0.5, 1., 0.05):
			precision = []
			recall = []
			TP = 0
			FP = 0
			log.debug('  Checking %d detections at thresh %.2f' % (len(detections), thresh))
			for detection in detections:
				imgId = detection[2]
				box = detection[1]
				conf = detection[0]
				imgInfo = coco.loadImgs(imgId)
				h = int(imgInfo[0]['height'])
				w = int(imgInfo[0]['width'])
				# Translate the normalized coords to image coords
				txbbox = []
				txbbox.append(box[0] * w)
				txbbox.append(box[1] * h)
				txbbox.append(box[2] * w)
				txbbox.append(box[3] * h)
				# Wow, do an O(n) check of IOU (TODO is there a faster way?)
				iouPass = False
				# Only use annotations from this image
				annIds = coco.getAnnIds(imgIds=[imgId], catIds=catId)
				anns = coco.loadAnns(annIds)
				# TODO Optimize here by only checking images once
				#log.debug('    Checking against %d annotations from image id %d' % (len(anns), imgId))
				for ann in anns:
					abbox = []
					# COCO bboxes are in the form [x0, y0, w, h], convert:
					abbox.append(ann['bbox'][0])
					abbox.append(ann['bbox'][1])
					abbox.append(ann['bbox'][2] + ann['bbox'][0])
					abbox.append(ann['bbox'][3] + ann['bbox'][1])
					iou = getIou(txbbox, abbox)
					if iou >= thresh:
						iouPass = True
						break
				if iouPass:
					TP = TP + 1
				else:
					FP = FP + 1
				_precision = float(TP) / (TP + FP)
				if N == 0:
					_recall = 0
				else:
					_recall = float(TP) / N
				recall.append(_recall)
				precision.append(_precision)
				log.debug('    [TP %d FP %d N %d] recall: %.5f precision %.5f' % (TP, FP, N, _recall, _precision))
			ap = calcCocoAp(recall, precision)
			log.debug('    = AP for category %d at %.2f is %.2f' % (catId, thresh, ap * 100))
			allAp.append(ap)
	log.debug('Calculate mAP...')
	return np.mean(allAp) * 100

def calcTopN(mi, results):
	'''Compute the Top-N scores for N=1 and 5

	Compare the validation result schema to the ground truth file. The 
	ground truth file is assumed to be a two-column list: input filename and
	classification integer. TopN is defined as the average number of times
	the correct class was found in the top N predictions.

	Args:
		mi (object): modelinfo structure
		results (list): results from the classification run (see report.py)

	Returns:
		(float): Top-1 as a percent
		(float): Top-5 as a percent
		(int): Number of samples, N
	'''
	fn = os.path.join(paths.DATASETS, mi['dataset'], 'annotations', mi['groundtruth'])
	log.debug("Comparing to ground truth %s:" % fn)
	truth = {}
	with open(fn) as fp:
		for line in fp:
			image, index = str.split(line)
			truth[image] = int(index)
	top1 = 0.;
	top5 = 0.;
	N = 0.;
	for (file, scores) in results:
		scores = [int(i) for i in scores]
		path, key = os.path.split(file)
		correct = truth.get(key)
		if len(scores) != 5:
			log.warn("Target did not return 5 predictions for input %s" % file)
		p = 0
		test = correct in scores;
		if test == True:
			top5 = top5 + 1
			p = 5
			if int(scores[0]) == correct:
				top1 = top1 + 1
				p = 1
		log.debug("%s %d in %s %d" % (key, correct, tuple(scores), p))
		N = N + 1
	return [top1 / N * 100, top5 / N * 100, N]

def run(task, results):
	'''Run validation on a task's resullts

	Routes the validation to the correct algorithm based on the model. Each
	function returns its own data schema.

	Args:
		task (object): The JSON defining the task
		result (list): The result data with the scema relevant for the task
	'''
	mi = models.modelinfo[task['workload']]
	if mi['validfunc'] == constants.TOPN:
		return calcTopN(mi, results)
	elif mi['validfunc'] == constants.MAP:
		return calcMap(mi, results)
	else:
		log.warn("No score function written for type '%s'" % mi['validfunc'])
		return None


if __name__ == "__main__":
	# This is a quick hack to run mAP computation offline. Simply store the
	# result JSON schema in a file and supply it as argv[1]
	log.basicConfig(
		format="-%(levelname)s- %(message)s",
		level=log.DEBUG)
	if len(sys.argv) < 2:
		raise Exception('Missing required output JSON file.')
	script, fn = sys.argv
	if not os.path.isfile(fn):
		raise Exception('{} is not a valid file.'.format(fn))
	with open(fn, 'r') as fp:
		results = json.load(fp)
	try:
		if results[0][1][0].get('box'):
			pass
		else:
			raise Exception('Cannot determine validation mode for {}'.format(fn))
	except:
		raise Exception('Cannot determine validation mode for {}'.format(fn))
	# Hack. The output should contain the task, yes?
	mi = models.modelinfo[models.SSDMOBILENET]
	print('mAP is {}'.format(calcMap(mi, results)))

