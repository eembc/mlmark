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

from eelib import models
from eelib import paths
import glob
import os
import logging as log
import sys
from collections import defaultdict

# Validation debug flag to avoid processing *all* images
debug = os.environ.get('MLMARK_MAXGETTER')
if debug is None:
	debug = 0
else:
	debug = int(debug)

def getClassificationValidationSubset(mi):
	'''ILSVRC2012 has many images, so return 5 from each class.

	Args:
		mi: modelinfo object

	Return:
		list of filneames to images
	'''
	# Learn all categories from ground-truth
	gtfn = os.path.join(paths.DATASETS, mi['dataset'], 'annotations', mi['groundtruth'])
	cats = defaultdict(list)
	with open(gtfn) as fp:
		for line in fp:
			imageFn, i = line.strip().split()
			catId = int(i)
			cats[catId].append(imageFn)
	files = []
	# Select 5 from each category/class
	for i in cats:
		cats[i].sort()
		for j in range(5):
			# Full path
			fn = os.path.join(paths.DATASETS, mi['dataset'], mi['inputs'], cats[i][j])
			if os.path.isfile(fn):
				files.append(fn)
			else:
				raise Exception("File %s does not exist" % fn)
	return files

def getInputs(modelName, count, cache=False):
	'''Return a list of input image filenames, and maybe pre-process them.

	Creates a list of input images from the model's dataset folder according
	to the input glob for that dataset. If the 'path' arg is set, processed
	images of the file will be saved to that path (the path must exist).

	Args:
		modelName (text): Name key of the model in the models.modelinfo
		count (int): Number of images to return (Use -1 for all images)
		path (path): If set (path must exist) store a copy of the processed file

	Returns:
		(list): List of fully-qualified filenames
	'''
	mi = models.modelinfo.get(modelName)
	if mi == None:
		log.critical("Invalid model '%s' passed to input getter" % modelName)
		sys.exit(1)
	if (modelName == models.SSDMOBILENET):
		# For SSDMOBILENET just return every image
		files = glob.glob(os.path.join(paths.DATASETS, mi['dataset'], mi['inputs'], mi['dataglob']))
	else:
		# For RESNET/MOBILENET, return 5,000 images (5 of each category)
		files = getClassificationValidationSubset(mi);
	# Always make sure files are sorted to avoid O/S randomness issues
	files = sorted(files)
	# Truncate the file list if user only needs a subset
	if count > 0:
		files = files[:count]
	if debug and count < 0:
		files = files[:debug]
	if len(files) == 0:
		log.critical("No files found for dataset '%s'" % mi['dataset'])
		sys.exit(1)
	log.info("Acquired %d file(s) for model '%s'" % (len(files), mi['name']))
	if len(files) > 1000:
		log.info('(Note: this many inputs may take a long time to complete.)')
	return files

def apiGetValidationInputs(modelName, cache=False):
	'''In the future this may only select a subset of classes'''
	return getInputs(modelName, -1, cache)

def apiGetTestInputs(modelName, count, cache=False):
	return getInputs(modelName, count, cache)
