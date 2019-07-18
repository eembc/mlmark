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

from eelib import constants

# TODO2.0: In the next version store this with each model, not in the harness,
# TODO2.0: and build dynamically
# TODO2.0: For example, the in/out tensor names are also dependent on the 
# TODO2.0: model, so the target should not have to guess.

# Add new "official" workload codes here so that everyone is consistent
RESNET50 		= 'resnet50'
MOBILENET 		= 'mobilenet'
SSDMOBILENET 	= 'ssdmobilenet'

# Since EEMBC codes this, there are no constants for the field names
modelinfo = {
	RESNET50: {
		'name': 'ResNet-50 v1.0',
		# This is highly dataset depending
		'dataglob': '*.JPEG',
		'dataset': 'ILSVRC2012',
		'validfunc': constants.TOPN,
		'inputs': 'images',
		'groundtruth': 'validation.txt',
		'units': constants.FPS
	},
	MOBILENET: {
		'name': 'MobileNet v1.0',
		# This is highly dataset depending
		'dataglob': '*.JPEG',
		'dataset': 'ILSVRC2012',
		'validfunc': constants.TOPN,
		'inputs': 'images',
		'groundtruth': 'validation.txt',
		'units': constants.FPS
	},
	SSDMOBILENET: {
		'name': 'SSD-MobileNet v1.0',
		# This is highly dataset depending
		'dataglob': '*.jpg',
		'dataset': 'COCO2017',
		'validfunc': constants.MAP,
		'inputs': 'images',
		'groundtruth': 'instances_val2017.json',
		'units': constants.FPS
	}
}
