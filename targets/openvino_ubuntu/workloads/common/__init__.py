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
from eelib import getter
from eelib import paths
from eelib import constants as const
from progress.bar import Bar

import logging as log
import os
import sys
import numpy
import time
import json

# Note: Putting checks in JSON forces limits on the dataType of the value. Hence,
# it was pulled from a specification file into python code. It is up to the
# developer to ultimately code parameter checking, but it should be very clear
# to the user what is wrong. This file is just an example of how it could be
# done.

# NOTE: The params field must be COMPLETELY filled in
VALID_PARAMS={
const.CPU : [const.FP32, const.INT8],
const.GPU : [const.FP32, const.FP16],
const.VPU : [const.FP16],
const.HDDL : [const.FP16]
}

def checkParam(param, value):
	'''Checks a paramater against known values for this hardware. Rather
	than put the supported features in a JSON file, this forces the developer
	to rely on constants and provide more succinct detail. Returns True if
	the value is correct for the param.'''

	if param == const.BATCH:
		if value < 1 or value > 256:
			log.error("Range of '%s' values is [1, 256]" % const.BATCH)
			return False
	elif param == const.MODE:
		valid = [const.THROUGHPUT, const.ACCURACY, const.LATENCY]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.MODE, valid))
			return False
	elif param == const.RUNITERATIONS:
		if value < 1:
			log.error('Iterations must be > 0')
			return False
	elif param == const.HARDWARE:
		valid = [const.CPU, const.GPU, const.HDDL, const.VPU]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.HARDWARE, valid))
			return False
	elif param == const.PRECISION:
		valid = [const.FP32, const.FP16, const.INT8]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.PRECISION, valid))
			return False
	elif param == const.CONCURRENCY:
		if value < 1 or value > 256:
			log.error("Range of '%s' values is [1, 256]" % const.CONCURRENCY)
			return False
	else:
		log.error("Unknown parameter '%s'" % param)
		return False
	return True

def checkParams(params):
	'''Checks all parameters in the config to see if they are valid. Does
	NOT check for consistency, just validity. Returns True or False.'''

	status = True
	for k, v in params.items():
		# Don't exit on the first bad param, check them all for the user
		if checkParam(k, v) == False:
			status = False
	return status

def validateParams(params):
	'''This function makes sure the input parameters are consistent for the
	type of scenario requested. Where checkParams() validates the values,
	this function checks for logicial consistency. Returns a modified 'params'
	structure that should replace the input 'params'.'''

	# First make sure the parameters are set to valid values
	if checkParams(params) == False:
		return None

	batch = params.get(const.BATCH)
	mode = params.get(const.MODE)
	runIterations = params.get(const.RUNITERATIONS)
	hardware = params.get(const.HARDWARE)
	precision = params.get(const.PRECISION)
	concurrency = params.get(const.CONCURRENCY)
	
	if not precision in VALID_PARAMS[hardware]:
		log.warning("'%s' is not supported on '%s'" % (const.PRECISION, hardware))
		return None
	if precision == None:
		log.warning("'%s' not specified, using '%s'" % (const.PRECISION, const.FP32))
		params[const.PRECISION] = const.FP32
	if mode == None:
		log.warning("'%s' not specified, using '%s'" % (const.MODE, const.THROUGHPUT))
		params[const.MODE] = const.THROUGHPUT
	if concurrency == None:
		log.warning("'%s' not specified, using '%d'" % (const.CONCURRENCY, 1))
		params[const.CONCURRENCY] = 1
	if runIterations == None:
		log.warning("'%s' not specified, using '%d'" % (const.RUNITERATIONS, 1))
		params[const.RUNITERATIONS] = 1
	if mode == const.LATENCY or mode == const.ACCURACY:
		if batch and batch > 1:
			log.warning("'%s' forced to '%d' for latency/accuracy" % (const.BATCH, 1))
		params[const.BATCH] = 1
	elif mode == const.THROUGHPUT:
		if batch == None:
			log.warning("'%s' not specified, using '%d'" % (const.BATCH, 1))
			params[const.BATCH] = 1
	if hardware == None:
		log.warning("'%s' not specified, using '%s'" % (const.HARDWARE, const.CPU))
		params[const.HARDWARE] = const.CPU
	return params

def runLatency(modelName, net, params):
	log.info('Running latency...')
	fns = getter.apiGetTestInputs(modelName, 1, cache=True)
	times = net.predict(fns, params[const.RUNITERATIONS])
	return times

def runThroughput(modelName, net, params):
	log.info('Running throughput...')
	fns = getter.apiGetTestInputs(modelName, params[const.BATCH], cache=True)
	times = net.predict(fns, params[const.RUNITERATIONS])
	return times 

def runValidation(modelName, net, params):
	'''Feed the specified images into the model at any batch size; and return
	an array of the image filename and the top-5 predictions'''
	imageData = []
	# Results will be accumulate here
	results = []
	imageFiles = getter.apiGetValidationInputs(modelName, cache=True)
	t = len(imageFiles)
	if modelName == models.MOBILENET or modelName == models.RESNET50:
		log.info('Running prediction...')
		bar = Bar('Running prediction...', max=t)
		for i in range(t):
			single_result = net.predict([imageFiles[i]], 1)
			fn=single_result[0]
			single_result_list=single_result[1:]
			results.append([fn,single_result_list])
			bar.next()
		bar.finish()
	elif modelName == models.SSDMOBILENET:
		from . import parse_ssd_output
		bar = Bar('Running prediction...', max=t)
		for i in range(t):
			out = net.predict([imageFiles[i]],1)
			bar.next()
		bar.finish()

		output_path=os.path.join(paths.TARGETS,"openvino_ubuntu","workloads","ssdmobilenet","ssd_detection_output.csv")
		results = parse_ssd_output.parse_ssd_detection(output_path)

	return results

# Need to validate parameters before creating the network
def pre(modelName, params):
	return validateParams(params)

def run(modelName, net, params):
	# Keep the exit point of the module the apiRun() function
	if params == None:
		log.warning("No valid parameters for this run")
		return None
	log.debug("Final params: %s" % json.dumps(params))
	if params[const.MODE] == const.ACCURACY:
		return runValidation(modelName, net, params)
	elif params[const.MODE] in [ const.THROUGHPUT, const.LATENCY ]:
		return runThroughput(modelName, net, params)
	else:
		log.critical("Invalid configuration mode '%s'" % params[const.MODE])
		return None
