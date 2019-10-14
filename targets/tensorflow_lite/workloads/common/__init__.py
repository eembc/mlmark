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

import sys

if sys.version_info[0] < 3:
	raise Exception("This target requires python 3.x")

from eelib import models
from eelib import getter
from eelib import paths
from eelib import constants as const

from . import preprocessor

import logging as log
import os
import numpy
import time
import json
import tempfile


# Note: Putting checks in JSON forces limits on the dataType of the value. Hence,
# it was pulled from a specification file into python code. It is up to the
# developer to ultimately code parameter checking, but it should be very clear
# to the user what is wrong. This file is just an example of how it could be
# done.

# NOTE: The params field must be COMPLETELY filled in

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
		valid = [const.CPU, const.GPU]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.HARDWARE, valid))
			return False
	elif param == const.PRECISION:
		valid = [const.FP32,const.INT8]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.PRECISION, valid))
			return False
	elif param == const.CONCURRENCY:
		if value != 1:
			log.error("This target only supports concurrency of 1")
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
	concurrency = params.get(const.CONCURRENCY)
	mode = params.get(const.MODE)
	runIterations = params.get(const.RUNITERATIONS)
	hardware = params.get(const.HARDWARE)
	precision = params.get(const.PRECISION)

	if precision == None:
		log.warning("'%s' not specified, using '%s'" % (const.PRECISION, const.FP32))
		params[const.PRECISION] = const.FP32
	if concurrency is None or concurrency != 1:
		log.warning("'%s' not specified, using '%d'" % (const.CONCURRENCY, 1))
		params[const.CONCURRENCY] = 1
	if mode == None:
		log.warning("'%s' not specified, using '%s'" % (const.MODE, const.THROUGHPUT))
		params[const.MODE] = const.THROUGHPUT
	if runIterations == None:
		log.warning("'%s' not specified, using '%d'" % (const.RUNITERATIONS, 1))
		params[const.RUNITERATIONS] = 1
	if mode == const.ACCURACY:
		if params[const.RUNITERATIONS] != 1:
			log.warning("'%s' forced to '%d' for accuracy" % (const.RUNITERATIONS, 1))
			params[const.RUNITERATIONS] = 1;
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
	log.info('Running prediction...')
	fns = getter.apiGetTestInputs(modelName, 1)
	if params[const.PRECISION] == const.FP32:
		img = preprocessor.apiProcess(modelName, fns[0])
	if params[const.PRECISION] == const.INT8:
		img = preprocessor.apiProcess_int8(modelName, fns[0])
	times = []
	for j in range(params[const.RUNITERATIONS]):
		t0 = time.time()
		res = net.predict([img])
		tn = time.time()
		t = tn - t0
		log.debug('%f sec' % t)
		times.append(t)
	return times

def runThroughput(modelName, net, params):
	log.info('Running prediction...')
	b = params[const.BATCH];
	fns = getter.apiGetTestInputs(modelName, b)
	tsum = 0
	# Load the entire batch into system/accelerator memory for iterating
	imgs = []
	for i in range(b):
		if params[const.PRECISION] == const.FP32:
			img = preprocessor.apiProcess(modelName, fns.pop())
		if params[const.PRECISION] == const.INT8:
			img = preprocessor.apiProcess_int8(modelName, fns.pop())
		imgs.append(img)
	# Allow one warmup prediction outside the overall timing loop
	times = []
	t=net.predict_runtime(imgs,params)
	times.append(t)
	# Overall timing loop for throughput
	t_start = time.time()
	for i in range(params[const.RUNITERATIONS]):
		# individual timing for latency; we don't support concurrency > 1
		t=net.predict_runtime(imgs,params)
		times.append(t)
	t_finish = time.time()
	times.insert(0, t_finish - t_start)
	return times

def runValidation(modelName, net, params):
	'''Process all of the validation inputs from the apiGet* function
	
	This is a reference example. Batch size can be set to whatever is optimal.
	The important things to note are: [a] inputs come from the getter, and [b]
	the results are returned in the form of a list of lists defined in the
	report.py comments, depending on the workload/model.
	
	Note: This function returns its value upstream to the harness through the
	apiRun() API function. It is only an example, apiRun() can be implemented
	however a developer chooses.'''
	imageFiles = getter.apiGetValidationInputs(modelName)
	imageData = []
	# Results will be accumulate here
	results = []
	t = len(imageFiles)
	# Feeding 10 at a time
	log.info('Running prediction...')
	N = len(imageFiles)
	B = 1 # resnet is now batch 1?
	m = 0
	for i in range(t):
		if params[const.PRECISION] == const.FP32:
			data = preprocessor.apiProcess(modelName, imageFiles[i])
		if params[const.PRECISION] == const.INT8:
			data = preprocessor.apiProcess_int8(modelName, imageFiles[i])
		imageData.append(data)
		# Batch size of B; TODO test for one-off fail
		if len(imageData) == B or (i+1) == t:
			res = net.predict(imageData,params)
			log.debug("%d predictions left" % (N-(i+1)))
			imageData = []
			for j in range(len(res['predictions'])):
				results.append([imageFiles[m], res['predictions'][j]])
				m = m + 1
	# For debugging
	tmpdir = tempfile.mkdtemp()
	outputFn = os.path.join(tmpdir, 'outputs.json')
	with open(outputFn, 'w') as fp:
		json.dump(results, fp)
	log.debug('Validation output file is {}'.format(outputFn))
	#print(results)
	return results

# Need to validate parameters before creating the network
def pre(modelName, params):
	#if tf.__version__ != '1.13.1':
	#	log.warning("This target was tested on tensorflow-1.31.1, you are using %s "
	#		% tf.__version__)
	return validateParams(params)

def run(modelName, net, params):
	# Keep the exit point of the module the apiRun() function
	if params == None:
		log.warning("No valid parameters for this run")
		return None
	log.debug("Final params: %s" % json.dumps(params))
	if params[const.MODE] == const.ACCURACY:
		return runValidation(modelName, net, params)
	elif params[const.MODE] in [const.THROUGHPUT, const.LATENCY]:
		return runThroughput(modelName, net, params)
	else:
		log.critical("Invalid configuration mode '%s'" % params[const.MODE])
		return None
