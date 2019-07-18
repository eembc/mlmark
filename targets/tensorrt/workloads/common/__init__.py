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


import logging as log
import os
import sys
import numpy
import time
import json
from common import preprocessor

if sys.version_info[0] < 3:
    raise Exception("This target requires python 3.x")

# Tensorflow 1.13.1 is very chatty. Turn off info...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
		valid = [const.GPU]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.HARDWARE, valid))
			return False
	elif param == const.PRECISION:
		valid = [const.FP32, const.FP16, const.INT8]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (const.PRECISION, valid))
			return False
	elif param == const.CONCURRENCY:
                if value < 1 or value > 32:
                        log.error("Range of '%s' values is [1,32]" % const.CONCURRENCY)
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
	# Check if we have a GPU?
	#local_device_protos = device_lib.list_local_devices()
	#gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
	#if len(gpus) == 0 and params[const.HARDWARE] == 'gpu':
	#	log.error("User requested '%s' but none installed" % const.GPU)
	#	return None
	return params

def runThroughput(modelName, net, params):
	log.info('Running prediction...')
	b = params[const.BATCH];
	fns = getter.apiGetTestInputs(modelName, b)
	tsum = 0
	# Load the entire batch into system/accelerator memory for iterating
	imgs_list=[]
	for i in range(b):
		img = preprocessor.apiProcess(modelName, fns[i])
		imgs_list.append(img)
	imgs_np=numpy.asarray(imgs_list)
	
	results=net.predict(fns[0],imgs_np,5,0,params[const.RUNITERATIONS])
	#tsum=sum(results['seconds'])
	# New schema is [ total_time_s, time0, time1, ..., timeN ]
	return results['seconds']

def runValidation(modelName, net, params):
        imageFiles = getter.apiGetValidationInputs(modelName)
        imageData = [] #empty list
        # Results will be accumulate here
        results = [] #empty list
        t = len(imageFiles) 
        log.info('Running prediction...')
        N = len(imageFiles)
        B = 1 
        for i in range(N):
            data = preprocessor.apiProcess(modelName, imageFiles[i])
            res = net.predict(imageFiles[i],data,5,0,1)
            log.debug("%d predictions left" % (N-(i+1)))
            #for j in range(len(res['predictions'])):
            results.append([imageFiles[i], res['predictions'][0]])
        net.destructor()
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
	elif params[const.MODE] in [const.LATENCY, const.THROUGHPUT]:
		return runThroughput(modelName, net, params)
	else:
		log.critical("Invalid configuration mode '%s'" % params[const.MODE])
		return None
