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
# Original Author: Peter Torelli
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
import tempfile
import subprocess

# Store the path to this module
commonDir = os.path.dirname(__file__)


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
		if value != 1:
			log.error("'%s' must be 1" % param)
			return False
	elif param == const.CONCURRENCY:
		if value != 1:
			log.error("'%s' must be 1" % param)
			return False
	elif param == const.MODE:
		valid = [const.THROUGHPUT, const.ACCURACY, const.LATENCY]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (param, valid))
			return False
	elif param == const.RUNITERATIONS:
		if value < 1:
			log.error("'%s' must be > 0" % param)
			return False
	elif param == const.HARDWARE:
		valid = [const.CPU, const.GPU]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (param, valid))
			return False
	elif param == const.PRECISION:
		valid = [const.FP32, const.FP16]
		if not value in valid:
			log.error("Valid values for '%s' are %s" % (param, valid))
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

	# First make sure the existing parameters are set to valid values
	if checkParams(params) == False:
		return None

	batch = params.get(const.BATCH)
	concurrency = params.get(const.CONCURRENCY)
	mode = params.get(const.MODE)
	runIterations = params.get(const.RUNITERATIONS)
	hardware = params.get(const.HARDWARE)
	precision = params.get(const.PRECISION)

	if batch == None:
		log.warning("'%s' not specified, using '%d'" % (const.BATCH, 1))
		params[const.BATCH] = 1
	if concurrency == None:
		log.warning("'%s' not specified, using '%d'" % (const.CONCURRENCY, 1))
		params[const.CONCURRENCY] = 1
	if precision == None:
		log.warning("'%s' not specified, using '%s'" % (const.PRECISION, const.FP32))
		params[const.PRECISION] = const.FP32
	if mode == None:
		log.warning("'%s' not specified, using '%s'" % (const.MODE, const.THROUGHPUT))
		params[const.MODE] = const.THROUGHPUT
	if runIterations == None:
		log.warning("'%s' not specified, using '%d'" % (const.RUNITERATIONS, 1))
		params[const.RUNITERATIONS] = 1
	if hardware == None:
		log.warning("'%s' not specified, using '%s'" % (const.HARDWARE, const.CPU))
		params[const.HARDWARE] = const.CPU
	if mode == const.ACCURACY:
		if params[const.RUNITERATIONS] != 1:
			log.warning("'%s' forced to '%d' for accuracy" % (const.RUNITERATIONS, 1))
			params[const.RUNITERATIONS] = 1;
	if mode == const.LATENCY or mode == const.ACCURACY:
		if batch and batch > 1:
			log.warning("'%s' forced to '%d' for latency/accuracy" % (const.BATCH, 1))
		params[const.BATCH] = 1
		if concurrency and concurrency > 1:
			log.warning("'%s' forced to '%d' for latency/accuracy" % (const.CONCURRENCY, 1))
		params[const.CONCURRENCY] = 1

	if hardware == const.CPU and precision != const.FP32:
		log.error("'%s' only supports '%s'" % (const.CPU, const.FP32))
		return None

	# TODO: Check if GPU is functional here, rather than failing later.

	return params

def run(modelName, modelFileName, params):
	# TODO careful this over-writes (not appends to) original
	if (params[const.HARDWARE] == 'gpu'):
		os.environ['LD_LIBRARY_PATH'] = commonDir + ':' + os.path.join(commonDir, 'GpuAcc');
	else:
		os.environ['LD_LIBRARY_PATH'] = commonDir + ':' + os.path.join(commonDir, 'CpuAcc');
	tmpdir = tempfile.mkdtemp()
	inputFn = os.path.join(tmpdir, 'inputs.json')
	outputFn = os.path.join(tmpdir, 'outputs.json')
	if params[const.MODE] == const.ACCURACY:
		imageFileNames = getter.apiGetValidationInputs(modelName, cache=True)
	else:
		imageFileNames = getter.apiGetTestInputs(modelName, params[const.BATCH], cache=True)
	cxxParams = {
		'images': imageFileNames, # Note: this can be up to 5,000 filenames
		'model': os.path.join(paths.MODELS, 'tensorflow', modelName, 'frozen_graph.pb'),
		'params': params
	}
	with open(inputFn, 'w') as fp:
		json.dump(cxxParams, fp)
	exeCmd = os.path.join(commonDir, '..', modelName, modelName + '.exe')
	cmd = [
		exeCmd,
		inputFn,
		outputFn
	]
	log.info('Running prediction...')
	log.debug(cmd)
	ret = subprocess.call(cmd)
	if ret:
		log.error('Inference failed')
		return None
	log.info('Loading results file %s' % outputFn)
	with open(outputFn) as fp:
		returnData = json.load(fp)
	if params[const.MODE] == const.ACCURACY:
		return returnData['predictions']
	else:
		return returnData['times']
