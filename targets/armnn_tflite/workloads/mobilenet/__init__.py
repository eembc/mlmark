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

import os
import common
import logging as log
from eelib import models
from eelib import paths
from eelib import constants as const

def apiRun(params):
	model = models.MOBILENET
	params = common.validateParams(params)
	if params == None:
		return None, None
	# FP32/FP16 use the TF->TFLite converted model
	mfw = 'tensorflow_lite'
	mfn = 'converted_model.tflite'
	# If INT8, switch to the PTIQ quantized model
	if params[const.PRECISION] == const.INT8:
		mfw = 'tensorflow_lite_quantised'
		mfn = 'quantised_PTIQ.tflite'
	fullpath = os.path.join(paths.MODELS, mfw, model, mfn);
	log.debug('Model location: %s' % fullpath)
	return params, common.run(model, fullpath, params)
