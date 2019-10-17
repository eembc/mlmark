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
import common
from eelib import models
from eelib import paths

from .Net import ResNet

def apiRun(params):
	model = models.RESNET50
	params = common.pre(model, params)
	if params == None:
		return None, None
	net = ResNet(
		os.path.join(paths.MODELS, 'tensorflow_lite_quantised', model, 'quantised_PTIQ_edgetpu.tflite'))
	return params, common.run(model, net, params)
