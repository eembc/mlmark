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
from eelib import constants as const

from .Net import ResNet50

def apiRun(params):
	model = models.RESNET50
	params = common.pre(model, params)
	if params == None:
		return None, None
	if params[const.MODE] in [const.THROUGHPUT, const.LATENCY]:
		net = ResNet50(
			os.path.join(paths.MODELS, 'openvino', model, 'resnet-50_'+params[const.PRECISION]+'.xml'),
			params[const.HARDWARE], params[const.PRECISION], params[const.MODE], params[const.BATCH], params[const.CONCURRENCY])
	else:
		net = ResNet50(
			os.path.join(paths.MODELS, 'openvino', model, 'resnet-50_'+params[const.PRECISION]+'.xml'),
			params[const.HARDWARE], params[const.PRECISION], params[const.MODE], params[const.BATCH] )
	return params, common.run(model, net, params)
