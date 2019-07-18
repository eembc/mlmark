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

from .Net import SSDMOBILENET

SSDMOBILENET_DIR=os.path.join(paths.TARGETS,"openvino_ubuntu","workloads","ssdmobilenet")

if os.path.exists(os.path.join(SSDMOBILENET_DIR,"ssd_detection_output.csv")):
    os.remove(os.path.join(SSDMOBILENET_DIR,"ssd_detection_output.csv"))

def apiRun(params):
	model = models.SSDMOBILENET
	params = common.pre(model, params)

	modelFileName='ssdmobilenet_'+params[const.PRECISION] + '.xml'
    
	if params == None:
		return None, None
	if params[const.HARDWARE] == const.GPU:
		modelFileName='ssdmobilenet_'+params[const.PRECISION]+'_b'+str(params[const.BATCH])+'.xml'
		if os.path.exists(modelFileName):
			print("Cannot find model file {}. Please ensure an IR was generated for the requested batch size and precision during compilation.".format(modelFileName))
			sys.exit(1)
	else:
		if os.path.exists(modelFileName):
			print("Cannot find model file {}. Please ensure an IR was generated for the requested precision during compilation.".format(modelFileName))
			sys.exit(1)

	if params[const.MODE] in [const.THROUGHPUT, const.LATENCY]:
		net = SSDMOBILENET(
			os.path.join(paths.MODELS, 'openvino', model, modelFileName),
			params[const.HARDWARE], params[const.PRECISION], params[const.MODE], params[const.BATCH], params[const.CONCURRENCY])
	else:
		net = SSDMOBILENET(
			os.path.join(paths.MODELS, 'openvino', model,modelFileName),
			params[const.HARDWARE], params[const.PRECISION], params[const.MODE], params[const.BATCH] )
	return params, common.run(model, net, params)
