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

from .Net import SsdMobilenet

def apiRun(params):
        model = models.SSDMOBILENET
        params = common.pre(model, params)
        if params == None:
                return None, None
        if params[const.MODE] in [const.THROUGHPUT, const.LATENCY]:
                net = SsdMobilenet(os.path.join(paths.MODELS, 'tensorflow_uff', model, 'frozen_graph_ssdmobilenet.uff'),params[const.BATCH],params[const.HARDWARE],params[const.MODE],params[const.PRECISION],params[const.CONCURRENCY])
        else:
                net = SsdMobilenet(os.path.join(paths.MODELS, 'tensorflow_uff', model, 'frozen_graph_ssdmobilenet.uff'),params[const.BATCH],params[const.HARDWARE],params[const.MODE],params[const.PRECISION])
        return params, common.run(model, net, params)
