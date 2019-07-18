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
import inspect
import logging as log

# Common paths used by everyone.
HARNESS = os.path.split(
	os.path.dirname(
		os.path.abspath(inspect.getfile(inspect.currentframe()))
	))[0]
HOME = os.path.split(HARNESS)[0]
CONFIG = os.path.join(HOME, 'config')
TARGETS = os.path.join(HOME, 'targets')
MODELS = os.path.join(HOME, 'models')
RESULTS = os.path.join(HOME, 'results')
DATASETS = os.path.join(HOME, 'datasets')
CACHE = os.path.join(DATASETS, 'cache')

import os

def appendPath (target, add):
	if not os.path.isdir(add):
		log.error("Invalid add path %s" % add)
		return
	current = os.environ.get(target)
	if current is None:
		os.environ[target] = add
	else:
		parts = current.split(":")
		do_add = True
		for part in parts:
			if part == add:
				log.warning("Path %s already exists in %s" % (add, target))
				do_add = False
				break
		if do_add:
			current = current + ':' + add
			os.environ[target] = current
			log.info("Added %s to %s (%s)" % (add, target, current))

