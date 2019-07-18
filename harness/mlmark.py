#!/usr/bin/env python3
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

import glob
import json
import os
import sys
import argparse
import time
import datetime
import logging as log
from future.utils import viewitems
from eelib import paths
from eelib.sysinfo import SysInfo
from eelib.report import Report
from eelib.models import modelinfo

reporter = Report()

# TODO: Hack to force only one type of target per invocation
lastTarget = None

def runTask(task):
	'''Import the target workload package and run the configuration. A task
	contains:
	{ 
		"target": an entry in targets/,
		"workload": a entry in targets/<target>/workloads/
		"params": the run paramters for that target's workload
	}
	Returns: the actual parameters that were run and a results object
	'''
	# TODO document this schema ^^^^^^
	global lastTarget
	targetDir = os.path.join(paths.TARGETS, task['target'])
	pkgDir = os.path.join(targetDir, 'workloads')
	checkDir = os.path.join(pkgDir, task['workload'])
	if not os.path.isdir(checkDir):
		log.error("Path '%s' does not exist" % checkDir)
		return None, None
	# TODO: Hack to force only one type of target per invocation
	if lastTarget is None:
		lastTarget = task['target']
	elif lastTarget != task['target']:
		raise Exception('Only one target is allowed per invocation of MLMark')
	# Since targets might import modules with the same name as other targets,
	# and since python doesn't associate the path of the module, we need to
	# "unload" each target module chain by deleting them from the sys.modules.
	# TODO: For rev2+ come up with a better expansion strategy
	# Dynamically import our local module
	sys.path.append(pkgDir)
	pkg = __import__(task['workload'])
	log.info("Task: Target '%s', Workload '%s'" % (task['target'], task['workload']))
	for k, v in sorted(task['params'].items()):
		log.info("    %-20s : %s" % (k, v))
	finalParams,results = pkg.apiRun(task['params'])
	# Now cleanup. First, remove the search path
	sys.path.remove(pkgDir)
	# ...second, delete every module that was loaded from the search path
	# note: make a copy b/c sys.modules can change at any time
	modulesCopy = {k: v for k,v in viewitems(sys.modules) if v}
	toDel = []
	for module, x in viewitems(modulesCopy):
		if hasattr(x, '__path__'):
			if (type(x.__path__) == list):
				if len(x.__path__) > 0:
					if pkgDir in x.__path__[0]:
						toDel.append(module)
	# Now delete the modules...
	for module in toDel:
		del sys.modules[module]
	return finalParams, results

def parseConfig(fn):
	'''Parses a single config file and attempts to execute the tasks stored
	in it. The config JSON is a list of tasks (see above). Returns nothing.'''
	log.info('Parsing config file %s' % fn)
	with open(fn) as file:
		try:
			cfg = json.load(file)
		except ValueError as e:
			log.warning("%s in %s" % (e, fn))
			return
	for i in range(len(cfg)):
		task = cfg[i]
		ok = True
		for x in ['target', 'workload', 'params']:
			if task.get(x) == None:
				log.warn("Task %d is missing '%s'" % (i + 1, x))
				ok = False
		if not ok:
			continue
		t0 = time.time()
		finalParams, results = runTask(task)
		tn = time.time()
		log.info('Task runtime: %s' % datetime.timedelta(seconds=(tn - t0)))
		# This is a handy way to capture the output results for debug
		debugJson = os.environ.get('MLMARK_OUTPUT_JSON')
		if debugJson is not None:
			try:
				with open(debugJson, "w") as fp:
					json.dump(results, fp, indent=4)
				log.info('Wrote output JSON to %s' % debugJson)
			except:
				log.warning('Failed to write output JSON to %s' % debugJson)
		if results == None:
			log.warn('Task did not succeed')
		else:
			# When reporting, use the final paramters
			task['params'] = finalParams
			reporter.addResults(task, results)

def begin(args):
	'''Scans all of the config/*.cfg files and processes them.'''
	for config in args.config:
		if not os.path.isfile(config):
			log.error('File %s does not exist' % config)
		else:
			parseConfig(config)
	for line in reporter.text():
		log.info(line)

if __name__ == "__main__":
	t0 = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
		help="Print debug info")
	parser.add_argument("-c", "--cfg", dest="config", action="store", nargs='+',
		help="Run a specific configuration file", required=True)
	args = parser.parse_args()
	log.basicConfig(
		format="-%(levelname)s- %(message)s",
		level=(log.INFO,log.DEBUG)[args.verbose])
	log.info("-" * 80)
	log.info("Welcome to the EEMBC MLMark(tm) Benchmark!")
	log.info("-" * 80)
	for kv in SysInfo().get():
		log.info("%-20s : %s" % (kv[0], kv[1]))
	log.info("-" * 80)
	log.info("Models in this release:")
	for k, v in modelinfo.items():
		log.info("    %-15s: %s [%s]" % (k, v['name'], v['dataset']))
	log.info("-" * 80)
	begin(args)
	tn = time.time()
	log.info('Total runtime: %s' % datetime.timedelta(seconds=(tn - t0)))
	log.info('Done')
