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

'''

Here are the schema expected from the workload apiRun functions:

Throughput
----------
Throughput is the total time to run all requests divided by the # of requests
[
	time:	Total time in seconds to run all requests,
	inference_0: time in seconds for inference 0,
	inference_1: time in seconds for inference 1,
	inference_2: time in seconds for inference 2,
	:
	inference_N: time in seconds for inference N,

]

TopN Accuracy
-------------
A list where each element is a list containing a filename (the input filename
provide by the getter) and a list of the topN class id predictiosn, where 
prediction[0] is the highest ranked.

[
	[
		image_fn,
		[ 
			predicted class ID 1,
			predicted class ID 2,
			...
			predicted class ID N,
		]
	]+
]

mAP Accuracy
------------
Similar to TopN accuracy, but each prediction consists of a list of detections
that are objects containing "box", "score" and "class":

[
	[
		image_fn,
		[ 
			{
				"class": class id,
				"box": [x0, y0, x1, y1] normalized to [0,1]
				"score": predicted score
			}+
		]
	]+
]
'''

from eelib import constants as const
from eelib import validate
from eelib.models import modelinfo as mi
import numpy as np
import logging as log
import json

def floatFormat(n):
	return np.format_float_positional(
		n,
		precision=3,
		fractional=False)

class Report:
	def __init__(self):
		self.ledger = []

	def text(self, format="txt"):
		if len(self.ledger) == 0:
			return []
		text = []
		if format == "csv":
			fmt = "%s,%s,%s,%s,%s,%s,%s,%s,%s"
		else:
			fmt = "%-15s %-15s %-5s %-5s %5s %5s %-10s %10s %5s"
		text.append(fmt %
			(
				"Target",
				"Workload",
				"H/W",
				"Prec",
				"Batch",
				"Conc.",
				"Metric",
				"Score   ",
				"Units"
			))
		text.append('-' * 83)
		for item in self.ledger:
			text.append(fmt %
				(
					item['target'],
					item['workload'],
					item['hardware'],
					item['precision'],
					item['batch'],
					item['concurrency'],
					item['metric'],
					np.format_float_positional(item['score'], precision=3,
						pad_left=5, pad_right=4, fractional=False),
					item['units']
					))
		return text

	def addResults(self, task, results):
		if results == None:
			log.critical("Target did not generate results for task")
			return
		if task['params'][const.MODE] in [const.THROUGHPUT, const.LATENCY]:
			# Cast as floats in case they come in as strings
			results = [float(i) for i in results]
			log.debug(results)
			batch = task['params'][const.BATCH]
			iterations = task['params'][const.RUNITERATIONS]
			total_sec = np.sum(results[0]);
			ips = (iterations * batch) / float(total_sec)
			# Units of throughput depend on the model
			log.info("Throughput: %s %s" % (floatFormat(ips), mi[task['workload']]['units']))
			self.ledger.append({
				"target": task['target'],
				"metric": "throughput",
				"workload": task['workload'],
				"precision": task['params'][const.PRECISION],
				"hardware": task['params'][const.HARDWARE],
				"batch": task['params'][const.BATCH],
				"concurrency": task['params'][const.CONCURRENCY],
				"iterations": task['params'][const.RUNITERATIONS],
				"score": ips,
				"units": mi[task['workload']]['units']
				})
			if len(results) > 1:
				# Since lower is better for latency, extract the max of the 
				# bottom 5%, instead the min of the top 95%
				perc = const.PERFORMANCE_PERCENTILE * 100
				latency_sec = np.percentile(results[1:], perc)
				# Latency is always milliseconds
				ms = latency_sec * 1000
				log.info("Latency: %s ms" % floatFormat(ms))
				log.debug("Latency StdDev: %f (s)" % np.std(results[1:]))
				self.ledger.append({
					"target": task['target'],
					"metric": "latency",
					"workload": task['workload'],
					"precision": task['params'][const.PRECISION],
					"hardware": task['params'][const.HARDWARE],
					"batch": task['params'][const.BATCH],
					"concurrency": task['params'][const.CONCURRENCY],
					"iterations": task['params'][const.RUNITERATIONS],
					"score": ms,
					"units": 'ms'
					})
		elif task['params'][const.MODE] == const.ACCURACY:
			# vres for map = [ map ]
			# vres for topn = [ top1, top5 ]
			vres = validate.run(task, results)
			if vres == None:
				# Error already generated in validate.riun()
				return
			if mi[task['workload']]['validfunc'] == const.TOPN:
				log.info("Accuracy results:")
				log.info("    Top-1 : % 5.1f %%" % vres[0])
				log.info("    Top-5 : % 5.1f %%" % vres[1])
				log.info("    N     : % 5d" % vres[2])
				self.ledger.append({
					"target": task['target'],
					"metric": "accuracy",
					"workload": task['workload'],
					"precision": task['params'][const.PRECISION],
					"hardware": task['params'][const.HARDWARE],
					"batch": task['params'][const.BATCH],
					"concurrency": task['params'][const.CONCURRENCY],
					"iterations": task['params'][const.RUNITERATIONS],
					"score": vres[0],
					"units": "Top1"
					})
				self.ledger.append({
					"target": task['target'],
					"metric": "accuracy",
					"workload": task['workload'],
					"precision": task['params'][const.PRECISION],
					"hardware": task['params'][const.HARDWARE],
					"batch": task['params'][const.BATCH],
					"concurrency": task['params'][const.CONCURRENCY],
					"iterations": task['params'][const.RUNITERATIONS],
					"score": vres[1],
					"units": "Top5"
					})
			else:
				log.info("Accuracy results:")
				log.info("    mAP   : % 5.1f" % vres)
				self.ledger.append({
					"target": task['target'],
					"metric": "accuracy",
					"workload": task['workload'],
					"precision": task['params'][const.PRECISION],
					"hardware": task['params'][const.HARDWARE],
					"batch": task['params'][const.BATCH],
					"concurrency": task['params'][const.CONCURRENCY],
					"iterations": task['params'][const.RUNITERATIONS],
					"score": vres,
					"units": "mAP"
					})

