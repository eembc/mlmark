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

import tensorflow as tf
import numpy as np
import time

class SsdMobileNet:

	def __init__(self, frozenGraphFilename, device='cpu'):
		self.frozenGraphFilename = frozenGraphFilename
		self.graph = None
		self.session = None
		self.input = None
		self.boxes = None
		self.scores = None
		self.classes = None
		self.num_detections = None
		self.__load_graph(device)
		self.__init_predictor()

	def __load_graph(self, device):
		with tf.gfile.GFile(self.frozenGraphFilename, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
		with tf.Graph().as_default() as graph:
			# force prefix to "" using name=
			tf.import_graph_def(graph_def, name="")
		self.graph = graph
		if device == 'cpu':
			config = tf.ConfigProto(device_count={'GPU': 0})
			self.session = tf.Session(config=config, graph=self.graph)
		else:
			self.session = tf.Session(graph=self.graph)

		self.graph = graph
		self.session = tf.Session(graph=self.graph)
	
	def __init_predictor(self):
		self.input = self.graph.get_tensor_by_name('image_tensor:0')
		self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
		self.scores = self.graph.get_tensor_by_name('detection_scores:0')
		self.classes = self.graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

	def predict(self, images, thresh=0, batch=1, warmup=0, iterations=1):
		with self.graph.as_default():
			# Warmup
			for i in range(warmup):
				(scores, classes, num_detections, boxes) = self.session.run(
					[self.scores, self.classes, self.num_detections, self.boxes],
					{
						self.input: images
					})
			# Measure
			times = []
			for i in range(iterations):
				t0 = time.time()
				(scores, classes, num_detections, boxes) = self.session.run(
					[self.scores, self.classes, self.num_detections, self.boxes],
					{
						self.input: images
					})
				t1 = time.time()
				ts = t1 - t0
				times.append(ts)
			# Report
			results = {
				"seconds": times,
				"predictions": []
			}
			for i in range(len(num_detections)):
				thisResult = []
				for d in range(int(num_detections[i])):
					# Note the weird bbox coords: y1, x1, y2, x2 !!
					box = boxes[i][d].tolist()
					x = {
						'score': scores[i][d].tolist(),
						'box': [box[1], box[0], box[3], box[2]],
						'class': classes[i][d].tolist()
					}
					thisResult.append(x)
				results['predictions'].append(thisResult)
			return results
