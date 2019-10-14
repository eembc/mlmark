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

#import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time
from eelib import constants as const

class SsdMobileNet:

	def __init__(self, frozenGraphFilename, device='cpu'):
		self.boxes = None
		self.scores = None
		self.classes = None
		self.num_detections = None
		#self.interpreter = tf.lite.Interpreter(frozenGraphFilename)
		self.interpreter = Interpreter(frozenGraphFilename)
		self.__load_graph(device)
		self.__init_predictor()

	def __load_graph(self, device):

		# TFLITE INTERPRETER CON.
		#tf.logging.set_verbosity(tf.logging.DEBUG)                
		self.interpreter.allocate_tensors()                              
		
	def __init_predictor(self):
		# obtaining the input-output shapes and types
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()  
		
	def predict(self, images, params, thresh=0, batch=1, warmup=0, iterations=1):
			# Warmup
			x_matrix=np.array(images)
			if params[const.PRECISION] == const.FP32:
				x_matrix=np.array(images, dtype=np.float32)
			
			self.interpreter.set_tensor(self.input_details[0]['index'], x_matrix)
			for i in range(warmup):
				self.interpreter.invoke()
				matrix_0 = self.interpreter.get_tensor(self.output_details[0]['index'])
				matrix_1 = self.interpreter.get_tensor(self.output_details[1]['index'])
				matrix_2 = self.interpreter.get_tensor(self.output_details[2]['index'])
				matrix_3 = self.interpreter.get_tensor(self.output_details[3]['index'])                                
				
			# Measure
			times = []
			for i in range(iterations):
				t0 = time.time()
				self.interpreter.invoke()
				boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
				classes = self.interpreter.get_tensor(self.output_details[1]['index'])
				scores = self.interpreter.get_tensor(self.output_details[2]['index'])
				num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])
				
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
				for d in range(int(num_detections[0])):
					# Note the weird bbox coords: y1, x1, y2, x2 !!
					box = boxes[i][d].tolist()
					x = {
						'score': scores[i][d].tolist(),
						'box': [box[1], box[0], box[3], box[2]],
						'class': (classes[i][d]+1).tolist()
					}
					thisResult.append(x)
				results['predictions'].append(thisResult)
			return results
	def predict_runtime(self, images,params, max=5):
		x_matrix=np.array(images)
		if params[const.PRECISION] == const.FP32:
			x_matrix=np.array(images, dtype=np.float32)

		start=time.time()
		self.interpreter.invoke()
		boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
		classes = self.interpreter.get_tensor(self.output_details[1]['index'])
		scores = self.interpreter.get_tensor(self.output_details[2]['index'])
		num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])
		end = time.time() - start
		return end
