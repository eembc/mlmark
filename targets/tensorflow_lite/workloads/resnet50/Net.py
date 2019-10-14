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

class ResNet50:

	def __init__(self, frozenGraphFilename, device='cpu'):
                #self.interpreter = tf.lite.Interpreter(frozenGraphFilename)
                self.interpreter = Interpreter(frozenGraphFilename)
                self.__load_graph(device)
                self.__init_predictor()

	def __load_graph(self, device):
                # TFLite Interpreter con
                #tf.logging.set_verbosity(tf.logging.DEBUG)
                self.interpreter.allocate_tensors()
                
	def __init_predictor(self):
                # obtaining the input-output shapes and types
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

	def predict(self, images,params, max=5):
		x_matrix=np.array(images)
		if params[const.PRECISION] == const.FP32:
			x_matrix=np.array(images, dtype=np.float32)
		#if params[const.PRECISION] == const.INT8:
		
		self.interpreter.set_tensor(self.input_details[0]['index'], x_matrix)
		self.interpreter.invoke()
		predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
		results = {'predictions': []}
		for i in range(len(predictions)):
			ordered = predictions[i].argsort()[-len(predictions[i]):][::-1]
			topn = []
			for j in ordered[:max]:
				# convert numpy.int64 to int for JSON serialization later
				topn.append(int(j))
			results['predictions'].append(topn)
			return results
	def predict_runtime(self, images,params, max=5):
		x_matrix=np.array(images)
		if params[const.PRECISION] == const.FP32:
			x_matrix=np.array(images, dtype=np.float32)

		start=time.time()
		self.interpreter.set_tensor(self.input_details[0]['index'], x_matrix)
		self.interpreter.invoke()
		predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
		end = time.time() - start
		return end
