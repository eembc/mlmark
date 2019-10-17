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

#from edgetpu.detection.engine import DetectionEngine
from edgetpu.basic.basic_engine import BasicEngine
import numpy as np

class SsdMobileNet:

	def __init__(self, tfLiteFilename):
		#self.engine = DetectionEngine(tfLiteFilename)
		self.basic_engine = BasicEngine(tfLiteFilename)
		
	def predict(self, images, max=5):
		#predictions = self.engine.detect_with_input_tensor(images, threshold=0.3, top_k=10)
		raw= self.basic_engine.run_inference(images)
		raw_split=np.split(raw[1], [40, 50, 60, 61]) #Output is an array with 61 elements. first 40 are bounding box coordinates, next 10 are classes,next 10 are their respective confidence values and last 1 is number of detections)

		#print(raw_split[0]) #boxes
		#print(raw_split[1]) #classes
		#print(raw_split[2]) #confidences/scores
		#print(raw_split[3]) #num of detections
		
		threshold=float(0.3)
		count = (raw_split[2] > threshold).sum() #number of scores above selected threshold

		boxes  =raw_split[0]
		classes=raw_split[1]
		scores =raw_split[2]
		
		results = {
			"predictions": []
		}
		# Currently only support batch size of 1
		for i in range(1):
			thisResult = []
			for d in range(int(count)):
				x = {
					'score': float(scores[d]),
					'box': [float(boxes[4*d+1]), float(boxes[4*d+0]), float(boxes[4*d+3]), float(boxes[4*d+2])],
					'class': int(classes[d]+1)
				}
				thisResult.append(x)
			results['predictions'].append(thisResult)
		return results
