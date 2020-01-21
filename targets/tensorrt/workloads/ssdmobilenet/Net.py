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

import numpy as np
import time
import numpy.ctypeslib as ctl
import ctypes
import os

from ctypes import cdll
from eelib import paths
from numpy.ctypeslib import ndpointer

TRT_DIR=os.path.join(paths.TARGETS,"tensorrt")

class SsdMobilenet:
    def __init__(self, uffModelName,batchsize, device,mode,precision,concurrency=1):        

        self.batchsize=batchsize
        self.mode = mode
        self.streams = concurrency
        #load library. Initilaize Mobilenet class.
        ssdmobilenet_lib=os.path.join(TRT_DIR,"cpp_environment","libs","libclass_ssdmobilenet.so")
        self.lib = cdll.LoadLibrary(ssdmobilenet_lib)
        self.lib.return_object.restype = ctypes.c_ulonglong
        self.obj = self.lib.return_object()
        if precision == "int8" :
            engine_file_name = "SSDMobilenet_int8_"
        if precision == "fp16" :
            engine_file_name = "SSDMobilenet_fp16_"
        if precision == "fp32" :
            engine_file_name = "SSDMobilenet_fp32_"
        
        cudaDevice = int(os.getenv("cudaDevice")) if "cudaDevice" in os.environ else 0
        engine_file_name=engine_file_name+"B"+str(batchsize)+"_CUDA"+str(cudaDevice)+".engine"
        engine_folder=os.path.join(TRT_DIR,"engines")
        if not os.path.isdir(engine_folder):
            os.mkdir(engine_folder)

        #check if engine file exists. If yes, then use it. If doesnt, create one.
        engine_path = os.path.join(engine_folder,engine_file_name)

        exists=os.path.isfile(engine_path)
        

        if exists:
            #load already created tensorRT engine.
            py_load_engine = self.lib.deserialize_load_trt
            engineName = engine_path.encode('utf-8')
            py_load_engine.argtypes = [ctypes.c_ulonglong,ctypes.c_char_p]
            self.lib.deserialize_load_trt(self.obj,engineName,cudaDevice)
            
        else:
            #create optimized engine.       
            uffName = uffModelName.encode('utf-8') 
            engineName = engine_path.encode('utf-8')
            precision = precision.encode('utf-8')
            py_create = self.lib.create_trt
            py_create.argtypes = [ctypes.c_ulonglong,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p]
            self.lib.create_trt(self.obj,uffName,engineName,batchsize,precision, cudaDevice)
            #load engine
            py_load_engine = self.lib.deserialize_load_trt
            py_load_engine.argtypes = [ctypes.c_ulonglong,ctypes.c_char_p]
            self.lib.deserialize_load_trt(self.obj,engineName,cudaDevice)      
        
        

    def predict(self, images, data, max=5, warmup=0, iterations=1):
        # Measure
        times = []
        out_text_name=images[-16:-4]  #cropping image name. removed path and extension.
        result_folder=os.path.join(TRT_DIR,"Results")
        ssdmobilenet_folder=os.path.join(TRT_DIR,"Results","ssdmobilenet")
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
        if not os.path.isdir(ssdmobilenet_folder):
            os.mkdir(ssdmobilenet_folder)

        if self.mode in ['throughput', 'latency']:
            py_stream = self.lib.infer_stream_trt
            py_stream.argtypes = [
                ctypes.c_ulonglong,
                ndpointer(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ndpointer(ctypes.c_float),
                ctypes.c_bool
                ]
            # Schema: [ totalTime, time0, time1, ... timeN ]
            allTimes = np.ndarray(iterations + 1, dtype=np.float32)
            self.lib.infer_stream_trt(
                self.obj,
                data.astype(np.float32),
                self.batchsize,
                self.streams,
                iterations,
                allTimes,
                False
                )
            for t in allTimes:
                times.append(t)
            #call destructor.
            py_destructor=self.lib.destroy_trt
            py_destructor.argtypes=[ctypes.c_ulonglong]
            self.lib.destroy_trt(self.obj)
        else:
            for i in range(iterations):
                output_text=os.path.join(ssdmobilenet_folder,out_text_name)
                imageName = images.encode('utf-8')
                outTxtName = output_text.encode('utf-8')
                py_inference = self.lib.doInference_trt
                py_inference.argtypes = [ctypes.c_ulonglong,ctypes.c_char_p,ndpointer(ctypes.c_float),ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
                py_inference.restype  = ctypes.c_float
                if self.mode=="accuracy" :
                    ts = self.lib.doInference_trt(self.obj,imageName,data.astype(np.float32),outTxtName,self.batchsize,True)
                               
                
                        
        # Report
        results = {
            "seconds": times,
            "predictions": []}
        
        #for accuracy, read files and fill in variable predictions
        if self.mode=="accuracy" :
            output_text_path=os.path.join(TRT_DIR,"Results","ssdmobilenet",out_text_name)
            output_text_file=output_text_path+ ".txt"            
            boxes=np.zeros([1,100,4])
            num_detections=np.zeros([1])                
            scores=np.zeros([1,100])
            classes=np.zeros([1,100])
            i=int(0)
            with open(output_text_file) as filestream:
                for line in filestream:
                    currentline=line.split(";")
                    scores[0,i]=currentline[1]
                    classes[0,i]=currentline[0]
                    boxes[0,i,0]=float(currentline[2])
                    boxes[0,i,1]=float(currentline[3])
                    boxes[0,i,2]=float(currentline[4])
                    boxes[0,i,3]=float(currentline[5])
                    i=i+1
                                
            num_detections[0]=int(i)
            results = {"seconds": times,"predictions": []}
            for i in range(len(num_detections)):
                thisResult = []
                for d in range(int(num_detections[i])):
                    box = boxes[i][d].tolist()
                    x = {'score': scores[i][d].tolist(),
                         'box': [box[0], box[1], box[2], box[3]],
                         'class': classes[i][d].tolist()
                         }
                    thisResult.append(x)
                results['predictions'].append(thisResult)

        return results            
    def destructor(self):

            py_destructor=self.lib.destroy_trt
            py_destructor.argtypes=[ctypes.c_ulonglong]
            self.lib.destroy_trt(self.obj)

          

            
