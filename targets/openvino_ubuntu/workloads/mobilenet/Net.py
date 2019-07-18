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
import platform
import subprocess
import shutil
import time
from eelib import paths
from eelib import constants

UBUNTU_VERSION_18="18"
UBUNTU_VERSION_16="16"
SUCCESS_CODE=0
OPENVINO_DIR=os.path.join(paths.TARGETS,"openvino_ubuntu")
MOBILENET_DIR=os.path.join(OPENVINO_DIR,"workloads","mobilenet")
COMMON_DIR=os.path.join(OPENVINO_DIR,"workloads","common")

class MobileNet:
	def __init__(self, frozenGraphFilename, device='cpu', precision='precision', mode='latency', batch_size=1, concurrent_instances=1):
		self.modelFilename = frozenGraphFilename
		self.device = device
		self.precision = precision
		self.mode = mode
		self.batch_size = batch_size
		self.concurrent_instances = concurrent_instances
		self.set_env_path()

	def set_env_path(self):

                os.environ['LD_LIBRARY_PATH']=os.path.join(OPENVINO_DIR,'lib')

                if not 'HDDL_INSTALL_DIR' in os.environ.keys():
                        openvino_version_file = os.path.join(paths.HARNESS,"harness","OpenVINO_BUILD.txt")
                        if os.path.isfile(openvino_version_file):
                                with open(openvino_version_file,'r') as fid:
                                        text = fid.read().splitlines()
                                        OpenVINO_PATH = text[0].split(': ')[-1]
                                os.environ['HDDL_INSTALL_DIR']= os.path.join(OpenVINO_PATH,"deployment_tools","inference_engine","external","hddl")
                        else:
                                 os.environ['HDDL_INSTALL_DIR']="/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/external/hddl"

	def predict(self, image_fns, iterations=1000, max=5):
		# Output file to write c++ logs
		file_out = os.path.join(MOBILENET_DIR,"out.txt")

		# Checker to see if preprocessor sent the right batch
		if not self.mode == constants.ACCURACY:
			if not len(image_fns) == self.batch_size:
				print("[ERROR] Batch-size mismatch")

		# Creating a batch folder as EXE takes image folder with # = batch size as argument
		batch_folder=MOBILENET_DIR+"/batch"+str(self.batch_size)
		if os.path.isdir(batch_folder):
			shutil.rmtree(batch_folder)
		os.mkdir(batch_folder)
		for fn in image_fns:
			shutil.copy(fn,batch_folder)

		#Setting up commands to c++ executable
		application = os.path.realpath(COMMON_DIR+"/image_classification_async")
		result_file_path=MOBILENET_DIR+"/result_"+time.strftime('%a%H%M%S')+".txt"
		command = application + " -b "+str(self.batch_size)+" -prec "+self.precision+" -d "+self.device.upper()+" -i "+batch_folder+" -m "+self.modelFilename+" -nt "+str(max)+" -ni "+str(iterations)+" -nireq "+str(self.concurrent_instances)+" -r "+result_file_path+" -mode "+self.mode+" -cl 1001" + " -a " + "mobilenet"

		# Run Inference
		with open(file_out, 'w') as out:
			self.return_code = subprocess.call(command, stdout=out, shell=True)

		# Post processing reading the output file for predictions or latencies
		result=[]
		if self.return_code == SUCCESS_CODE:
			for line in open(result_file_path):
				if not self.mode == constants.ACCURACY:
	    				result = [float(el) for el in line.split()]
				else:
					result = [el for el in line.split()]
		# Clean the batch folder
		shutil.rmtree(batch_folder)
		os.remove(result_file_path)
		return result


