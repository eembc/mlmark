import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import glob
import pathlib

tf.compat.v1.enable_eager_execution()

#constants
calib_num = int(200) #Number of calibration images to use. 
cwd = os.getcwd()
HOME_DIR= cwd[:-31]

#Mobilenet
model_path= HOME_DIR +"models/tensorflow/mobilenet/frozen_graph.pb"
input_arrays = ['input']
output_arrays = ["MobilenetV1/Predictions/Reshape_1"]

#Define the converter
converter = tf.lite.TFLiteConverter.from_frozen_graph(model_path, input_arrays, output_arrays, input_shapes={"input":[1,224,224,3]})
converter.optimizations = [tf.lite.Optimize.DEFAULT]

train = []
path = HOME_DIR + 'datasets/ILSVRC2012/images/*.JPEG'

i = 0
#Reading images
for file in glob.glob(path):
    print(file)
    ##preprocessing as tensorflow fp32
    im = cv2.imread(file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR)
    im = im[16:240, 16:240]
    rgb_means = (123.68, 116.78, 103.94)
    data = im - rgb_means
    data = data / 128.
    train.append(data)
    i=i+1
    if(i==calib_num):
        break

train = tf.convert_to_tensor(np.array(train, dtype='float32'))
my_ds = tf.data.Dataset.from_tensor_slices((train)).batch(1)


#POST TRAINING QUANTIZATION with calibration images
def representative_data_gen():    
	for input_value in my_ds.take(calib_num):        
		yield [input_value]

converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("PTIQ")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_quant_file = tflite_models_dir/"qunatized_model_mobilenet.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
