import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import glob
import pathlib

tf.enable_eager_execution()

#constants
calib_num = int(200) #Number of calibration images to use. 
cwd = os.getcwd()
HOME_DIR= cwd[:-31]

#Resnet50
model_path= HOME_DIR +"models/tensorflow/resnet50/frozen_graph.pb"
input_arrays = ["input"]
output_arrays = ["resnet_v1_50/SpatialSqueeze"]

#define the converter
converter = tf.lite.TFLiteConverter.from_frozen_graph(model_path, input_arrays, output_arrays, input_shapes={"input":[1,224,224,3]})
converter.optimizations = [tf.lite.Optimize.DEFAULT]

train = []
path = HOME_DIR + 'datasets/ILSVRC2012/images/*.JPEG'

i=0
#Read calibration images
for file in glob.glob(path):
    print(file)
    #Preprocessing the images
    im = cv2.imread(file)    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR)
    im = im[16:240, 16:240]   
    rgb_means = (123.68, 116.78, 103.94)    
    data = im - rgb_means
    train.append(data)
    i=i+1
    if(i==calib_num):
        break

train = tf.convert_to_tensor(np.array(train, dtype='float32'))
my_ds = tf.data.Dataset.from_tensor_slices((train)).batch(1)

#POST TRAINING QUANTIZATION with calibration image
def representative_data_gen():
    for input_value in my_ds.take(calib_num):
        yield [input_value]

converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("PTIQ")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_quant_file = tflite_models_dir/"quantized_model_resnet50.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
