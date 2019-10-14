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

##MobilenetSSD
model_path=HOME_DIR +"models/tensorflow/ssdmobilenet/frozen_graph_intermediate.pb" # This is not straightforward. Kindly have a look at readme file in theis utility folder.
input_arrays = ["normalized_input_image_tensor"]
output_arrays = [ 'TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']

#define the optimizer
converter = tf.lite.TFLiteConverter.from_frozen_graph(model_path, input_arrays, output_arrays, input_shapes={"normalized_input_image_tensor":[1,300,300,3]})
converter.optimizations = [tf.lite.Optimize.DEFAULT]

train = []
path = HOME_DIR + 'datasets/COCO2017/images/*.jpg'

i=0
for file in glob.glob(path):
    print(file)
    #Preprocessing
    im = cv2.imread(file)    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    data = cv2.resize(im, (300, 300), interpolation=cv2.INTER_LINEAR)
    #normalization
    data=(((2.0/255.0)*data)-1)
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
#converter.inference_output_type = tf.uint8
converter.allow_custom_ops=True

tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("PTIQ")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_quant_file = tflite_models_dir/"quantized_model_ssdmobilenet.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)

