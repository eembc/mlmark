# TensorFlow Lite Post-Training Full-Integer Quantisation calibration scripts

These scripts convert a TensorFlow `*.pb` model into an `int8` quantized `*.tflite` model.

An overview of this process can be found [here](https://www.tensorflow.org/lite/performance/post_training_quantization).

Simply run the scripts with python3:

~~~
$ python3 mobilenet_PTIQ_script.py
~~~

A folder named PTIQ will be created containing converted and quantized tflite file for a model. 
