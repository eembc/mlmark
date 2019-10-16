# Tensorflow Lite Post Training Full Integer Quantisation callibration scripts

overview:
https://www.tensorflow.org/lite/performance/post_training_quantization

## System Requrirements

* As on first week of Oct2019, tensorflow nightly build.

* An OS running Python3 that has OpenCV support (it was tested on Ubuntu 16.04 and 18.04, and macOS Mojave)

### Tensorflow 1.15+
At the time of writing this readme, the scripts worked with tensorflow 1.15.rc2 version.
Install it like this:
~~~
$ pip3 install tensorflow==1.15rc2
~~~

### OpenCV 4.x

Image preprocessing is done via Python OpenCV. Most systems can install Python OpenCV as follows:

~~~
$ python -m pip install opencv-python --user --upgrade
~~~

However, some Arm systems do not have a Python distribution at the PIP web wheel URI, so you must install via `apt`:

~~~
$ sudo apt install python-opencv
~~~

## Execution of scripts
Simply run the scripts with python3.
~~~
$ python3 mobilenet_PTIQ_script.py
~~~
A folder named PTIQ will be created containing converted and quantized tflite file for a model. 


