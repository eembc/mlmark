# Tensorflow Lite Reference Target

This is a generic target for running on hardware-independent TensorFlow Lite using Python3. It is provided to facilitate an out-of-the-box experience that will work on the greatest number of platforms, however, the models used by the target may not be optimized, hence it is only a reference.

# System Requrirements

* An OS running Python3 that has OpenCV support (it was tested on Ubuntu 16.04 and 18.04):

* TensorFlow Lite runtime API (tested on 1.14.0):
 https://www.tensorflow.org/lite/guide/python

* Some systems may require TensorFlow 1.15rc2:
~~~
sudo pip3 install tensorflow==1.15rc2
~~~

## Model Conversion
The "golden" MLMark Tensorflow models provided in this benchmark were converted from Tensorflow to TensorflowLite, and then quantized post-training using PTIQ on 200 images from the relevant data set. Refer to the scripts in the `utility` for the exact commands used to perform the conversion.

## OpenCV 4.x

Image preprocessing is done with Python OpenCV. Most systems can install Python OpenCV as follows:

~~~
$ python -m pip install opencv-python --user --upgrade
~~~

However, some Arm systems do not have a Python distribution at the PIP web wheel URI, so you must install via `apt`:

~~~
$ sudo apt install python-opencv
~~~

If it is still not avaialable, Python OpenCV will need to be built [with these instructions](https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html).
