# Tensorflow Reference Target

This is a generic target for running on hardware-independent Tensorflow using Python. It is provided to facilitate an out-of-the-box experience that will work on the greatest number of platforms, however, the frozen models used by the target may not be optimized, hence it is only a reference.

The target supports any number of batch size, however, `concurrency` is limited to one. If `tensorflow-gpu` is installed, `hardware` may be `gpu` as well as `cpu`. The only supported `precision` is `fp32`. Again, optimization is not performed on the model, this is only a reference.

## System Requrirements

* An OS running Python3 that has a Python tensorflow wheel and Python OpenCV support (it was tested on Ubuntu 16.04 and 18.04, and macOS Mojave)

The following hardware configurations are supported:

* `cpu` (precision `fp32`):
* `gpu` (precision `fp32`)
	* via `tensorflow-gpu`, which at this time only supports NVIDIA.

## Installation Requirements

### Tensorflow 1.13.1

If you want to use the native Tensorflow target, install Tensorflow for Python using `pip`:

~~~
$ python3 -m pip install tensorflow==1.13.1 --user
~~~

The Tensorflow 1.13.1 Python wheel for a variety of Arm platforms can be found here:

https://github.com/lhelontra/tensorflow-on-arm/releases/tag/v1.13.1

And to install:

~~~
% python -m pip install path_to_*.whl --user
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

If it is still not avaialable, Python OpenCV will need to be built (with these instructions)[https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html].

### Tensorflow GPU

To use Tensorflow GPU on nVidia, (this is not the same as TensorRT) you will need to install CUDA V10.0 and cuDNN > 7.4.1 (see the [tensorflow-gpu](https://www.tensorflow.org/install/gpu) page).

https://developer.nvidia.com/cuda-downloads

Then install `tensorflow-gpu`:

~~~
$ python3 -m pip install tensorflow-gpu==1.13.1 --user
~~~

## Issues With `cpu` and `gpu` after Installing TF-GPU

Note that after installing `tensorflow-gpu` the GPU will always be selected. In addition to setting `hardware` to `cpu` in the configuration file, you must also set the environment variable `CUDA_VISIBLE_DEVICES` to blank, like this:

~~~
CUDA_VISIBLE_DEVICES= ./harness/mlmark.py -v -c run.json 
~~~

... or for the session ...

~~~
export CUDA_VISIBLE_DEVICES=
~~~

... in order to completely avoid using the GPU.
