# Edge TPU Target

This target makes use of Google's edge TPU to find latency, throughput, and accuracy for a given workload. This target has been coded and tested for both the edge TPU USB accelerator and the Coral TPU development board. Both products can be found [here](https://coral.withgoogle.com/products/).

Several configuration files are already provided. When writing a custom configuration file, keep in mind this target only supports `batch` size and `concurrency` of one. The hardware code is `e_tpu`, and the only supported `precision` is `int8`. These six pre-made configuration files are located in the top-level of the repository in the `config` folder and can be executed by the harness as follows:

~~~
% harness/mlmark.py -c config/tflite-tpu-*
~~~

# System Requrirements

## Disk Space

MLMark may exceed the disk space of the default flash partition on the Coral Dev Board; it is recommended to use a high-speed SD Card in the micro-SD slot and work from that mount.

## Platform Setup

Before running MLMark on the Edge TPU device, first verify that the examples from Google work. Requirements can be found at the official Coral page for the [USB accelerator](https://coral.withgoogle.com/docs/accelerator/get-started/) or the [development board](https://coral.withgoogle.com/docs/dev-board/datasheet/). The image classification example is sufficient to verify the board is set up properly.

## Software Setup

As stated in the main MLMark docs, you will need to install Python `future`, `numpy` and `cv2`. The upgrade to `numpy` is required, since the version installed in the Mendel OS is not recent enough.

~~~
% pip3 install future
% pip3 install numpy --upgrade
~~~

OpenCV installation is discussed in the following section, as it is a bit more complex.

## OpenCV 4.x
Image pre-processing is done with Python OpenCV, which means OpenCV must be installed. For the USB accelerator, most systems can install Python OpenCV as follows:

~~~
% pip3 install opencv-python --upgrade
~~~

However, OpenCV is not installed by default on the Coral board OS image. Instructions are provided below.

### Installing OpenCV on the Coral Development Board

Create a temporary 1G swap for the build. This is necessary to avoid running out of memory:
~~~
% sudo fallocate -l 1G /swapfile
% sudo chmod 600 /swapfile
% sudo mkswap /swapfile
% sudo swapon /swapfile
~~~
Update the OS packages:
~~~
% sudo apt-get update
% sudo apt-get upgrade
~~~
Install the standard build prerequisites (including Python3 development, otherwise CMake will not build the Python libraries).
~~~
% sudo apt-get install build-essential cmake unzip pkg-config \
 libjpeg-dev libpng-dev libtiff-dev \
 libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
 libxvidcore-dev libx264-dev \
 libgtk-3-dev \
 libatlas-base-dev gfortran \
 python3-dev \
 -y
~~~
Working from the MicroSD card partition, download OpenCV source:
~~~
% cd /sdcard (or whatever your SD partition is named)
% wget -O opencv.zip https://github.com/opencv/opencv/archive/4.0.0.zip
% unzip opencv.zip
% mv opencv-4.0.0 opencv
~~~
The build process uses CMake, which follows a common progression. Create the build directory:
~~~
% cd opencv
% mkdir build
% cd build
~~~
Then issue this CMake command:
~~~
% cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D INSTALL_PYTHON_EXAMPLES=OFF \
 -D INSTALL_C_EXAMPLES=OFF \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D ENABLE_FAST_MATH=1 \
 -D ENABLE_NEON=ON -D WITH_LIBV4L=ON \
 -D WITH_V4L=ON \
 -D BUILD_EXAMPLES=OFF \
 ..
~~~
Now run make: 
~~~
% make
~~~
It should take approx 3-3.5 hours. When complete, install as `sudo`:
~~~
% sudo make install
% sudo ldconfig
~~~
For some reason the installer does not properly link the library, so do it manually:
~~~
% cd /usr/local/lib/python3.5/dist-packages
% sudo ln -s /usr/local/python/cv2/python-3.5/cv2.cpython-35m-aarch64-linux-gnu.so cv2.so
~~~
And test that the Python library was installed:
~~~
% python3
>>> import cv2  
>>> cv2.__version__  
'4.0.0'  
~~~

# Model Conversion

The "golden" MLMark Tensorflow models provided in this benchmark were converted and quantized from the `.pb` file to `.tflite` by using post-training full-integer quantisation (PTIQ) with 200 calibration images from the relevant data set. The code for this can be found in the `utility` directory in this target.

The models were then compiled for the Edge TPU using the Google TPU flow. If the TFLite models are not compiled using the Google flow, the TFLite will run on the host CPU, rather than the TPU. There is no error generated when this happens, but the scores will be lower. Refer to [this Coral/Google document](https://coral.withgoogle.com/docs/edgetpu/compiler/) for more information on the Edge TPU Compiler.

# Known Issues

* To solve this error: `RuntimeError: Error in device opening (/sys/bus/usb/devices/1-1)!`, unplug and re-plug the USB accelerator.
* The error: `RuntimeError: Error in device opening (/dev/apex_0)!` on the dev board may indicate that the `sudo` access is required.
