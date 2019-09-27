# Arm NN Target

This target implements the workloads with the Arm NN SDK and Arm Compute Library.

## System Requrirements

* **Operating System**: Ubuntu 16.04 LTS on aarch64 (armv8a)

The following hardware configurations are supported:

* `cpu` (precision `fp32`):
	* Arm Cortex-A
* `gpu` (precision `fp32`, `fp16`):
	* Arm Mali GPU

## Setting Up the Arm NN Target

The supplied binaries should not need recompiling and should be ready to go out of the box. 

To obtain optimal performance:

* You may need to set an affinity mask via `taskset` to achieve optimal performance
* You may need to set the max/min frequency and governor of the DDR, CPU and GPU in order to achieve peak performance, depending on the platform. Be aware that fixing the frequency to the max value my introduce thermal shutdown issues to use adequate cooling. See below.

## Libraries

For convenience, the required libraries are included in the `common` folder as shared-object libraries: OpenCV (for image pre-processing), Protobuf (for reading models), and ArmNNTfParser (for reading models) and two versions of `libarmnn.so`, one compiled for CPU only (in `CpuAcc`) and one compiled for both CPU and GPU (`GpuAcc`). The former is included for CPU-only systems that do not have OpenCL installed.

The Arm NN libraries were compiled with the Arm ComputeLibrary configured for Neon, OpenCL and Mali. Follow [these instructions](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-tensorflow/single-page) from the Arm website and add `neon=1 opencl=1` to the ComputeLibrary build. The included libraries will built using GCC 5.4.0 for aarch64 on Ubuntu 16.04.

Each workload compiles a `[modelName].exe` file, which is called via subprocess by the function `run()` in `common/__init__.py`. The code uses JSON files to facilitate data passing between the Python harness and the target binaries. The input JSON is created by the `run()` function in the Python wrapper. The output JSON file is created by the binary, and then parsed into return schema by the `run()` functions, and returned to the calling process. This also makes debugging easier because the harness uses a temporary folder which contains the input JSON.

The Arm NN libraries provided are split into to folders `GpuAcc` for Mali acceleration and `CpuAcc` for Neon accelerated-libraries. The main API interface selects between `GpuAcc` and `CpuAcc` library paths based on the `hardware` setting in the parameter file. Refer to [these lines](https://github.com/eembc/mlmark/blob/a88f3f43f0e0efaaab28aade5f7d9081c51a54b9/targets/armnn_ubuntu/workloads/common/__init__.py#L139-L144) for details. This is only relevant if the libraries are recompiled.

## Frequency Pinning, Governors, and Affinity

Each Ubuntu system has different targets, so these paths may differ. Here is an example of configuring frequencies for the HiKey970 Arm/Mali platform (it lists the available settings first):

~~~
!/bin/bash
echo "Available DDR Settings (frequencies/governors):"
cat /sys/devices/platform/ddr_devfreq/devfreq/ddr_devfreq/available_frequencies
cat /sys/devices/platform/ddr_devfreq/devfreq/ddr_devfreq/available_governors
echo 1866000000 > /sys/devices/platform/ddr_devfreq/devfreq/ddr_devfreq/max_freq
echo 1866000000 > /sys/devices/platform/ddr_devfreq/devfreq/ddr_devfreq/min_freq
echo userspace > /sys/devices/platform/ddr_devfreq/devfreq/ddr_devfreq/governor
# cpu - Cortex A7's only
for x in $(seq 4 7); do
echo "Available CPU Settings for core ${x} (frequencies/governors):"
        cat /sys/devices/system/cpu/cpu$x/cpufreq/scaling_available_frequencies
        cat /sys/devices/system/cpu/cpu$x/cpufreq/scaling_available_governors
        echo 2362000 > /sys/devices/system/cpu/cpu$x/cpufreq/scaling_max_freq
        echo 2362000 > /sys/devices/system/cpu/cpu$x/cpufreq/scaling_min_freq
        echo performance > /sys/devices/system/cpu/cpu$x/cpufreq/scaling_governor
done
# gpu
echo "Available eu2c0000.mali GPU settings (frequencies/governors):"
cat /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/available_frequencies
cat /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/available_governors
echo 767000000 > /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/max_freq
echo 767000000 > /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/min_freq
echo userspace > /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/governor
~~~

Affinity is set by prefacing the run with `taskset`, for example

~~~
% taskset F0 harness/mlmark.py -v -c ...
~~~

Will only use cores 4-7, as hex F0 is a CPU mask.

## Compiling (not required)

Ubuntu binareis are provided, so there is no need to recompile. Until `cmake` rules are implemented, each file includes the compile comand in the header, for example:

~~~
$ /usr/bin/c++ \
	-fPIC \
	-std=c++14 \
	main.cpp \
	-I ~/armnn/include \
	-I ~/boost_1_64_0/install/include \
	-I ~/opencv/modules/imgcodecs/include \
	-I ~/opencv/modules/core/include \
	-I ~/opencv/build \
	-I ~/opencv/modules/imgproc/include \
	-L ~/armnn/build \
	-L ~/opencv/build/lib \
	-larmnn \
	-larmnnTfParser \
	-lopencv_core \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	~/boost_1_64_0/install/lib/libboost_filesystem.a \
	~/boost_1_64_0/install/lib/libboost_system.a \
	-o mobilenet.exe
~~~

Includes:
* `~/opencv/build`: If `libopencv-dev` is not installed on your system, you will need to build it locally and supply these includes.
* `~/boost_1_64_0`: If Boost (1.64) is not installed on your systel, you will need to build it locally and supply these includes.
* `~/armnn/include`: path to the ArmNN SDK include folder

Libraries:
* `~/armnn/build_neon`: path to where the libarmnn.so and armnnTfParser.so files are located
* `pthread`, `boost_filesystem`: Boost 1.64.0 is used throughout ArmNN, as are pthreads
* `armnn`, `armnnTfParser`: These are the two libraries for ArmNN and the TensorFlow parser used to convert the models.
* `opencv_*`: OpenCV is used to preprocess the images.

## Common Issues

Q1. Severe run-to-run variation on mobile eval boards
A1. The board may be throttling. See above notes on governors.

Q2. Cannot find OpenCL device when selecting `gpu`
A2. Some OpenCL setups require root access, e.g. `sudo harnes/mlmark.py ...`

Q3. Scores on the HiKey970 do not match published scores, even after using the speed configuration and taskset?
A3. The default HiKey970 board does not have sufficient cooling; an external fan is required to achieve the reported scores.

## Notes

1. Batching is currently not supported by ArmNN 19.02. When this changes in the future, the target will be released.
2. SSD-MobileNet is not implemented

