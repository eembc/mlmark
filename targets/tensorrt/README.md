# TensorRT Target

This target runs on NVIDIA hardware by using the TensorRT framework.

**NOTE**: As NVIDIA releases new TensorRT and Jetpack versions, it may become necessary to recompile the C++ interface libraries in this target. Please refer to the *Compiling* section below. An indication of this may be an error such as:

~~~
python3: engine.cpp:1104: bool nvinfer1::rt::Engine::deserialize(const void*, std::size_t, nvinfer1::IGpuAllocator&, nvinfer1::IPluginFactory*): Assertion `size >= bsize && "Mismatch between allocated memory size and expected size of serialized engine."' failed.
~~~

...this is a sign you should rebuild.

## System Requriements

* **Operating System**: Ubuntu 16.04 LTS (x86_64 or aarch64)
* NVIDIA JetPack 4.3

This target only supports hardware type `gpu`. Precision depends on the platform:

* Jetson Nano only supports `fp32` and `fp16`.
* Jetson Xavier supports `fp32`, `fp16`, and `int8`
* NVIDIA K80 supports `fp32`
* Support will vary depending on the hardware; the build engine will notify the user of any incompatibilities

## Setting up the TensorRT Target


Please refer to the "Getting Started" guide for the [Jetson Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit), it explains how to install JetPack and configure the device for both the Nano and the Xavier development kits.


Details about Jetson Nano can be found [here](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano) and the Xavier [here](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit).

* It is strongly advised to use 5V-4A external power supply and a fast microSD card with at least 64GB of memory

### Power Modes

NVIDIA provides multiple power modes for their devices. It is recommended to use `mode 0` for optimal performance. Verify this by checking the power mode:

You can verify the power mode with `nvpmodel`:

~~~
% sudo nvpmodel -q
NV Power Mode: MAXN
0
~~~

And set it as follows:

~~~
% sudo nvpmodel -m 0
~~~

### Clock Frequency

The JetPack environment comes with a `jetson_clocks` command to enable fast-mode operation. For optimal performance, this should be run prior to collecting scores:
~~~
% sudo jetson_clocks 
~~~ 

## Running the Benchmark

The included libraries were compiled directly to aarch64 for the Nano and Xavier, and should not require recompiling. To recompile on a different host architecture, please see the section below on Compiling.

Sample configuration files are provided:

~~~
% harness/mlmark.py -c config/trt-nano-mobilenet-fp16.json
~~~

The first thing the target does is generate an optimized model for the requested batch size. This takes about ~5 minutes, and is re-used for all subsequent runs. The model is optimized for the particular hardware via the NVIDIA framework.
 
## Target Folder Structure

The following folders come from the GitHub repo:

`cpp_environment`: C++ source files for CUDA interface are provided here, and compiled as libraries that are loaded by Python directly.

`utility`: This folder contains instructions to convert a frozen buffer `*.pb` file to a `*.uff` file. There are 'calibrationTable' files which are required by the TensorRT builder to optimize and build an `int8` engine. Currently, these tables are generated using the first 500 images from ILSVRC2012 dataset for Mobilenet & Resnet50 and COCO 2017 dataset for SSDMobileNet.

`targets`: The Python wrappers for `mobilenet`, `resnet50` and `ssdmobilenet` are located here, like the other targets.

These folders are created after the first run:

`engines`: This folder contains TensorRT's optimized engine files. For each batchsize, a separate engine file is created.

`Results`: Interim results in the form of text files are stored in this folder.

## Compiling & Dependencies

As stated earlier, the default libraries were compiled on Arm architecture for both Nano and Xavier. The libraries can be recompiled for x86_64 as well by doing the following:

In order to compile, you must checkout NVlabs [cub](https://github.com/NVlabs/cub) and then add this line to `cpp_environment/CMakeLists.txt`:

~~~
include_directories(/home/dev/cub)
~~~

...where `/home/dev/cub` is the path to the repository. Then you can build:

~~~
% cd targets/tensorrt/cpp_environment
% cmake .
% make
~~~

## Known Issues:

* Sometimes process may get "killed" while creating an engine. This may be due to insufficient system resources, so make sure no other applications are running. Simply re-running should solve the problem, as all models have been tested for batch sizes up to 16 with success.
* Using large numbers for `concurrency` and `batch_size` might exaust system resources, causing a "Killed" message. This is a known issue and does not yet have a workaround.
