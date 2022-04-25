
# Introduction

The EEMBC® MLMark&trade; benchmark is a machine-learning (ML) benchmark designed to measure the performance and accuracy of embedded inference. It was developed by a team at [EEMBC](http://www.eembc.org), which is a non-profit consortium founded in 1997 that produces benchmarks for embedded devices. Engineers from Intel, Ignitarium, NVIDIA, Texas Instruments, Arm and more participated in its development. The motivation for creating this benchmark grew from the lack of standardization of the environment required for analyzing ML performance. Most of the challenges the team faced were about how to sufficiently define the boundaries of an environment to ensure adequate reliability, flexibility, and confidence in the results. Further, the source-code implementations are provided here to help demystify exactly what the environment looks like at this point in time, as it is rapidly evolving. MLMark will continue to grow the database of supported target platforms and workloads over time.

The MLMark benchmark adopts the following philosophy:

1. Clearly Define the Metrics

"Latency" and "throughput" mean different things to different people. Similar can be said of model "accuracy", where ground-truths are used. MLMark clearly defines each term, exactly how the measurements are taken, and how accuracy is calculated. Over time, all new additions to the benchmark will be measured the same way creating a historic record of inference trends.

2. Publish the Implementations

Rather than setting rules and allowing individuals to perform optimizations in a private setting, MLMark requires the implementations be made public in the repository. Version 1.0 includes source code (and libraries) for:
	
	* Intel® CPUs, GPUs and neural compute sticks using OpenVINO®
	* NVIDIA® GPUs using TensorRT
	* Arm® Cortex®-A CPUs and Arm Mali™ family of GPUs using Neon™ technology and OpenCL™, respectively.
	* Google Edge TPU using TensorFlow Lite

3. Select Specific Models
	
There are many, many variables that impact ML performance, with the neural-net graph (model) being the most important. Consider a residual network for image classification: there are many such networks online, ResNet-50 is a popular model. ResNet-50 has many variations and optimizations (different input layer strides), as well as model formats (Caffe, Tensorflow), and different training datasets. To provide a consistent measurement, EEMBC selected specific models that are the most common and well-documented at this point in time, as well as the most likely to run on edge hardware.

4. Educate and Enable Users

Publishing the implementations not only ensures transparency, but also helps educate people working with performance analysis. Many embedded engineers with decades experience are being asked to tackle this new technology, and the learning curve is steep. By consolidating models and code for multiple targets, and by keeping the implementations as simple as possible, MLMark provides broad insight into the nuts-and-bolts of inference across different SDKs.

More information can be found on the [MLMark website](https://www.eembc.org/mlmark).

# Installation

The following checklist enumerates the installation steps. See below for more details on each step.

1. Clone this repo, or download a stable release and uncompress it.
2. Download image datasets (see below under 'Datasets')
3. Link the image datasets to their respective `datasets/*/images` folder
4. Install Python requirements
5. Set up a target to run on (there are four in this release)
6. Run the benchmark

Note: Due to copyright licensing, the benchmark and its datasets must be installed separately.

## Benchmark Structure

It helps to understand the directory structure of MLMark, which is divided into five areas:

* `config` - This contains JSON configuration files that describe benchmark tasks. Each task generates a single metric, either *accuracy*, *latency*, or *throughtput*. A configuration file contains a list of JSON objects describing the task(s). Some pre-made configurations are provided to get started.

* `datasets` - The location of all input data (e.g., images and ground truths). This folder is empty by default, see below about downloading datasets.

* `harness` - The harness is the Python code for invoking the benchmark and collecting and parsing results. The test harness coordinates the invocation of the workloads defined in the *configuration* files. Each configuration defines a list of tasks for each target and workload, and the corresponding parameters. Each task generates a score. `harness/mlmark.py` is the main program.

* `models` - The library of models used for for the benchmark. Sometimes models are added dynamically by targets, and each target may use a different model format (Tensorflow, OpenVINO, UFF, etc.).

* `targets` - The code used to benchmark on a particular platform, architecture, framework and/or SDK combination.

## Harness

The test harness is a thin Python layer that imports and calls into one of the target modules. All but one of the targets is implemented as a C++ library or executable that accepts commands from the harness and returns data to it. The harness performs the score computations based on the inference results from the target. The harness also selects which input data the target uses.

Some targets run on the host device, such as the Arm NN or Tensorflow targets. Others run on remote devices, such as the OpenVINO support for USB compute sticks. Future versions of MLMark will support inference over USB (like the compute stick) or serial ports to enable constrained devices.

## Datasets 

The datasets used for this version of MLMark have copyright license requirements and cannot be redistributed by EEMBC. Two data sets are needed the ImageNet ILSVRC2012 and the Microsoft COCO2017. The annotation files are already included in the `mlmark/datasets/<DATASET>/annotations` folders, but the images must be installed separately into the `images` folder under each dataset.

**ImageNet ILSVRC 2012**

~~For ILSVRC 2012, visit [this link](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads) and download the the "Validation Images", which is a 6.3GB archive. After uncompressing this archive, navigate to `mlmark/datasets/ILSVRC2012` and link (or copy) the dataset folder to `images`. The contents of `mlmark/datasets/ILSVRC2012/images` should now be all of the `.JPEG` files.~~

ILSVRC is only available by requesting access and agreeing to their terms. More information for obtaining ImageNet datasets can be found [here](https://www.image-net.org/download.php).

TIP: To avoid duplicating the files and taking up drive space, it is preferrable to link to the files rather than copy them

**Microsoft COCO 2017**

For COCO 2017, visit [this link](http://cocodataset.org/#download) and download the 2017 validation images, which is about 1.1GB. Un-tar this file. It will create a folder called `val2017`, full of JPEGs. After uncompressing this archive, navigate to `mlmark/datasets/COCO2017` and link (or copy) the dataset folder to `images`. The contents of `mlmark/datasets/COCO2017/images` should now be all of the `.jpg` files.

## Package Requirements

Anyone who as worked in machine learning knows there are a large number of packages that need to be installed on a system in order to run an inference. To newcomers, it is a good idea to familiarize yourself with Python [`pip`](https://pip.pypa.io/en/stable/installing/) and [`apt`](https://help.ubuntu.com/lts/serverguide/apt.html.en). Detailed discussion of package managers is beyond the scope of this document.

This benchmark requires Python 3.x and was built and tested on Ubuntu 16.04. It may work on later versions of Ubuntu but this is not guaranteed.

**Python Packages**

The following Python3 packages are required:

~~~
$ python3 -m pip install numpy --user --upgrade
$ python3 -m pip install future --user
$ python3 -m pip install progress --user
$ python3 -m pip install opencv-python --user
~~~

NOTE #1: Upgrade `numpy` since the reporting code uses new formatting functions not found in versions before 1.15.

NOTE #2: Not all Arm installations have an OpenCV wheel available via Pip. For those systems please use:

~~~
$ sudo apt install python3-opencv
~~~

The Ubuntu package is called `python3-opencv` and not `opencv-python`.  If the Ubuntu package does not exist, you will need to compile your own Python version of OpenCV, simply follow these [instructions](https://docs.opencv.org/4.1.0/d2/de6/tutorial_py_setup_in_ubuntu.html), and don't forget to install the `python3-dev` system package first.

**Target-Specific Packages**

Each target may have its own set-up requirements, for example Intel OpenVINO will require the OpenVINO SDK, and nVidia Jetpack will require the TensorRT SDK. Refer to the README file in each `targets/<TARGET>` folder.

## Run the Benchmark

At least one target needs to be configured to run MLMark. The easiest target to set up is the native Python tensorflow target. Follow the instructions in the `targets/tensorflow/README.md` to install the necessary packages.

The benchmark is invoked with the following command (see an explanation of config files in 'Configuration', below)

~~~
$ harness/mlmark.py -c config/tf-cpu-mobilenet-fp32-throughput.json
~~~

The `-c` flag can take multiple configuration files, like this (which will run all models and formats on Tensorflow and measure throughtput and latency, but not accuracy, which can take up to an hour or more on some devices):

~~~
$ harness/mlmark.py -c config/tf-cpu-*-throughput.json
~~~

The resulting output should look something like this, with some lines in the middle removed to shorten out the results. (The `-v` option enables verbose output, which varies by target and can generate a lot of extra text.)

~~~
-INFO- --------------------------------------------------------------------------------
-INFO- Welcome to the EEMBC MLMark(tm) Benchmark!
-INFO- --------------------------------------------------------------------------------
-INFO- MLMark Version       : 1.0.0
-INFO- Python Version       : 3.7
-INFO- CPU Name             : Intel(R) Core(TM) i7-4750HQ CPU @ 2.00GHz
-INFO- Total Memory (MiB)   : 8192.0
-INFO- # of Logical CPUs    : 8
-INFO- Instruction Set      : x86_64
-INFO- OS Platform          : Darwin-18.5.0-x86_64-i386-64bit
-INFO- --------------------------------------------------------------------------------
-INFO- Models in this release:
-INFO-     resnet50       : ResNet-50 v1.0 [ILSVRC2012]
-INFO-     mobilenet      : MobileNet v1.0 [ILSVRC2012]
-INFO-     ssdmobilenet   : SSD-MobileNet v1.0 [COCO2014]
-INFO- --------------------------------------------------------------------------------
-INFO- Parsing config file config/tf-mobilenet.json
:
:
:
-INFO- Target          Workload        H/W   Prec  Batch Conc. Metric       Score    Units
-INFO- -----------------------------------------------------------------------------------
-INFO- tensorflow      mobilenet       cpu   fp32      1     1 throughput    50.9      fps
-INFO- tensorflow      mobilenet       cpu   fp32      1     1 latency       20.4       ms
-INFO- tensorflow      resnet50        cpu   fp32      1     1 throughput    12.0      fps
-INFO- tensorflow      resnet50        cpu   fp32      1     1 latency       86.3       ms
-INFO- tensorflow      ssdmobilenet    cpu   fp32      1     1 throughput    17.6      fps
-INFO- tensorflow      ssdmobilenet    cpu   fp32      1     1 latency       58.1       ms
-INFO- Total runtime: 0:03:06.323851
-INFO- Done
~~~

See the section on *Scores*, later on in this document.

## Configuration Files

There are many ways to perform neural-network inference on a platform. In fact, even defining an innocuous term like *platform* requires a long list of conditions describing the hardware and software used to build that platform. A platform consists of a host CPU that runs a test harness, a target ML acceleration device, an ML framework for parsing models, an SDK and compute library for handling computations, and possibly drivers for the accelerator. 

In its most basic form, MLMark collects data by running a *tasks* which consists of a *target*, and *workload*, and a set of *parameters*. Each task measures an aspect of neural net behavior--such as latency, or accuracy--for a certain set of conditions. Collecting measurements over different parameters offers a characterization of a target for that workload.

Tasks are fed to the MLMark test harness via configuration files.

Each configuration file in the `config` folder is a JSON file that contains the following list of objects:

~~~
[
	{
		"target": a target name from the targets/ folder,
		"workload": a workload name from the targets/TARGET/workloads folder,
		"params": {
			"mode": throughput, latency, or accuracy
			"hardware": cpu, gpu, npu, etc. depending on the target
			"precision": can be fp32, fp16, int8, etc. depending on the target
			"iterations": how many iterations to run before measurement
			"concurrency": number of parallel inference operations (not supported on all targets)
			"batch": how many inputs to batch per iteration (not supported on all targets)
		}
	},
	:
]
~~~

Here is an example:

~~~
[
	{
		"target": "tensorflow",
		"workload": "mobilenet",
		"params": {
			"mode": "throughput",
			"hardware": "gpu",
			"precision": "fp32",
			"iterations": 1024,
			"batch_size": 1,
			"concurrency": 1
		}
	}
]
~~~

It is up to the target implementation to determine if a parameter is invalid, unsupported, or out of range. For example, the basic Tensorflow target will fail to run on a 'gpu' hardware if no GPUs are present, or if a precision other than 'fp32' is specified.

* `mode` Can be `throughput`, `latency`, or `accuracy`. In `throughput` mode, concurrency and batch may be set. `latency` is an alias for `throughput` that sets `batch` and `concurrency` to '1', as this implies the best-case latency. `accuracy` mode ignores batch, iterations and concurrency, and runs the accuracy (aka, validation) computation (which can take a long time for some targets).

* `batch` refers to how many inputs are sent to the model simultaneously. In the case of a neural-net, batch refers to the shape of the input tensor. All four inputs are presented to the same inference code simultaneously. There is a subtle difference between `batch` and `concurrency`.

* `concurrency` refers to how many parallelism inferences should be configured. Not all targets support this. The inferences need not start at the same time, like with batch. Typically `concurrency` involves multiple instances of the model copied on the hardware, whereas `batch` just increases the size of the input tensor. Concurrency may be implemented as threads or in TensorRT terms, streams (not to be confused with image streams from a video camera). These threads/streams are launched asynchronously and polled for completion. Each one may also employ batch parallelism, leading to performance tradeoffs between concurrency and batch size.

* `iterations` refers to how many inferences calls are invoked. For example, if `iterations` is 100, and `batch` is 5, 500 images are inspected in 100 inferences. If `batch` is 1 and `concurrency` is 5, 100 inferences are performed with a maximum parallelism of 5.

* `hardware` and `precision` are specific to the target.

This single task selects target `tensorflow`, which is a native Python Tensorflow implementation, and a workload that uses the *model* `mobilenet`, which is a keyword that stands for "MobileNet V1.0 224 trained on ILSVRC2012". The parameters will be discussed in more depth later on.

## Default Targets and Workloads

In order to provide a consistent benchmark experience, and eliminate confusion around implementations, EEMBC only allows scores to be submitted for targets that have been qualified to be part of the benchmark. Over time, EEMBC will add more targets, expand on existing targets, and add more workloads. Each target in the repository has its own readme explaining how to set it up and it's supported hardware & precision.

The targets provided are:

* Tensorflow (`tensorflow`) - This is native Tensorflow running under Python. Only precision `fp32` and conurrency of 1 are supported. If `tensorflow-gpu` is installed, hardware type `gpu` is available.

* Tensorflow Lite (`tensorflow_lite`) - Similar to the Tensorflow target, this target uses TensorFlow Lite native Python. Both frameworks are advancing rapidly, and whether or not this target functions properly depends on the status of the `tensorflow` and/or `tflite_runtime` frameworks. Results vary considerably with the Python wheel for the runtime. As a result, this target is provided as a tool for sanity checks.

* Intel OpenVINO (`openvino_ubuntu`) - This target supports Intel CPUs with OpenVINO and MKL, GPUs, Movidius Neural Compute Sticks, HDDLr and FPGA.

* TensorRT Nano (`tensorrt`) - This target uses TensorRT, cuDNN and Cuda optimized for the Jetson Nano platform.

* ArmNN (`armnn_tf`) - This target uses the ArmNN 19.02 API and Arm Compute Library with Neon (ACL) on Arm Cortex-A5 and Cortext-A7 CPUs as well as Mali GPUs. The C++ files use the Tensorflow model loader, not the TensorFlow Lite API (Currently SSDMobileNet is not supported.)

* Google Edge TPU (`google_tpu`) - This target uses the native Python edge-TPU package under Tensorflow for the Google developer board and USB accelerator versions of the edge TPU. Only `int8` precision on `tpu` hardware is supported. The models were compiled from the Tensorflow Lite model folder using the Edge TPU flow. Refer to the target and model READMEs for more information. 

For more details about the targets, please see the README.md files for each respective target in this repository. Vendors seeking to include targets in the repository should contact info@eembc.org.

The workloads selected are:

* ResNet-50 v1.0 (`resnet50`) trained on ILSVRC2012 (see `models/tensorflow/resnet50/README.md`)

* MobileNet v1.0 224x224 (`mobilenet`) trained on ILSVRC2012 (see `models/tensorflow/mobilenet/README.md`)

* SSDMobileNet v1.0 300x300 (`ssdmobilenet`) - trained on COCO2014 (note that MLMark inference is done on COCO2017; see `models/tensorflow/ssdmobilenet/README.md`)


# Scoring

*Throughput* depends on the model. Since all three models in this release are image processors, the units are frames-per-second. We measure throughput as X number of iterations times Y batch sizes, divided by total time.

*Latency* is defined as the time it takes to process a single input for a single iteration. We measure the time for all iterations, take the 95th percentile, and report that in milliseconds. 95th percentile means 95% of the time the machine can do better than this, which is a common industry metric. `latency` mode forces a batch size one and a concurrency of one.

*Accuracy* depends on the model.

ResNet50 and MobileNet are scored by Top-1 and Top-N accuracy. During an accuracy run, the harness supplies a subset of the test data to optimize for runtime. Only the first 5 images of each class are used, totally 5,000 images from ILSVRC2012.

SSD-MobileNet is scored using IOU mAP at an interpolation interval from [0.50, 0.95] at 0.05 increments, so 101 interpolation samples. All 5000 images in the val2017 dataset are used.

All floating-point results are reported with three significant figures (not fixed decimal points).

# Submitting Scores

Scores may be submitted to the database from the [MLMark Score Submission](https://www.eembc.org/mlmark/submit.php) page. Scores are reviewed for errors or pecularities, and follow up may be required before the scores are published to the live database.

# Run Rules

1. Each performance run must operate for more than a minimum 1024 iterations; this is to ensure proper steady state conditions for both fast and slow devices and to accumulate enough samples for the 95th percentile latency to be meaningful.
2. Only targets that have been verified by EEMBC and made available for download in this repository are considered "valid" scores. Scores generated from modified models, input data, targets, or benchmark code do not constitute a valid MLMark score. Only configuration files may be changed to be considered valid scores. To have your target verified and included please contact info@eembc.org, as it will need to be inspected and tested by EEMBC labs.

# Submission Rules

1. The submission must adhere to the run rules.
2. Once a score has been submitted, it can no longer be edited by the submitter. Any changes requests must be sent to info@eembc.org for review.
3. Scores are reviewed by EEMBC before appearing on the website. The review checks for omissions, mistakes, or need for more explanation. Scores typically go live within a week.

# Publication Rules

1. As stated in the license, a "Commercial MLMARK License" from EEMBC is required for Licensee to disclose, reference, or publish test results generated by MLMARK in Licensee’s marketing of any of Licensee’s commercially‐available, product‐related materials, including, but not limited to product briefs, website, product brochures, product datasheets, or any white paper or article made available for public consumption. (This does not include academic research.)
2. Scores must be uploaded to the EEMBC MLMark website before being published in any capacity. This ensures that the score is valid according the run rules, and allows viewing of additional submission disclosure details.

# Versioning

Each version contains three numbers: MAJOR.HARNESS.MINOR.

The MAJOR verison indicates what models are supported by that release. For version 1, only three models are supported.

The HARNESS release indicates incompatible changes made to the harness that won't work with other harness versions. It may require a target update.

MINOR changes are compatible with the current MAJOR.HARNESS version. This includes adding a new target or model format.

# Copyright and Licensing

EEMBC and MLMark are trademarks of EEMBC. Please refer to the file LICENSE.md for the license associated with this benchmark software.

The dataset images are covered under various copyright and licenses, the same goes for the models. Please refer to the README.md files in the model areas for copyright attributions.

