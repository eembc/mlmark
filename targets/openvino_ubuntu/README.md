# OpenVINO Target

This target provides MLMark support for Intel CPUs, GPUs and Neural Compute Sticks using OpenVINO on the V1.0 workloads (`resnet`, `mobilenet`, and `ssdmobilenet`).

## System Requirements

* **Operating System**: Ubuntu 16.04 LTS on x86_64

The following hardware configurations are supported:

* `cpu` (precision `fp32`):
	* 6th to 8th generation Intel Core and Intel Xeon processors 
	* Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics 
* `gpu` (precision `fp32`, `fp16`, `int8`):
	* 6th to 8th generation Intel Core processor with Iris® Pro graphics and Intel HD Graphics 
	* 6th to 8th generation Intel Xeon processor with Iris Pro graphics and Intel HD Graphics (excluding the e5 product family, which does not have graphics) 
* `myriad` or `hddl` (precisions `fp16`):
	* Intel Movidius Neural Compute Stick, HDDL-r  ( Only Batch size 1 is supported, and on **Linux**)
        
## Setting Up the OpenVINO Target

1. Install the MLMark benchmark following the instructions in the main [README file](https://github.com/eembc/mlmark/blob/master/README.md). This includes installing several Python3 packages and OpenCV. In addition, this target requires `pip3 install progress --user`.

2. Install [OpenVINO](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux) full package version, following the instructions for [Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux). Please follow the instructions exactly as specificed. **NOTE:** MLMark was developed and tested with OpenVINO 2018.R5
       
3. Build the workload binaries:

~~~
% cd ~/mlmark/targets/openvino_ubuntu/install
% chmod +x compile_MLMARK_sources.sh
% sudo ./compile_MLMARK_sources.sh </path/to/MLMARK> </path/to/OpenVINO>
~~~
4. To run on Intel Neural Compute Sticks (NCS1 or NCS2), from `install`, run:
~~~
% sudo install_myriad_bootrules.sh
~~~

5. Navigate back to the `mlmark` root directory, and run the benchmark (this example runs all workloads & precision performance workloads on an Intel CPU):

~~~
% cd ~/mlmark
% harness/mlmark.py -c config/ov-cpu-*-throughput.json
~~~

**NOTE**: If the above command fails on Linux, please try running with ```sudo ``` 
   
## Additional instructions

### Running on HDDL-r

If you want to benchmark on HDDL-r, follow the additional steps for running on **VPU** in the [Guide](https://docs.openvinotoolkit.org/2018_R5/_docs_install_guides_installing_openvino_linux.html)

After setting up the drivers and boot rules, the hddldaemon must be initilized in the different terminal before the benchmark can be run.

Follow the following steps to initialize the hddldaemon from your command line:

~~~
% sudo -Es
% source /path/to/openvino/bin/setupvars.sh
% /path/to/openvino/inference_engine/external/hddl/bin/hddldaemon
~~~

After the above steps, the hddldaemon initizlizes, and then you can run your benchmark with hddl hardware.

### Running SSDMobileNet on Integrated Graphics

For the SSDMobileNet workload, all batch sizes intended to run on the GPU must have corresponding model-optimizer generated IRs.
You need to edit the ```compile_MLMARK_sources.sh``` [here](https://github.com/eembc/mlmark/blob/master/targets/openvino_ubuntu/install/compile_MLMARK_sources.sh#L318) and run the script again.

### `int8` Model Calibration
The install script provided does not generate INT8 models, so the user must calibrate the generated fp32 models manually to int8. The steps required for the calibration process are outlined in the openvino [Python-calibration-tool](https://docs.openvinotoolkit.org/latest/_inference_engine_tools_calibration_tool_README.html) or [C++ calibration-tool](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_calibration_tool_README.html). The dataset preparation guidelines outlined in the [validation-app](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_validation_app_README.html) will be useful in successfully calibrating your models.

## Known Issues

1. If your system hosts HDDL-r, please do not connect the NCS1 or NCS2. As yet, inference on NCS cannot be done on a system hosting an HDDL-r
2. `apt` update commands in install/.sh scripts may not execute properly, **causing installation issues**

     For instance, when `sudo apt update` fails with `E: Could not get lock /var/lib/apt/lists/lock`. 
     - **Fix**: Run `sudo rm /var/lib/dpkg/lock /var/cache/apt/archives/lock /var/lib/apt/lists/lock` before running setup scripts


