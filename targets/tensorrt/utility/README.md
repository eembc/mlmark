## Utility

This code is developed using TensorRT's UFF parser. Pretrained models of Resnet50,Mobilenet and SSD-Mobilenet in Tensorflow's frozen buffer format(*pb) are downloaded. These files need to be converted into (*.uff) format.


pb to uff conversion should be done on a x86 machine(laptop or a desktop with Nvidia GPU).

Prerequisits:
CUDA, cuDNN and TensorRT

convert_to_uff utility is part of TensorRT installation.
You can simply run the utility as:
This command is sufficient for Resnet50 and Mobilenet. You can find the name of last layer using first command. Use second command to convert the file.

~~~
$convert_to_uff input_file.pb -l  //to visulize all the layers.

$convert_to_uff input_file.pb -o output_file.uff -O name_of_output_node  //to convert a pb file to uff file.

~~~

Things are slightly different in case of SSD-Mobilenet. The 'config.py' file available in this folder is required to convert SSD-Mobilenet's pb file to uff format. 
Command to do this:

~~~

$ convert-to-uff --input-file ../../../models/tensorflow/ssdmobilenet/frozen_graph.pb -o ../../../models/tensorflow_uff/ssdmobilenet/frozen_graph_ssdmobilenet.uff -O NMS -p config.py

~~~

Converted uff files should be placed at mlmark/models/tensorflow_uff location. 

## Int8 calibration

TensorRT supports int8 precision mode. To use this precision mode, you need to do a calibration process. This process is a one time process for a network.
Output of this process is a "Calibration Table file". This folder contains calibration files for Mobilenet, Resnet50 and SSDMobilenet. These calibration Tables are generated using first 500 images of ILSVRC2012 dataset for Mobilenet & Resnet50 and COCO2017 dataset for SSDMobilenet. 

In case you want to do calibration with any other dataset, then please do following steps:
1. Delete these existing calibrationTable files.
2. Make sure that you have all the image file names in text files. For example, have a look at ILSVRC2012_list.txt
3. Update image path in files - targets/tensorrt/cpp_environment/BatchStream.h and BatchStream_ssdmobilenet.h
4. Build new libs by cmake . and make
5. Delete any exisiting int8 engines inside tensorrt/engines/ folder
6. execute mlmark in int8 mode. You will see calibration process on terminal while engine creation process. 
