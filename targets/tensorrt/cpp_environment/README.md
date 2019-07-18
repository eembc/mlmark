# CPP lib files

TensorRT's cpp API is supported on Jetson devices.
Classes are created for Resnet50, Mobilenet and SSD-Mobilenet.
Comments are available in every cpp file.

In order to make changes, just open a class_xyz.cpp file, do modifications, save and use 'make' command to generate *.so files.
Files are created inside 'libs' folder.

Folder path of images used for int8 calibration are hardcoded inside BatchStream.h and BatchStream_ssdmobilenet.h. Incase paths are changed, modification in required. As of now, paths are datasets/COCO2017/images/ and datasets/ILSVRC2012/images/.  
