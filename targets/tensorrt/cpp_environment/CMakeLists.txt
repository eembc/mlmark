cmake_minimum_required(VERSION 2.8)

project(res)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
	-O0 
	-gencode arch=compute_53,code=sm_53
	
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs)

find_package(CUDA)

include_directories(/usr/local/cuda/include /usr/include/aarch64-linux-gnu /usr/local/cuda/lib64 )
link_directories(/usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)

ADD_LIBRARY(class_resnet50 SHARED resnet50_class.cpp loadImage.cpp )
ADD_LIBRARY(class_mobilenet SHARED mobilenet_class.cpp loadImage.cpp )
ADD_LIBRARY(class_ssdmobilenet SHARED ssdmobilenet_class.cpp loadImage.cpp )

#add_executable(res resnet50_class.cpp)

target_link_libraries(class_resnet50 cudart nvparsers nvinfer_plugin nvinfer)
target_link_libraries(class_mobilenet cudart nvparsers nvinfer_plugin nvinfer)
target_link_libraries(class_ssdmobilenet cudart nvparsers nvinfer_plugin nvinfer)
