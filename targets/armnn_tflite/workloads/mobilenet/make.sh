#!/bin/sh
/usr/bin/c++ \
        -fPIC \
        -std=c++14 \
        main.cpp \
        -I ~/armnn-build/armnn/include \
        -I ~/armnn-build/boost/install/include \
        -I ~/opencv/modules/imgcodecs/include \
        -I ~/opencv/modules/core/include \
        -I ~/opencv/build \
        -I ~/opencv/modules/imgproc/include \
        -L ~/armnn-build/armnn/build \
        -L ~/opencv/build/lib \
        -larmnn \
        -larmnnTfLiteParser \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        ~/armnn-build/boost/install/lib/libboost_filesystem.a \
        ~/armnn-build/boost/install/lib/libboost_system.a \
        -o mobilenet.exe
