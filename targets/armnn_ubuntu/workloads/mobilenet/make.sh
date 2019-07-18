#!/bin/sh
/usr/bin/c++ \
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
