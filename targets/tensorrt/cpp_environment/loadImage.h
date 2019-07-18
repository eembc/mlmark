/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __IMAGE_LOADER_H_
#define __IMAGE_LOADER_H_

#include "cudaUtility.h"

// These are necessary for templated structure. ToDo: understand why.
static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 224;
static constexpr int INPUT_W = 224;

// Just a structure to hold data. Much useful for batchsizes>1.
template <int C, int H, int W> struct bImage { float buffer[C * H * W]; };

/**
 * Load a color image from disk into CUDA memory.
 * This function loads the image into shared CPU/GPU memory, using the functions
 * from cudaMappedMemory.h
 *
 * @param filename Path to the image file on disk.
 * @param cpu Pointer to CPU buffer allocated containing the image.
 * @param gpu Pointer to CUDA device buffer residing on GPU containing image.
 * @param width Variable containing width in pixels of the image.
 * @param height Variable containing height in pixels of the image.
 *
 * @ingroup util
 */
bool loadData(const char *filename, float3 **cpu, float3 **gpu, float *data,
              int batchsize, int width, int height);

#endif
