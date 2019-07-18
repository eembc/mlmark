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

#include "loadImage.h"
#include "cudaMappedMemory.h"

#include <iostream> //for std::cout
#include <vector>   //for vector

// loadImageRGB. ToDo: write flexible routines for both the cases. 1.images are
// read from file. 2.preprocessed data is received from python.
bool loadData(const char *filename, float3 **cpu, float3 **gpu, float *data,
              int batchsize, int width, int height) {
  if (!filename || !cpu || !gpu || !width || !height) {
    printf("loadImageRGB - invalid parameter\n");
    return false;
  }

  const int imgChannels = INPUT_C;
  const uint32_t imgWidth = width;
  const uint32_t imgHeight = height;
  const uint32_t imgPixels = imgWidth * imgHeight;
  const int volImg = imgWidth * imgHeight * imgChannels;
  const size_t imgSize =
      batchsize * imgWidth * imgHeight * sizeof(float) * imgChannels;

  // allocate buffer for the image
  if (!cudaAllocMapped((void **)cpu, (void **)gpu, imgSize)) {
    printf(LOG_CUDA "failed to allocated %zu bytes for image\n", imgSize);
    return false;
  }

  float *cpuPtr = (float *)*cpu;

  // bImage is data structure to hold data for 1 image. Now create vector of
  // that data structure whoes length is equal to batchsize.
  std::vector<bImage<INPUT_C, INPUT_H, INPUT_W>> b_images(batchsize);

  // now, empty vector of structures is available. Fill in data from data
  // pointer!
  for (int i = 0; i < batchsize; i++) {
    for (int j = 0; j < volImg; j++) {
      b_images[i].buffer[j] = data[i * volImg + j];
    }
  }

  // actual pixel format conversion.
  for (int i = 0; i < batchsize; ++i) // Batch N
  {
    for (int c = 0; c < imgChannels; ++c) // Channel C
    {
      for (unsigned j = 0; j < imgPixels; ++j) // H*W
      {
        cpuPtr[i * volImg + c * imgPixels + j] =
            b_images[i].buffer[j * imgChannels + c];
      }
    }
  }

  // printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth,
  // imgHeight, imgSize);

  return true;
}
