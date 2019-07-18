/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <MnistUbyte.h>
#include <fstream>
#include <iostream>
#include <string>

using namespace FormatReader;

int MnistUbyte::reverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = (unsigned char)(i & 255);
  ch2 = (unsigned char)((i >> 8) & 255);
  ch3 = (unsigned char)((i >> 16) & 255);
  ch4 = (unsigned char)((i >> 24) & 255);
  return (static_cast<int>(ch1) << 24) + (static_cast<int>(ch2) << 16) +
         (static_cast<int>(ch3) << 8) + ch4;
}

MnistUbyte::MnistUbyte(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return;
  }
  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;
  file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
  magic_number = reverseInt(magic_number);
  if (magic_number != 2051) {
    return;
  }
  file.read(reinterpret_cast<char *>(&number_of_images),
            sizeof(number_of_images));
  number_of_images = reverseInt(number_of_images);
  file.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
  n_rows = reverseInt(n_rows);
  _height = (size_t)n_rows;
  file.read(reinterpret_cast<char *>(&n_cols), sizeof(n_cols));
  n_cols = reverseInt(n_cols);
  _width = (size_t)n_cols;
  if (number_of_images > 1) {
    std::cout << "[MNIST] Warning: number_of_images  in mnist file equals "
              << number_of_images << ". Only a first image will be read."
              << std::endl;
  }

  size_t size = _width * _height * 1;

  _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
  size_t count = 0;
  if (0 < number_of_images) {
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_cols; ++c) {
        unsigned char temp = 0;
        file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
        _data.get()[count++] = temp;
      }
    }
  }

  file.close();
}
