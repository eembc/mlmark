/*============================================================================*
 * Copyright (C) 2018 BenchmarkXPRT Development Community
 * Licensed under the BENCHMARKXPRT DEVELOPMENT COMMUNITY MEMBER LICENSE
 *AGREEMENT (the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License by contacting Principled
 *Technologies, Inc. Unless required by applicable law or agreed to in writing,
 *software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing grants and
 * restrictions under the License.
 *============================================================================*/

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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <time.h>
#include <vector>

#include <ext_list.hpp>
#include <format_reader_ptr.h>
#include <inference_engine.hpp>

#include "object_detection_ssd.h"
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
  // ---------------------------Parsing and validation of input
  // args--------------------------------------
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_h) {
    showUsage();
    return false;
  }

  slog::info << "Parsing input parameters" << slog::endl;

  if (FLAGS_ni < 1) {
    throw std::logic_error(
        "Parameter -ni should be greater than 0 (default: 1)");
  }

  if (FLAGS_i.empty()) {
    throw std::logic_error("Parameter -i is not set");
  }

  if (FLAGS_m.empty()) {
    throw std::logic_error("Parameter -m is not set");
  }

  return true;
}

// 0=================================================================================================================================================

// -------------------------- Compute percentile
// -----------------------------------------------------------------
double computePercentile(std::vector<double> arr, int perc) {

  float loc =
      (float(perc) / 100.0) * arr.size() - 1.0; // Index starts from zero
  int l = static_cast<int>(floor(loc));
  int h = static_cast<int>(ceil(loc));

  double lower = arr[l];
  double upper = arr[h];
  double value = lower + (upper - lower) * (loc - l);
  return value;
}

// 0=================================================================================================================================================

std::vector<std::string> split(std::string str, char delimiter) {
  std::vector<std::string> internal;
  std::stringstream ss(str); // Turn the string into a stream.
  std::string tok;
  while (getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  return internal;
}

// -------------------------- Main
// -----------------------------------------------------------------
int main(int argc, char *argv[]) {
  try {
    /** This sample covers certain topology and cannot be generalized for any
     * object detection one **/
    slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";

    // --------------------------- 1. Parsing and validation of input args
    // ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
      return 0;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 2. Read input
    // -----------------------------------------------------------
    /** This vector stores paths to the processed images **/
    std::vector<std::string> images;
    parseInputFilesArguments(images);
    if (images.empty())
      throw std::logic_error("No suitable images were found");
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Load Plugin for inference engine
    // -------------------------------------
    slog::info << "Loading plugin" << slog::endl;
    InferencePlugin plugin =
        PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""})
            .getPluginByDevice(FLAGS_d);
    if (FLAGS_p_msg) {
      static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)
          ->SetLogCallback(error_listener);
    }

    /*If CPU device, load default library with extensions that comes with the
     * product*/
    if (FLAGS_d.find("CPU") != std::string::npos) {
      /**
       * cpu_extensions library is compiled from "extension" folder containing
       * custom MKLDNNPlugin layer implementations. These layers are not
       *supported by mkldnn, but they can be useful for inferring custom
       *topologies.
       **/
      plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    if (!FLAGS_l.empty()) {
      // CPU(MKLDNN) extensions are loaded as a shared library and passed as a
      // pointer to base extension
      IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
      plugin.AddExtension(extension_ptr);
      slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
    }

    if (!FLAGS_c.empty()) {
      // clDNN Extensions are loaded from an .xml description and OpenCL kernel
      // files
      plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
      slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
    }

    /** Setting plugin parameter for per layer metrics **/
    if (FLAGS_pc) {
      plugin.SetConfig(
          {{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
    }

    /** Printing plugin version **/
    printPluginVersion(plugin, std::cout);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Read IR Generated by ModelOptimizer (.xml
    // and .bin files) ------------
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    slog::info << "Loading network files:"
                  "\n\t"
               << FLAGS_m << "\n\t" << binFileName << slog::endl;

    CNNNetReader networkReader;
    /** Read network model **/
    networkReader.ReadNetwork(FLAGS_m);

    /** Extract model name and load weights **/
    networkReader.ReadWeights(binFileName);
    CNNNetwork network = networkReader.getNetwork();
    network.setBatchSize(images.size());
    // -----------------------------------------------------------------------------------------------------
    // Adding harness variables
    std::string model_name, aarch, precision, output_dir;
    int batch_size = FLAGS_b;
    model_name = FLAGS_a;
    aarch = FLAGS_aarch;
    precision = FLAGS_prec;
    output_dir = FLAGS_dir;

    std::cout << "[ INFO ] " << model_name << "\t" << precision << "\t"
              << batch_size << "\t" << output_dir << std::endl;
    slog::info << "Model name " << FLAGS_a << slog::endl;

    // --------------------------- 5. Prepare input blobs
    // --------------------------------------------------
    slog::info << "Preparing input blobs" << slog::endl;

    /** Taking information about all topology inputs **/
    InputsDataMap inputsInfo(network.getInputsInfo());

    /** SSD network has one input and one output **/
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2)
      throw std::logic_error(
          "Sample supports topologies only with 1 or 2 inputs");

    /**
     * Some networks have SSD-like output format (ending with DetectionOutput
     * layer), but having 2 inputs as Faster-RCNN: one for image and one for
     * "image info".
     *
     * Although object_datection_sample_ssd's main task is to support clean SSD,
     * it could score the networks with two inputs as well. For such networks
     * imInfoInputName will contain the "second" input name.
     */
    std::string imageInputName, imInfoInputName;

    InputInfo::Ptr inputInfo = inputsInfo.begin()->second;

    SizeVector inputImageDims;
    /** Stores input image **/

    /** Iterating over all input blobs **/
    for (auto &item : inputsInfo) {
      /** Working with first input tensor that stores image **/
      if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
        imageInputName = item.first;

        slog::info << "Network Batch size is "
                   << std::to_string(networkReader.getNetwork().getBatchSize())
                   << slog::endl;

        /** Creating first input blob **/
        Precision inputPrecision = Precision::U8;
        item.second->setPrecision(inputPrecision);
      } else if (item.second->getInputData()
                     ->getTensorDesc()
                     .getDims()
                     .size() == 2) {
        imInfoInputName = item.first;

        Precision inputPrecision = Precision::FP32;
        item.second->setPrecision(inputPrecision);
        if ((item.second->getTensorDesc().getDims()[1] != 3 &&
             item.second->getTensorDesc().getDims()[1] != 6) ||
            item.second->getTensorDesc().getDims()[0] != 1) {
          throw std::logic_error(
              "Invalid input info. Should be 3 or 6 values length");
        }
      }
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Prepare output blobs
    // -------------------------------------------------
    slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputsInfo(network.getOutputsInfo());

    std::string outputName;
    DataPtr outputInfo;
    for (const auto &out : outputsInfo) {
      if (out.second->creatorLayer.lock()->type == "DetectionOutput") {
        outputName = out.first;
        outputInfo = out.second;
      }
    }

    if (outputInfo == nullptr) {
      throw std::logic_error(
          "Can't find a DetectionOutput layer in the topology");
    }

    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

    const int maxProposalCount = outputDims[2];
    const int objectSize = outputDims[3];

    if (objectSize != 7) {
      throw std::logic_error("Output item should have 7 as a last dimension");
    }

    if (outputDims.size() != 4) {
      throw std::logic_error("Incorrect output dimensions for SSD model");
    }

    /** Set the precision of output data provided by the user, should be called
     * before load of the network to the plugin **/
    outputInfo->setPrecision(Precision::FP32);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Loading model to the plugin
    // ------------------------------------------
    slog::info << "Loading model to the plugin" << slog::endl;

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // -----------------------------------------------------------------------------------------------------
    slog::info << "Model loaded successfully to plugin" << slog::endl;
    // --------------------------- 8. Create infer request
    // -------------------------------------------------
    InferRequest infer_request = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 9. Prepare input
    // --------------------------------------------------------
    /** Collect images data ptrs **/
    std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
    std::vector<int> imageWidths, imageHeights;
    for (auto &i : images) {
      FormatReader::ReaderPtr reader(i.c_str());
      if (reader.get() == nullptr) {
        slog::warn << "Image " + i + " cannot be read!" << slog::endl;
        continue;
      }
      /** Store image data **/
      std::shared_ptr<unsigned char> originalData(reader->getData());
      std::shared_ptr<unsigned char> data(
          reader->getData(inputInfo->getTensorDesc().getDims()[3],
                          inputInfo->getTensorDesc().getDims()[2]));
      if (data.get() != nullptr) {
        originalImagesData.push_back(originalData);
        imagesData.push_back(data);
        imageWidths.push_back(reader->width());
        imageHeights.push_back(reader->height());
      }
    }
    if (imagesData.empty())
      throw std::logic_error("Valid input images were not found!");

    size_t batchSize = network.getBatchSize();
    slog::info << "Network Batch size is " << std::to_string(batchSize)
               << slog::endl;
    if (batchSize != imagesData.size()) {
      slog::warn << "Number of images " + std::to_string(imagesData.size()) +
                        " doesn't match batch size " + std::to_string(batchSize)
                 << slog::endl;
      batchSize = std::min(batchSize, imagesData.size());
      slog::warn << "Number of images to be processed is "
                 << std::to_string(batchSize) << slog::endl;
    }
    // if (batch_size > batchSize){
    //  throw std::logic_error("Requested batch_size bigger than Network
    //  BatchSize");
    //}

    /** Creating input blob **/
    Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

    /** Filling input tensor with images. First b channel, then g and r channels
     * **/
    size_t num_channels = imageInput->getTensorDesc().getDims()[1];
    size_t image_size = imageInput->getTensorDesc().getDims()[3] *
                        imageInput->getTensorDesc().getDims()[2];

    unsigned char *data = static_cast<unsigned char *>(imageInput->buffer());

    /** Iterate over all input images **/
    for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize);
         ++image_id) {
      /** Iterate over all pixel in image (b,g,r) **/
      for (size_t pid = 0; pid < image_size; pid++) {
        /** Iterate over all channels **/
        for (size_t ch = 0; ch < num_channels; ++ch) {
          /** [images stride + channels stride + pixel id ] all in bytes **/
          data[image_id * image_size * num_channels + ch * image_size + pid] =
              imagesData.at(image_id).get()[pid * num_channels + ch];
        }
      }
    }

    if (imInfoInputName != "") {
      Blob::Ptr input2 = infer_request.GetBlob(imInfoInputName);
      auto imInfoDim = inputsInfo.find(imInfoInputName)
                           ->second->getTensorDesc()
                           .getDims()[1];

      /** Fill input tensor with values **/
      float *p =
          input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

      for (size_t image_id = 0;
           image_id < std::min(imagesData.size(), batchSize); ++image_id) {
        p[image_id * imInfoDim + 0] = static_cast<float>(
            inputsInfo[imageInputName]->getTensorDesc().getDims()[2]);
        p[image_id * imInfoDim + 1] = static_cast<float>(
            inputsInfo[imageInputName]->getTensorDesc().getDims()[3]);
        for (int k = 2; k < imInfoDim; k++) {
          p[image_id * imInfoDim + k] =
              1.0f; // all scale factors are set to 1.0
        }
      }
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 10. Do inference
    // ---------------------------------------------------------
    slog::info << "Start inference (" << FLAGS_ni << " iterations)"
               << slog::endl;

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;

    double total = 0.0;
    // 1=================================================================================================================================================
    std::vector<double> sample = {}; // compute STDev

    /** warmup the inference engine **/
    for (int iter = 0; iter < 10; ++iter) {
      infer_request.Infer();
    }

    std::ofstream latency_times;
    latency_times.open(FLAGS_r, std::ios_base::trunc);

    /** Start inference & calc performance **/
    for (int iter = 0; iter < FLAGS_ni; ++iter) {
      auto t0 = Time::now();
      infer_request.Infer();
      auto t1 = Time::now();
      fsec fs = t1 - t0;
      ms d = std::chrono::duration_cast<ms>(fs);
      total += d.count();
      sample.push_back(d.count()); // = d.count();
      latency_times << std::setprecision(5) << d.count() / 1000.0 << " ";
      if ((iter % 10 == 0) && (iter > 0))
        latency_times << "\n";
    }
    latency_times << "\n";
    latency_times.close();

    // 1=================================================================================================================================================

    // --------------------------- 11. Process output
    // -------------------------------------------------------
    slog::info << "Processing output blobs" << slog::endl;

    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
    const float *detection =
        static_cast<PrecisionTrait<Precision::FP32>::value_type *>(
            output_blob->buffer());

    std::vector<std::vector<int>> boxes(batchSize);
    std::vector<std::vector<int>> classes(batchSize);

    /**Write Results to output **/
    std::ofstream outfile;

    std::string outfilename = FLAGS_dump;

    outfile.open(outfilename, std::ios_base::app | std::ios_base::out);

    std::vector<std::string> sep;       // split(images[image_id], '/');
    std::vector<std::string> file_name; // = split(sep.back(), '.');

    /* Each detection has image_id that denotes processed image */
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
      float image_id = detection[curProposal * objectSize + 0];
      if (image_id < 0) {
        break;
      }

      float label = detection[curProposal * objectSize + 1];
      float confidence = detection[curProposal * objectSize + 2];
      /* Normalized bounding box */
      float fl_xmin =
          detection[curProposal * objectSize + 3]; // * imageWidths[image_id];
      float fl_ymin =
          detection[curProposal * objectSize + 4]; // * imageHeights[image_id];
      float fl_xmax =
          detection[curProposal * objectSize + 5]; // * imageWidths[image_id];
      float fl_ymax =
          detection[curProposal * objectSize + 6]; // * imageHeights[image_id];

      /* Rescaled bounding boxes */

      float xmin = fl_xmin * imageWidths[image_id];
      float ymin = fl_ymin * imageHeights[image_id];
      float xmax = fl_xmax * imageWidths[image_id];
      float ymax = fl_ymax * imageHeights[image_id];

      /*
      std::cout << "[" << curProposal << "," << label << "] element, prob = " <<
      confidence << "    (" << xmin << "," << ymin << ")-(" << xmax << "," <<
      ymax << ")" << " batch id : " << image_id;
      */
      if (confidence > 0.2) {
        /** Drawing only objects with >50% probability **/
        classes[image_id].push_back(static_cast<int>(label));
        boxes[image_id].push_back(static_cast<int>(xmin));
        boxes[image_id].push_back(static_cast<int>(ymin));
        boxes[image_id].push_back(static_cast<int>(xmax - xmin));
        boxes[image_id].push_back(static_cast<int>(ymax - ymin));
        /**
        std::cout << "[" << curProposal << "," << label << "] element, prob = "
        << confidence << "    (" << xmin << "," << ymin << ")-(" << xmax << ","
        << ymax << ")" << " batch id : " << image_id << "classid: " << label;
        std::cout << " WILL BE PRINTED!" <<std::endl;**/

        sep = split(images[image_id], '/');
        file_name = split(sep.back(), '.');
        // outfile << file_name[0]+".JPEG,";
        outfile << file_name[0] + ",";
        outfile << std::setprecision(0) << std::fixed << label << ","
                << std::setprecision(4) << std::fixed << fl_xmin << ","
                << fl_ymin << "," << fl_xmax << "," << fl_ymax << ","
                << confidence << std::endl;
        // outfile << std::setprecision(0) << std::fixed << label << "," <<
        // std::setprecision(4) << std::fixed << xmin << "," << ymin << "," <<
        // xmax << "," << ymax << "," << confidence << std::endl;
      }
      // std::cout << std::endl;
    }

    outfile.close();

    // -----------------------------------------------------------------------------------------------------
    std::cout << std::endl << "total inference time: " << total << std::endl;
    std::cout << "Average running time of one iteration: "
              << total / static_cast<double>(FLAGS_ni) << " ms" << std::endl;
    std::cout << std::endl
              << "Throughput: "
              << 1000 * static_cast<double>(FLAGS_ni) * batch_size / total
              << " FPS" << std::endl;
    std::cout << std::endl;

    /** Show performance results **/
    if (FLAGS_pc) {
      printPerformanceCounts(infer_request, std::cout);
    }
    // -----------------------------------------------------------------------------------------------------
  } catch (const std::exception &error) {
    slog::err << error.what() << slog::endl;
    return 1;
  } catch (...) {
    slog::err << "Unknown/internal exception happened." << slog::endl;
    return 1;
  }

  slog::info << "Execution successful" << slog::endl;
  return 0;
}
