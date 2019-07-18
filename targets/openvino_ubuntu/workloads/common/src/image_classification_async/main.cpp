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

/**
 * @brief The entry point the Inference Engine sample application
 * @file classification_sample/main.cpp
 * @example classification_sample/main.cpp
 */
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>

#include <ext_list.hpp>
#include <sys/stat.h>

#include "image_classification_async.h"
#include <opencv2/opencv.hpp>
#include <vpu/vpu_plugin_config.hpp>

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

void cropImage(cv::Mat &image, float r) {
  int y, x, ybegin, yend, xbegin, xend; // size of images
  float keep_ratio = 1 - r;

  y = image.size().height;
  x = image.size().width;

  ybegin = 0.5 * keep_ratio * ((float)y);
  yend = y - (int)ybegin;

  xbegin = 0.5 * keep_ratio * ((float)x);
  xend = x - (int)xbegin;

  cv::Rect roi(cv::Point(xbegin, ybegin), cv::Point(xend, yend));
  image(roi).copyTo(image);
}

void aspect_preserving_resize(cv::Mat &imdata) {
  float height = (float)imdata.size().height;
  float width = (float)imdata.size().width;
  float min_dim = std::min(height, width);

  float _MIN_RESIZE_SIZE = 256.0;

  float aspect_ratio = _MIN_RESIZE_SIZE / min_dim;

  int new_height, new_width;
  new_height = 256;
  new_width = 256;

  cv::Size new_dims;
  new_dims.height = new_height;
  new_dims.width = new_width;

  cv::resize(imdata, imdata, new_dims);
}

void central_crop(cv::Mat &imdata, int crop_height, int crop_width) {

  int height, width, height_offset, width_offset;

  height = imdata.size().height;
  width = imdata.size().width;

  height_offset = 0.5 * (height - crop_height);
  width_offset = 0.5 * (width - crop_width);

  cv::Rect roi(width_offset, height_offset, crop_width, crop_height);
  imdata = imdata(roi);
}

void preprocess_imagedata(cv::Mat &imdata, std::string modelname) {

  imdata.convertTo(imdata, CV_32F);

  if (modelname == "resnet50") {
    /* Preprocessing for resnet50 goes here */
  } else if (modelname == "mobilenet") {
    /* Preprocessing for mobilenet goes here */
  } else {
  }
}

void ResizeCrop(cv::Mat &imdata, cv::Size inputSize,
                std::string modelname = "resnet50") {

  cv::resize(imdata, imdata, cv::Size(256, 256));
  central_crop(imdata, inputSize.height, inputSize.width);

  preprocess_imagedata(imdata, modelname);
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
  // ---------------------------Parsing and validation of input
  // args--------------------------------------
  slog::info << "Parsing input parameters" << slog::endl;

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_h) {
    showUsage();
    return false;
  }
  slog::info << "Parsing input parameters" << slog::endl;

  if (FLAGS_ni < 1) {
    throw std::logic_error("Parameter -ni must be more than 0 ! (default 1)");
  }

  if (FLAGS_nireq < 1) {
    throw std::logic_error(
        "Parameter -nireq must be more than 0 ! (default 1)");
  }

  if (FLAGS_i.empty()) {
    throw std::logic_error("Parameter -i is not set");
  }

  if (FLAGS_m.empty()) {
    throw std::logic_error("Parameter -m is not set");
  }

  if (FLAGS_ni < FLAGS_nireq) {
    throw std::logic_error(
        "Number of iterations could not be less than requests quantity");
  }

  return true;
}

int main(int argc, char *argv[]) {
  try {
    slog::info << "InferenceEngine: " << GetInferenceEngineVersion()
               << slog::endl;

    // ------------------------------ Parsing and validation of input args
    // ---------------------------------
    slog::info << "Parsing input parameters " << slog::endl;
    std::string model_name, aarch, precision, result_path, mode, classes;
    int batch_size = 0;

    if (!ParseAndCheckCommandLine(argc, argv)) {
      return 1;
    }

    model_name = FLAGS_a;
    batch_size = FLAGS_b;
    precision = FLAGS_prec;
    result_path = FLAGS_r;
    mode = FLAGS_mode;
    classes = FLAGS_cl;

    /** This vector stores paths to the processed images **/
    std::vector<std::string> imageNames;
    parseInputFilesArguments(imageNames);
    if (imageNames.empty())
      throw std::logic_error("No suitable images were found");

    // --------------------------- 1. Load Plugin for inference engine
    // -------------------------------------
    slog::info << "Loading plugin" << slog::endl;
    InferencePlugin plugin =
        PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""})
            .getPluginByDevice(FLAGS_d);
    if (FLAGS_p_msg) {
      static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)
          ->SetLogCallback(error_listener);
    }

    /** Loading default extensions **/
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

    ResponseDesc resp;
    /** Printing plugin version **/
    printPluginVersion(plugin, std::cout);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml
    // and .bin files) ------------
    slog::info << "Loading network files" << slog::endl;

    CNNNetReader networkReader;
    /** Read network model **/
    networkReader.ReadNetwork(FLAGS_m);

    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    networkReader.ReadWeights(binFileName);

    CNNNetwork network = networkReader.getNetwork();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Configure input & output
    // ---------------------------------------------

    // --------------------------- Prepare input blobs
    // -----------------------------------------------------
    slog::info << "Preparing input blobs" << slog::endl;

    /** Taking information about all topology inputs **/
    InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1)
      throw std::logic_error("Sample supports topologies only with 1 input");

    auto inputInfoItem = *inputInfo.begin();

    /** Specifying the precision and layout of input data provided by the user.
     * This should be called before load of the network to the plugin **/
    inputInfoItem.second->setPrecision(Precision::FP32);
    inputInfoItem.second->setLayout(Layout::NCHW);

    std::vector<cv::Mat> imagesData;
    cv::Mat image, imdata;

    for (auto &i : imageNames) {
      image = cv::imread(i);
      if (image.empty()) {
        throw std::logic_error("Invalid image at path: " + i);
      }

      /** Preprocess images and store **/
      cv::Size inputSize(
          cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3],
                   inputInfoItem.second->getTensorDesc().getDims()[2]));

      ResizeCrop(image, inputSize, model_name);
      imagesData.push_back(image);
    }

    if (imagesData.empty())
      throw std::logic_error("Valid input images were not found!");

    /** Setting batch size using image count **/
    network.setBatchSize(imagesData.size());
    size_t batchSize = network.getBatchSize();
    slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

    // ------------------------------ Prepare output blobs
    // -------------------------------------------------
    slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputInfo(network.getOutputsInfo());
    std::vector<Blob::Ptr> outputBlobs;
    for (size_t i = 0; i < FLAGS_nireq; i++) {
      auto outputBlob =
          make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(
              outputInfo.begin()->second->getTensorDesc());
      outputBlob->allocate();
      outputBlobs.push_back(outputBlob);
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Loading model to the plugin
    // ------------------------------------------
    slog::info << "Loading model to the plugin" << slog::endl;

    std::map<std::string, std::string> config;
    if (FLAGS_pc) {
      config[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
    }

    // set config values
    // Enable HW acceleration
    if ((FLAGS_d.find("MYRIAD") != std::string::npos) or
        (FLAGS_d.find("HDDL") != std::string::npos)) {
      config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
      config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
      config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] =
          CONFIG_VALUE(YES); // This is the important one for HW acceleration
      config[VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME)] = CONFIG_VALUE(YES);
    }

    if (FLAGS_d.find("CPU") != std::string::npos) { // CPU supports few special
                                                    // performance-oriented keys
      // limit threading for CPU portion of inference
      config[PluginConfigParams::KEY_CPU_THREADS_NUM] =
          std::to_string(FLAGS_nthreads);
      // pin threads for CPU portion of inference
      config[PluginConfigParams::KEY_CPU_BIND_THREAD] = FLAGS_pin;
      // for pure CPU execution, more throughput-oriented execution via streams
      if (FLAGS_d == "CPU")
        config[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] =
            std::to_string(FLAGS_nireq);
    }

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, config);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Create infer request
    // -------------------------------------------------
    std::vector<InferRequest> inferRequests;
    for (size_t i = 0; i < FLAGS_nireq; i++) {
      InferRequest inferRequest = executable_network.CreateInferRequest();
      inferRequests.push_back(inferRequest);
    }

    // --------------------------- 6. Prepare input
    // --------------------------------------------------------
    slog::info << "Preparing input data" << slog::endl;
    BlobMap inputBlobs;
    // InferRequest infer_request = executable_network.CreateInferRequest();
    for (auto &item : inputInfo) {
      // Blob::Ptr input = inferRequests[0].GetBlob(item.first);
      // auto input =
      // make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(item.second->getTensorDesc());
      // // Original
      auto input =
          make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(
              item.second->getTensorDesc());

      input->allocate();

      // auto data =
      // input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
      inputBlobs[item.first] = input;

      auto dims = input->getTensorDesc().getDims();
      /** Fill input tensor with images. First b channel, then g and r channels
       * **/
      size_t num_channels = dims[1];
      size_t image_size = dims[3] * dims[2];

      /** Iterate over all input images **/
      for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (size_t pid = 0; pid < image_size; pid++) {
          /** Iterate over all channels **/
          for (size_t ch = 0; ch < num_channels; ++ch) {
            /**          [images stride + channels stride + pixel id ] all in
             * bytes            **/
            // data[image_id * image_size * num_channels + ch * image_size +
            // pid] = imagesData.at(image_id).at<cv::Vec3f>(pid)[ch];
            // //imagesData.at(image_id).get()[pid(num_channel + ch];
            input->data()[image_id * image_size * num_channels +
                          ch * image_size + pid] =
                imagesData.at(image_id).at<cv::Vec3f>(pid)[ch]; // imagesData.at(image_id).get()[pid(num_channel
                                                                // + ch];
            // data[image_id * image_size * num_channels + ch * image_size + pid
            // ] = imagesData.at(image_id).at<cv::Vec3f>(pid)[ch];
          }
        }
      }
    }

    for (size_t i = 0; i < FLAGS_nireq; i++) {
      inferRequests[i].SetBlob(inputBlobs.begin()->first,
                               inputBlobs.begin()->second);
      inferRequests[i].SetBlob(outputInfo.begin()->first, outputBlobs[i]);
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Do inference
    // ---------------------------------------------------------
    slog::info << "Start inference (" << FLAGS_ni << " iterations)"
               << slog::endl;

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;

    size_t currentInfer = 0;
    size_t prevInfer = (FLAGS_nireq > 1) ? 1 : 0;

    // warming up
    inferRequests[0].StartAsync();
    inferRequests[0].Wait(
        InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    double total_single_iteration = 0.0;
    double total = 0.0;
    double times[FLAGS_ni];
    if (mode == "throughput" || mode == "latency") {
      std::cout << "Running throughput mode" << std::endl;
      auto t0 = Time::now();

      std::map<int, std::vector<decltype(t0)>> req_start_times;
      // std::map < int, std::vector<double> > req_total_times;
      std::vector<double> req_total_times;
      ms req_d;
      std::chrono::duration<double> st_d;

      slog::info << " Initializing request times " << slog::endl;
      for (int i = 0; i < FLAGS_nireq; i++) {
        std::vector<double> times_vec = {};
        std::vector<decltype(t0)> runtime_vec{Time::now()};

        req_start_times[i] = runtime_vec;
        // req_total_times[i] = times_vec;
      }

      /** Start inference & calc performance **/
      for (int iter = 0; iter < FLAGS_ni + FLAGS_nireq; ++iter) {
        if (iter < FLAGS_ni) {

          req_start_times[currentInfer].push_back(Time::now());
          inferRequests[currentInfer].StartAsync();
        }

        inferRequests[prevInfer].Wait(
            InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
        req_d = std::chrono::duration_cast<ms>(
            Time::now() - req_start_times[prevInfer]
                              .back()); // Time it took for prevInfer inference
        req_total_times.push_back(req_d.count());

        currentInfer++;
        if (currentInfer >= FLAGS_nireq) {
          currentInfer = 0;
        }
        prevInfer++;
        if (prevInfer >= FLAGS_nireq) {
          prevInfer = 0;
        }
      }
      auto t1 = Time::now();
      fsec fs = t1 - t0;
      ms d = std::chrono::duration_cast<ms>(fs);
      total = d.count();

      std::ofstream resultsFile;
      resultsFile.open(result_path);
      // loop through the elements of the array
      resultsFile << std::setprecision(5) << total / 1000.0 << " ";
      int iter = 0;
      for (auto it = req_total_times.cbegin() + 1; it != req_total_times.cend();
           ++it) {
        if (iter > FLAGS_nireq) {
          resultsFile << std::setprecision(5) << (*it) / 1000.0
                      << " "; // d.count()/1000.0 << " ";
        }
        iter++;
      }
      resultsFile.close();
      // -----------------------------------------------------------------------------------------------------
      std::cout << "total: " << total << std::endl;
      /** Show performance results **/
      std::cout << std::endl
                << "Throughput: "
                << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total
                << " FPS" << std::endl;
    } else if (mode == "accuracy") {
      std::cout << "Running accuracy mode" << std::endl;
      /** Start inference & calc performance **/
      inferRequests[0].StartAsync();
      inferRequests[0].Wait(
          InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    } else {
      std::cout << "Unable to detect Mode" << std::endl;
    }

    // --------------------------- 8. Process output
    // -------------------------------------------------------
    slog::info << "Processing output blobs" << slog::endl;

    for (size_t i = 0; i < FLAGS_nireq; i++) {
      /** Validating -nt value **/
      const int resultsCnt = outputBlobs[i]->size() / batchSize;
      if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
        slog::warn
            << "-nt " << FLAGS_nt
            << " is not available for this network (-nt should be less than "
            << resultsCnt + 1
            << " and more than 0)\n            will be used maximal value : "
            << resultsCnt << slog::endl;
        FLAGS_nt = resultsCnt;
      }
      /** This vector stores id's of top N results **/
      std::vector<unsigned> results;
      TopResults(FLAGS_nt, *outputBlobs[i], results);

      std::cout << std::endl
                << "Top " << FLAGS_nt << " results:" << std::endl
                << std::endl;

      /** Read labels from file (e.x. AlexNet.labels) **/
      bool labelsEnabled = false;
      std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
      std::vector<std::string> labels;

      std::ifstream inputFile;
      inputFile.open(labelFileName, std::ios::in);
      if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
          trim(strLine);
          labels.push_back(strLine);
        }
        labelsEnabled = true;
      }

      // loop through the elements of the array
      for (size_t i = 0; i < FLAGS_ni; i++) {
      }
      /** Print the result iterating over each batch **/
      std::cout << "FLAGS_nireq: " << i << std::endl;
      int batchSize = network.getBatchSize();
      std::string classes_bkg("1001");
      if (mode == "accuracy") {
        for (int image_id = 0; image_id < batchSize; ++image_id) {
          std::cout << "Image " << imageNames[image_id] << std::endl
                    << std::endl;
          std::ofstream resultsFile;
          resultsFile.open(result_path);
          resultsFile << imageNames[image_id] << ' ';
          for (size_t id = image_id * FLAGS_nt, cnt = 0; cnt < FLAGS_nt;
               ++cnt, ++id) {
            std::cout.precision(7);
            /** Getting probability for resulting class **/
            auto result =
                outputBlobs[i]
                    ->buffer()
                    .as<PrecisionTrait<Precision::FP32>::value_type *>()
                        [results[id] +
                         image_id * (outputBlobs[i]->size() / batchSize)];
            std::cout << std::left << std::fixed << results[id] << " "
                      << result;
            if (labelsEnabled) {
              std::cout << " label " << labels[results[id]] << std::endl;
              if (classes == classes_bkg) {
                resultsFile << labels[results[id] - 1] << ' ';
              } else {
                resultsFile << labels[results[id]] << ' ';
              }
            } else {
              std::cout << " label #" << results[id] << std::endl;
              if (classes == classes_bkg) {
                resultsFile << results[id] - 1 << ' ';
              } else {
                resultsFile << results[id] << ' ';
              }
            }
          }
          resultsFile.close();
          std::cout << std::endl;
        }
      }
    }
    // -----------------------------------------------------------------------------------------------------
    std::cout << std::endl << "total inference time: " << total << std::endl;
    std::cout << std::endl
              << "Throughput: "
              << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total
              << " FPS" << std::endl;
    std::cout << std::endl;

    /** Show performance results **/
    std::map<std::string, InferenceEngineProfileInfo> performanceMap;
    if (FLAGS_pc) {
      for (size_t nireq = 0; nireq < FLAGS_nireq; nireq++) {
        performanceMap = inferRequests[nireq].GetPerformanceCounts();
        printPerformanceCounts(performanceMap, std::cout);
      }
    }
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
