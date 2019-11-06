/**
 * Copyright (C) 2019 EEMBC(R). All Rights Reserved
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 *
 * Original Author: Peter Torelli
 *
 */

// ArmNN
#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>

// stdlib
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include <string.h>

#include <iostream>
// boost
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/irange.hpp>
// OpenCV
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

// global variables
string g_model_filename;
int g_batch_size;
int g_iterations;
string g_hardware;
string g_mode;
string g_precision;
int g_debug(0);
boost::property_tree::ptree g_results_tree;

// error handling. Just strings.
void LogErrorMessages(const std::vector<std::string> &errorMessages) {
  for (const auto &errorMessage : errorMessages) {
    cout << errorMessage << endl;
  }
}

// making input tensors. ARMNN api
armnn::InputTensors
MakeInputTensors(const pair<armnn::LayerBindingId, armnn::TensorInfo> &input,
                 const void *inputTensorData) {
  return {{input.first, armnn::ConstTensor(input.second, inputTensorData)}};
}

// output tensors.
armnn::OutputTensors
MakeOutputTensors(const pair<armnn::LayerBindingId, armnn::TensorInfo> &output,
                  void *outputTensorData) {
  return {{output.first, armnn::Tensor(output.second, outputTensorData)}};
}

map<float, int> maxSort(vector<float> &predictions) {
  map<float, int> r;
  // recall: maps sort by Less<Key> by default, and we don't
  // care if there are multiple predictions with the same
  // confidence, which is why we don't use a multimap
  for (int i = 0; i < predictions.size(); ++i) {
    r.insert(make_pair(predictions[i], i));
  }
  return r;
}

std::vector<float> loadImage(string &fn) {
  cout << "Load " << fn << endl;
  cv::Mat img;
  std::vector<float> data;
  double fx, fy;
  img = cv::imread(fn.c_str());

  fx = 300.0 / img.cols;
  fy = 300.0 / img.rows;

  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  img.convertTo(img, CV_32F); // very very important! otherwise pixels are not
                              // treated as floats.
  cv::resize(img, img, cv::Size(), fx, fy, cv::INTER_LINEAR);
  img = (img * (2.0 / 255.0)) - 1;

  data.assign(
      (float *)img.data,
      (float *)img.data +
          (300 * 300 * 3)); // this is just share the image data out of image.
  return data;
}

std::vector<uint8_t> loadImage_int(string &fn) {
  cout << "Load " << fn << endl;
  cv::Mat img;
  std::vector<uint8_t> data;
  double fx, fy;
  img = cv::imread(fn.c_str());

  fx = 300.0 / img.cols;
  fy = 300.0 / img.rows;

  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, cv::Size(), fx, fy, cv::INTER_LINEAR);

  data.assign(
      (uint8_t *)img.data,
      (uint8_t *)img.data +
          (300 * 300 * 3)); // this is just share the image data out of image.
  return data;
}

void runBatchInference(vector<string> &filenames) {

  // cout<<"entered in run batch"<<endl;
  // TODO: Placeholder
  const int imgSize(300 * 300 * 3);

  vector<float> imgData;
  float *d = new float[imgSize];

  // Load the model
  armnnTfLiteParser::ITfLiteParserPtr parser =
      armnnTfLiteParser::ITfLiteParser::Create();

  // Create the network from a flatbuffers binary. TF creates network from
  // protobuf binary.
  armnn::INetworkPtr network =
      parser->CreateNetworkFromBinaryFile(g_model_filename.c_str());

  // Bind the tensors
  armnnTfLiteParser::BindingPointInfo inputBindingInfo =
      parser->GetNetworkInputBindingInfo(0, "normalized_input_image_tensor");

  armnnTfLiteParser::BindingPointInfo outputBindingInfo0 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess");
  armnnTfLiteParser::BindingPointInfo outputBindingInfo1 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:1");
  armnnTfLiteParser::BindingPointInfo outputBindingInfo2 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:2");
  armnnTfLiteParser::BindingPointInfo outputBindingInfo3 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:3");

  // Optimize the model
  armnn::IRuntime::CreationOptions options;
  // is this causing always-on gpu? options.m_EnableGpuProfiling = true;
  armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
  std::vector<std::string> errorMessages;
  armnn::OptimizerOptions oopts;

  if (g_precision == "fp16") {
    cout << "Enabling fp16 optimization" << endl;
    oopts.m_ReduceFp32ToFp16 = true;
  } else {
    cout << "Disabling fp16 optimization." << endl;
    oopts.m_ReduceFp32ToFp16 = false;
  }

  armnn::Compute hardware;
  if (g_hardware == "gpu") {
    cout << "Enabling GpuAcc (mali/opencl)" << endl;
    hardware = armnn::Compute::GpuAcc;
  } else {
    cout << "Enabling CpuAcc" << endl;
    hardware = armnn::Compute::CpuAcc;
  }
  cout << "Compute device: " << hardware << endl;

  armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(
      *network, {armnn::Compute::CpuAcc, armnn::Compute::CpuRef},
      runtime->GetDeviceSpec(), oopts,
      armnn::Optional<std::vector<std::string> &>(errorMessages));
  LogErrorMessages(errorMessages);

  armnn::NetworkId networkIdentifier;
  runtime->LoadNetwork(networkIdentifier, move(optNet));

  // TODO What about batch size?

  std::vector<float> output0(
      outputBindingInfo0.second.GetShape().GetNumElements());
  std::vector<float> output1(
      outputBindingInfo1.second.GetShape().GetNumElements());
  std::vector<float> output2(
      outputBindingInfo2.second.GetShape().GetNumElements());
  std::vector<float> output3(
      outputBindingInfo3.second.GetShape().GetNumElements());

  // Make armnn output tensors.
  armnn::OutputTensors outputTensors{
      {outputBindingInfo0.first,
       armnn::Tensor(outputBindingInfo0.second, output0.data())},
      {outputBindingInfo1.first,
       armnn::Tensor(outputBindingInfo1.second, output1.data())},
      {outputBindingInfo2.first,
       armnn::Tensor(outputBindingInfo2.second, output2.data())},
      {outputBindingInfo3.first,
       armnn::Tensor(outputBindingInfo3.second, output3.data())}};

  chrono::time_point<chrono::high_resolution_clock> t0, t_start;
  chrono::time_point<chrono::high_resolution_clock> tn, t_stop;
  double dt;

  if (g_mode == "throughput" || g_mode == "latency") {
    // TODO: Batch soze 1
    imgData = loadImage(filenames[0]);
    // TODO: Passing imgData.data() is causing failures. Why?
    memcpy(d, imgData.data(), imgData.size() * sizeof(float));
    // Storage tree
    boost::property_tree::ptree t_node;
    boost::property_tree::ptree t_nodes;
    // Warmup
    t0 = chrono::high_resolution_clock::now();

    armnn::Status ret = runtime->EnqueueWorkload(
        networkIdentifier, MakeInputTensors(inputBindingInfo, d),
        outputTensors);
    tn = chrono::high_resolution_clock::now();
    dt = chrono::duration<double>(tn - t0).count();
    t_node.put("", dt);
    t_nodes.push_back(make_pair("", t_node));
    t_start = chrono::high_resolution_clock::now();

    for (auto i : boost::irange(0, g_iterations)) {
      t0 = chrono::high_resolution_clock::now();
      armnn::Status ret = runtime->EnqueueWorkload(
          networkIdentifier, MakeInputTensors(inputBindingInfo, imgData.data()),
          outputTensors);
      tn = chrono::high_resolution_clock::now();
      dt = chrono::duration<double>(tn - t0).count();
      t_node.put("", dt);
      t_nodes.push_back(make_pair("", t_node));
    } // for loop
    t_stop = chrono::high_resolution_clock::now();
    dt = chrono::duration<double>(t_stop - t_start).count();
    t_node.put("", dt);
    t_nodes.push_front(make_pair("", t_node));
    g_results_tree.add_child("times", t_nodes);

  } // if

  else if (g_mode == "accuracy") {
    boost::property_tree::ptree final_list_node;

    for (auto i :
         boost::irange(0, boost::numeric_cast<int>(filenames.size()))) {
      vector<float> imgData = loadImage(filenames[i]);

      // TODO: Passing imgData.data() is causing failures. Why?
      memcpy(d, imgData.data(), 3 * 300 * 300 * sizeof(float));
      armnn::Status ret = runtime->EnqueueWorkload(
          networkIdentifier, MakeInputTensors(inputBindingInfo, d),
          outputTensors);

      boost::property_tree::ptree prediction_list_node;
      for (int i = 0; i < output3[0]; i++) {
        boost::property_tree::ptree prediction_node;
        boost::property_tree::ptree x1, y1, x2, y2;
        boost::property_tree::ptree bbox_list_node;
        // TODO verify order
        x1.put("", output0[i * 4 + 1]);
        y1.put("", output0[i * 4 + 0]);
        x2.put("", output0[i * 4 + 3]);
        y2.put("", output0[i * 4 + 2]);
        bbox_list_node.push_back(make_pair("", x1));
        bbox_list_node.push_back(make_pair("", y1));
        bbox_list_node.push_back(make_pair("", x2));
        bbox_list_node.push_back(make_pair("", y2));
        prediction_node.add_child("box", bbox_list_node);
        prediction_node.put("class", output1[i] + 1);
        prediction_node.put("score", output2[i]);
        prediction_list_node.push_back(make_pair("", prediction_node));
      }

      boost::property_tree::ptree filename_node;
      filename_node.put("", filenames[i]);
      boost::property_tree::ptree file_predictions_node;
      file_predictions_node.push_back(make_pair("", filename_node));
      file_predictions_node.push_back(make_pair("", prediction_list_node));
      final_list_node.push_back(make_pair("", file_predictions_node));
    } // for
    g_results_tree.add_child("predictions", final_list_node);
  } // else if accuracy
}

void runBatchInference_int(vector<string> &filenames) {

  // cout<<"entered in run batch"<<endl;
  // TODO: Placeholder
  const int imgSize(300 * 300 * 3);

  vector<uint8_t> imgData;
  uint8_t *d = new uint8_t[imgSize];

  // Load the model
  armnnTfLiteParser::ITfLiteParserPtr parser =
      armnnTfLiteParser::ITfLiteParser::Create();

  // Create the network from a flatbuffers binary. TF creates network from
  // protobuf binary.
  armnn::INetworkPtr network =
      parser->CreateNetworkFromBinaryFile(g_model_filename.c_str());

  // Bind the tensors
  armnnTfLiteParser::BindingPointInfo inputBindingInfo =
      parser->GetNetworkInputBindingInfo(0, "normalized_input_image_tensor");

  armnnTfLiteParser::BindingPointInfo outputBindingInfo0 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess");
  armnnTfLiteParser::BindingPointInfo outputBindingInfo1 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:1");
  armnnTfLiteParser::BindingPointInfo outputBindingInfo2 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:2");
  armnnTfLiteParser::BindingPointInfo outputBindingInfo3 =
      parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:3");

  // Optimize the model
  armnn::IRuntime::CreationOptions options;
  // is this causing always-on gpu? options.m_EnableGpuProfiling = true;
  armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
  std::vector<std::string> errorMessages;
  armnn::OptimizerOptions oopts;

  if (g_precision == "fp16") {
    cout << "Enabling fp16 optimization" << endl;
    oopts.m_ReduceFp32ToFp16 = true;
  } else {
    cout << "Disabling fp16 optimization." << endl;
    oopts.m_ReduceFp32ToFp16 = false;
  }

  armnn::Compute hardware;
  if (g_hardware == "gpu") {
    cout << "Enabling GpuAcc (mali/opencl)" << endl;
    hardware = armnn::Compute::GpuAcc;
  } else {
    cout << "Enabling CpuAcc" << endl;
    hardware = armnn::Compute::CpuAcc;
  }
  cout << "Compute device: " << hardware << endl;

  armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(
      *network, {armnn::Compute::CpuAcc, armnn::Compute::CpuRef},
      runtime->GetDeviceSpec(), oopts,
      armnn::Optional<std::vector<std::string> &>(errorMessages));
  LogErrorMessages(errorMessages);

  armnn::NetworkId networkIdentifier;
  runtime->LoadNetwork(networkIdentifier, move(optNet));

  // TODO What about batch size?

  std::vector<float> output0(
      outputBindingInfo0.second.GetShape().GetNumElements());
  std::vector<float> output1(
      outputBindingInfo1.second.GetShape().GetNumElements());
  std::vector<float> output2(
      outputBindingInfo2.second.GetShape().GetNumElements());
  std::vector<float> output3(
      outputBindingInfo3.second.GetShape().GetNumElements());

  // Make armnn output tensors.
  armnn::OutputTensors outputTensors{
      {outputBindingInfo0.first,
       armnn::Tensor(outputBindingInfo0.second, output0.data())},
      {outputBindingInfo1.first,
       armnn::Tensor(outputBindingInfo1.second, output1.data())},
      {outputBindingInfo2.first,
       armnn::Tensor(outputBindingInfo2.second, output2.data())},
      {outputBindingInfo3.first,
       armnn::Tensor(outputBindingInfo3.second, output3.data())}};

  chrono::time_point<chrono::high_resolution_clock> t0, t_start;
  chrono::time_point<chrono::high_resolution_clock> tn, t_stop;
  double dt;

  if (g_mode == "throughput" || g_mode == "latency") {
    // TODO: Batch soze 1
    imgData = loadImage_int(filenames[0]);
    // TODO: Passing imgData.data() is causing failures. Why?
    memcpy(d, imgData.data(), imgData.size() * sizeof(uint8_t));
    // Storage tree
    boost::property_tree::ptree t_node;
    boost::property_tree::ptree t_nodes;
    // Warmup
    t0 = chrono::high_resolution_clock::now();

    armnn::Status ret = runtime->EnqueueWorkload(
        networkIdentifier, MakeInputTensors(inputBindingInfo, d),
        outputTensors);
    tn = chrono::high_resolution_clock::now();
    dt = chrono::duration<double>(tn - t0).count();
    t_node.put("", dt);
    t_nodes.push_back(make_pair("", t_node));
    t_start = chrono::high_resolution_clock::now();

    for (auto i : boost::irange(0, g_iterations)) {
      t0 = chrono::high_resolution_clock::now();
      armnn::Status ret = runtime->EnqueueWorkload(
          networkIdentifier, MakeInputTensors(inputBindingInfo, imgData.data()),
          outputTensors);
      tn = chrono::high_resolution_clock::now();
      dt = chrono::duration<double>(tn - t0).count();
      t_node.put("", dt);
      t_nodes.push_back(make_pair("", t_node));
    } // for loop
    t_stop = chrono::high_resolution_clock::now();
    dt = chrono::duration<double>(t_stop - t_start).count();
    t_node.put("", dt);
    t_nodes.push_front(make_pair("", t_node));
    g_results_tree.add_child("times", t_nodes);

  } // if

  else if (g_mode == "accuracy") {
    boost::property_tree::ptree final_list_node;

    for (auto i :
         boost::irange(0, boost::numeric_cast<int>(filenames.size()))) {
      vector<uint8_t> imgData = loadImage_int(filenames[i]);

      // TODO: Passing imgData.data() is causing failures. Why?
      memcpy(d, imgData.data(), 3 * 300 * 300 * sizeof(uint8_t));
      armnn::Status ret = runtime->EnqueueWorkload(
          networkIdentifier, MakeInputTensors(inputBindingInfo, d),
          outputTensors);

      boost::property_tree::ptree prediction_list_node;
      for (int i = 0; i < output3[0]; i++) {
        boost::property_tree::ptree prediction_node;
        boost::property_tree::ptree x1, y1, x2, y2;
        boost::property_tree::ptree bbox_list_node;
        // TODO verify order
        x1.put("", output0[i * 4 + 1]);
        y1.put("", output0[i * 4 + 0]);
        x2.put("", output0[i * 4 + 3]);
        y2.put("", output0[i * 4 + 2]);
        bbox_list_node.push_back(make_pair("", x1));
        bbox_list_node.push_back(make_pair("", y1));
        bbox_list_node.push_back(make_pair("", x2));
        bbox_list_node.push_back(make_pair("", y2));
        prediction_node.add_child("box", bbox_list_node);
        prediction_node.put("class", output1[i] + 1);
        prediction_node.put("score", output2[i]);
        prediction_list_node.push_back(make_pair("", prediction_node));
      }

      boost::property_tree::ptree filename_node;
      filename_node.put("", filenames[i]);
      boost::property_tree::ptree file_predictions_node;
      file_predictions_node.push_back(make_pair("", filename_node));
      file_predictions_node.push_back(make_pair("", prediction_list_node));
      final_list_node.push_back(make_pair("", file_predictions_node));

    } // for
  }   // else if accuracy

  boost::property_tree::ptree final_list_node;
  g_results_tree.add_child("predictions", final_list_node);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "Arguments: <input JSON> <output JSON>" << endl;
    return 1;
  }
  string input_json_filename = argv[1];
  string output_json_filename = argv[2];

  // undocumented
  if (argc == 4) {
    g_debug = argv[3][0] == '0' ? 0 : 1;
  }

  try {
    vector<string> filenames;
    boost::property_tree::ptree root;
    boost::property_tree::read_json(input_json_filename, root);
    if (g_debug) {
      boost::property_tree::write_json(cout, root);
    }
    for (const auto &file : root.get_child("images")) {
      filenames.push_back(file.second.data());
    }
    g_model_filename = root.get<string>("model");
    g_iterations = root.get<int>("params.iterations", 1);
    g_batch_size = root.get<int>("params.batch", 1);
    g_hardware = root.get<string>("params.hardware", "cpu");
    g_precision = root.get<string>("params.precision", "fp32");
    g_mode = root.get<string>("params.mode", "latency");
    if (g_mode == "latency") {
      g_batch_size = 1;
    }
    if (g_precision == "int8") {
      runBatchInference_int(filenames);
    } else {
      runBatchInference(filenames);
    }
    boost::property_tree::write_json(output_json_filename, g_results_tree);
    if (g_debug) {
      boost::property_tree::write_json(cout, g_results_tree);
      cout << "Output written to " << output_json_filename << endl;
    }
  } catch (const runtime_error &e) {
    cerr << e.what() << endl;
    return 1;
  }
  return 0;
}
