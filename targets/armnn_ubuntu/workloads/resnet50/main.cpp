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
#include <armnn/INetwork.hpp>
#include <armnnTfParser/ITfParser.hpp>
// stdlib
#include <chrono>
#include <map>
#include <string>
#include <vector>

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

string g_model_filename;
int g_batch_size;
int g_iterations;
string g_hardware;
string g_mode;
string g_precision;
int g_debug(0);
boost::property_tree::ptree g_results_tree;

void LogErrorMessages(const std::vector<std::string> &errorMessages) {
  for (const auto &errorMessage : errorMessages) {
    cout << errorMessage << endl;
  }
}

armnn::InputTensors
MakeInputTensors(const pair<armnn::LayerBindingId, armnn::TensorInfo> &input,
                 const void *inputTensorData) {
  return {{input.first, armnn::ConstTensor(input.second, inputTensorData)}};
}

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
  cv::Rect roi(16, 16, 224, 224);
  cv::Scalar means(123.69, 116.78, 103.94);
  std::vector<float> data;
  double fx, fy;
  img = cv::imread(fn.c_str(), cv::IMREAD_COLOR);
  fx = 256.0 / img.cols;
  fy = 256.0 / img.rows;
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, cv::Size(), fx, fy, cv::INTER_LINEAR);
  img = img(roi);
  img.convertTo(img, CV_32F);
  img = img - means;
  data.assign((float *)img.data, (float *)img.data + img.total() * 3);
  return data;
}

void
runBatchInference(vector<string>& filenames)
{
	// TODO: Placeholder
	const int imgSize(224 * 224 * 3);
	/* TODO support batching in future
	float *imgData = new float[imgSize * filenames.size()];
	size_t x(0);
	for (auto fn : filenames) {
		if (g_debug) {
			cout << "Loading: " << fn << "..." << endl;
		}
		// TODO: Shape...
		vector<uint8_t> bytes = loadImage(fn, 224, 224, 3);
		for (auto i : boost::irange(0, imgSize)) {
			imgData[x] = (float)bytes[i] / 128.0 - 1.0;
			++x;
		}
	}
	*/
	vector<float> imgData;
	float *d = new float[imgSize];

	// Load the model
	armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
	armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(
		g_model_filename.c_str(),
		{{ "input", {1, 224, 224, 3} }},
		{ "resnet_v1_50/SpatialSqueeze" }
	);
	// Bind the tensors
	armnnTfParser::BindingPointInfo inputBindingInfo = 
		parser->GetNetworkInputBindingInfo("input");
	armnnTfParser::BindingPointInfo outputBindingInfo = 
		parser->GetNetworkOutputBindingInfo("resnet_v1_50/SpatialSqueeze");
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
		cout << "Disabling fp16 optimization (using fp32)" << endl;
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
		*network, 
		{ hardware },
		runtime->GetDeviceSpec(),
		oopts,
		armnn::Optional<std::vector<std::string> &>(errorMessages)
		); 
	LogErrorMessages(errorMessages);

  armnn::NetworkId networkIdentifier;
  runtime->LoadNetwork(networkIdentifier, move(optNet));

  // TODO What about batch size?
  vector<float> output(1000);

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
			networkIdentifier,
			MakeInputTensors(inputBindingInfo, d),
			MakeOutputTensors(outputBindingInfo, &output[0])
		);
		tn = chrono::high_resolution_clock::now();
		dt = chrono::duration<double>(tn - t0).count();
		t_node.put("", dt);
		t_nodes.push_back(make_pair("", t_node));
		t_start = chrono::high_resolution_clock::now();
		for (auto i : boost::irange(0, g_iterations)) {
			t0 = chrono::high_resolution_clock::now();
			armnn::Status ret = runtime->EnqueueWorkload(
				networkIdentifier,
				MakeInputTensors(inputBindingInfo, imgData.data()),
				MakeOutputTensors(outputBindingInfo, &output[0])
			);
			tn = chrono::high_resolution_clock::now();
			dt = chrono::duration<double>(tn - t0).count();
			t_node.put("", dt);
			t_nodes.push_back(make_pair("", t_node));
		}
		t_stop = chrono::high_resolution_clock::now();
		dt = chrono::duration<double>(t_stop - t_start).count();
		t_node.put("", dt);
		t_nodes.push_front(make_pair("", t_node));
		g_results_tree.add_child("times", t_nodes);
	} else if (g_mode == "accuracy") {
		boost::property_tree::ptree final_list_node;
		for (auto i : boost::irange(0, boost::numeric_cast<int>(filenames.size()))) {
			vector<float> imgData = loadImage(filenames[i]);
			// TODO: Passing imgData.data() is causing failures. Why?
			memcpy(d, imgData.data(), imgData.size() * sizeof(float));
			armnn::Status ret = runtime->EnqueueWorkload(
				networkIdentifier,
				MakeInputTensors(inputBindingInfo, d),
				MakeOutputTensors(outputBindingInfo, &output[0])
			);
			map<float, int> results = maxSort(output);
			// Save top-5
			boost::property_tree::ptree prediction_node;
			boost::property_tree::ptree prediction_list_node;
			auto it = results.rbegin();
			for (int j=0; j<5 && it != results.rend(); ++j) {
				// Make an anonymous array of anonymous predictions (aka a list)
				prediction_node.put("", it->second);
				prediction_list_node.push_back(make_pair("", prediction_node));
				if (g_debug) {
					cout << filenames[i] << ": Top " << (j+1) << " is " << (it->second)
						<< " with conf " << 100.0*(it->first) << "%" << endl;
				}
				++it;
			}
			// Now make a two-element list [filename, [predictions]]
			boost::property_tree::ptree filename_node;
			filename_node.put("", filenames[i]);
			boost::property_tree::ptree file_predictions_node;
			file_predictions_node.push_back(make_pair("", filename_node));
			file_predictions_node.push_back(make_pair("", prediction_list_node));
			// Now push this two-element list onto the final list
			final_list_node.push_back(make_pair("", file_predictions_node));
		}
		g_results_tree.add_child("predictions", final_list_node);
	}
}

int
main(int argc, char *argv[]) {
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
		for (const auto& file : root.get_child("images")) {
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
		runBatchInference(filenames);
		boost::property_tree::write_json(output_json_filename, g_results_tree);
		if (g_debug) {
			boost::property_tree::write_json(cout, g_results_tree);
			cout << "Output written to " << output_json_filename << endl;
		}
	} catch(const runtime_error& e) {
		cerr << e.what() << endl;
		return 1;
	}
	return 0;
}

