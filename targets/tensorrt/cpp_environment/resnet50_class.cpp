#include "loadImage.h"
#include "BatchStream.h"
#include "NvInfer.h"
#include "NvUffParser.h"
#include "cudaMappedMemory.h"
#include <chrono> // high precision clock.
#include <cstdint>
#include <deque>
#include <fstream> // file i/o.
#include <iostream>
#include <map>      //for std::map
#include <numeric>  //for accumulate
#include <sstream>  //strings.
#include <stdlib.h> //exit
#include <vector>

#define CUDA(x) cudaCheckError((x), #x, __FILE__, __LINE__)
#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

// INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    if (severity != Severity::kINFO /*|| mEnableDebug*/) {
      // printf("%s\n", msg);
    }
  }
} gLogger;

struct outputLayer {
  std::string name;
  nvinfer1::DimsCHW dims;
  uint32_t size;
  float *CPU;
  float *CUDA;
};

struct TensorRTStreamData {
  nvinfer1::IExecutionContext *context;
  cudaStream_t stream;
  std::vector<void *> host_buffers;
  std::vector<void *> device_buffers;
};

static inline nvinfer1::Dims validateDims(const nvinfer1::Dims &dims) {
  if (dims.nbDims == nvinfer1::Dims::MAX_DIMS)
    return dims;
  nvinfer1::Dims dims_out = dims;
  // TRT doesn't set the higher dims, so make sure they are 1
  for (int n = dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++)
    dims_out.d[n] = 1;
  return dims_out;
}

std::string locateFile(const std::string &input) {
  std::vector<std::string> dirs{"targets/tensorrt/utility/"};
  return locateFile(input, dirs);
}

class Resnet50 {
protected:
  nvinfer1::IRuntime *mInfer;
  nvinfer1::ICudaEngine *mEngine;
  nvinfer1::IExecutionContext *mContext;
  float *mInputCPU;
  float *mInputCUDA;
  std::vector<outputLayer> mOutputs;

public:
  // Constructor
  Resnet50() {
    mEngine = NULL;
    mInfer = NULL;
    mContext = NULL;
    mInputCPU = NULL;
    mInputCUDA = NULL;
  }

  // Destructor
  ~Resnet50() {
    if (mEngine != NULL) {
      mEngine->destroy();
      mEngine = NULL;
    }

    if (mInfer != NULL) {
      mInfer->destroy();
      mInfer = NULL;
    }
  }

  // Create an optimized network from uff model file.
  bool create_engine(const std::string &modelFile, const char *engine_name,
                     int maxBatchSize, char *precision) {
    // create API root class - must span the lifetime of the engine usage
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition *network = builder->createNetwork();
    auto parser = nvuffparser::createUffParser();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string device_name(prop.name);
    std::string optimisation_precision(precision);

    char buff[500];
    GetCurrentDir(buff, 500);
    std::string current_working_dir(buff);
    std::string utility_folder =
        current_working_dir + "/targets/tensorrt/utility/";
    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES, "Resnet50");
    Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH,
                                     utility_folder +
                                         "CalibrationTable_resnet50");

    parser->registerInput(
        "input", nvinfer1::DimsCHW(3, 224, 224),
        nvuffparser::UffInputOrder::kNCHW); // input for frozen_model.
    parser->registerOutput("resnet_v1_50/SpatialSqueeze"); // Name of output

    if (!parser->parse(modelFile.c_str(), *network,
                       nvinfer1::DataType::kFLOAT)) {
      printf("failed to parse uff model\n");
      return false;
    }

    std::map<std::string, nvinfer1::Dims3>
        inputDimensions; // extract the dimensions of the network input blobs.
    for (int i = 0, n = network->getNbInputs(); i < n; i++) {
      nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3 &&>(
          network->getInput(i)->getDimensions());
      inputDimensions.insert(
          std::make_pair(network->getInput(i)->getName(), dims));
      std::cout << "1/4. Retrieved Input tensor \""
                << network->getInput(i)->getName() << "\":  " << dims.d[0]
                << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }
    // build the engine
    builder->setMinFindIterations(2);
    builder->setAverageFindIterations(2);
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(128 << 20); // 128MBs of space

    if (optimisation_precision == "int8") {
      if (device_name == "Xavier") {

        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
      } else {
        printf("Selected device is:%s . This device does not support int8 "
               "optimization. Kindly use correct config file.\n\n",
               device_name.c_str());
        exit(EXIT_FAILURE);
      }
    } else if (optimisation_precision == "fp16") {
      builder->setFp16Mode(true);
    } else if (optimisation_precision == "fp32") {
      builder->setFp16Mode(false);
      // Dont do anything. Fp32 is default mode.
    }
    builder->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);

    printf("initiated engine build for batchsize:%d. Wait few minutes for "
           "%s to build the optimized engine.\n",
           maxBatchSize, device_name.c_str());
    nvinfer1::ICudaEngine *engine = builder->buildCudaEngine(*network);
    printf("2/4. Engine built successfully\n");

    // we don't need the network definition any more, and we can destroy the
    // parser
    network->destroy();

    // serialize the engine, then close everything down
    nvinfer1::IHostMemory *trtModelStream = engine->serialize();
    printf("3/4. Engine serialized successfully\n");
    if (!trtModelStream) {
      printf(" failed to serialize CUDA engine\n");
      return false;
    }
    // Write an engine into a file and resue it next time.
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write((const char *)trtModelStream->data(),
                         trtModelStream->size());

    std::string filename(engine_name);
    std::ofstream outFile;
    outFile.open(filename, std::ofstream::out);
    outFile << gieModelStream.rdbuf();
    outFile.close();
    printf("4/4. Engine file created successfully\n");

    engine->destroy();
    builder->destroy();

    return true;
  }

  // this function reads the optimized engine file and loads it in mEngine
  // variable.
  bool deserialize_load_engine(const char *engine_name) {
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    std::string engineName(engine_name);
    std::ifstream engine_exists(engineName);

    if (!engine_exists) {
      printf("Entered in engine building part\n");
    } else {
      printf("Loading network profile from engine cache... %s\n",
             engineName.c_str());
      gieModelStream << engine_exists.rdbuf();
      engine_exists.close();
    }
    // create runtime inference engine execution context
    mInfer = nvinfer1::createInferRuntime(gLogger);
    if (!mInfer) {
      printf("Failed to create InferRuntime\n");
      return false;
    }

    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);

    void *modelMem = malloc(modelSize);

    if (!modelMem) {
      printf("failed to allocate %i bytes to deserialize model\n", modelSize);
      return false;
    }
    gieModelStream.read((char *)modelMem, modelSize);
    mEngine = mInfer->deserializeCudaEngine(modelMem, modelSize, NULL);

    free(modelMem);

    if (!mEngine) {
      printf(" failed to create CUDA engine\n");
      return false;
    }

  } // Function to deserialize engine ends.

  // aim of this function is to allocate input and output buffers as per the
  // engine. This is without considering streams.
  bool prepare_memory(int maxBatchSize) {
    const int numBindings = mEngine->getNbBindings();
    const int inputIndex = mEngine->getBindingIndex("input");
    nvinfer1::Dims inputDims =
        validateDims(mEngine->getBindingDimensions(inputIndex));
    size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) *
                       DIMS_W(inputDims) * sizeof(float);
    // allocate memory to hold the input buffer
    if (!cudaAllocMapped((void **)&mInputCPU, (void **)&mInputCUDA,
                         inputSize)) {
      printf("failed to alloc CUDA mapped memory for tensor input, %zu bytes\n",
             inputSize);
      return false;
    }
    // setup network output buffers
    const int numOutputs = 1;
    std::string output_blob = "resnet_v1_50/SpatialSqueeze";
    for (int n = 0; n < numOutputs; n++) {
      const int outputIndex = mEngine->getBindingIndex(output_blob.c_str());
      nvinfer1::Dims outputDims =
          validateDims(mEngine->getBindingDimensions(outputIndex));
      size_t outputSize = maxBatchSize * DIMS_C(outputDims) *
                          DIMS_H(outputDims) * DIMS_W(outputDims) *
                          sizeof(float);
      // output memory
      void *outputCPU = NULL;
      void *outputCUDA = NULL;
      if (!cudaAllocMapped((void **)&outputCPU, (void **)&outputCUDA,
                           outputSize)) // returns address of allocated memory.
      {
        printf(
            "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n",
            outputSize);
        return false;
      }
      outputLayer l;
      l.CPU = (float *)outputCPU;
      l.CUDA = (float *)outputCUDA;
      l.size = outputSize;

      DIMS_W(l.dims) = DIMS_W(outputDims);
      DIMS_H(l.dims) = DIMS_H(outputDims);
      DIMS_C(l.dims) = DIMS_C(outputDims);

      l.name = output_blob;
      mOutputs.push_back(l);
    }

    return true;

  } // function ends.

  // inference with default stream.
  float infer(const char *image_name, float *data, int batchsize) {
    float *imgCPU = NULL;
    float *imgCUDA = NULL;
    auto start = std::chrono::system_clock::now();
    if (!loadData(image_name, (float3 **)&imgCPU, (float3 **)&imgCUDA, data,
                  batchsize, 224, 224)) // Load image
    {
      printf("failed to load image \n");
      return 0;
    }
    void *bindBuffers[] = {imgCUDA, mOutputs[0].CUDA};
    nvinfer1::IExecutionContext *context = mEngine->createExecutionContext();
    if (!context) {
      printf("failed to create execution context\n");
      return 0;
    }
    // actual inference.
    if (!context->execute(batchsize, bindBuffers)) {
      printf("imageNet::Process() -- failed to execute TensorRT network\n");
      return 0;
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<float> diff = stop - start;

    context->destroy();
    CUDA(cudaFreeHost(imgCPU));

    return diff.count();
  }

  // This function creates number of streams and makes inference.
  std::deque<float> inference_streams(float *data, int batch_size,
                                      int max_requests_in_fly, int iterations,
                                      bool filewrite) {

    std::string input_name = "input";
    std::string output_name = "resnet_v1_50/SpatialSqueeze";
    int input_byte_size =
        batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    int output_byte_size =
        batch_size * 1000 *
        sizeof(float); // Number of output classes of resnet50- 1000
    int imgPixels = INPUT_H * INPUT_W;
    int volImg = INPUT_C * INPUT_H * INPUT_W;
    std::deque<float> streamTimes;
    // bImage is data structure to hold data for 1 image. Create a vector of
    // bImage. Vector length is equal to batchsize.
    std::vector<bImage<INPUT_C, INPUT_H, INPUT_W>> b_images(batch_size);
    // Empty vector of structures is available. Fill in data from data pointer.
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < volImg; j++) {
        b_images[i].buffer[j] = data[i * volImg + j];
      }
    }
    std::vector<float> buf(input_byte_size,
                           0); // Empty image buffer in NCHW format.
    // actual pixel format conversion.
    for (int i = 0; i < batch_size; ++i) // Batch N
    {
      for (int c = 0; c < INPUT_C; ++c) // Channel C
      {
        for (unsigned j = 0; j < imgPixels; ++j) // H*W
        {
          buf[i * volImg + c * imgPixels + j] =
              b_images[i].buffer[j * INPUT_C + c];
        }
      }
    }

    int input_index = mEngine->getBindingIndex(input_name.c_str());
    int output_index = mEngine->getBindingIndex(output_name.c_str());

    cudaError_t rc = cudaSuccess;
    size_t bindings_number = 2;
    std::vector<TensorRTStreamData> streams_data(max_requests_in_fly);
    int i = 0;

    // this loop is to create/allocate memories in advance.
    for (auto &s : streams_data) // c++11 standard's range based for loop.
    {
      // printf("Stream %d created.\n",++i);
      s.context = nullptr;
      s.stream = nullptr;

      s.host_buffers.resize(bindings_number);
      std::fill(s.host_buffers.begin(), s.host_buffers.end(), nullptr);

      s.device_buffers.resize(bindings_number);
      std::fill(s.device_buffers.begin(), s.device_buffers.end(), nullptr);

      rc = cudaMallocHost(&s.host_buffers[input_index],
                          input_byte_size); // allocated memory on host.
                                            // s.host_buffers[input_index]-
                                            // pointer to host memory.

      if (rc != cudaSuccess)
        throw std::runtime_error("Allocation failed: " +
                                 std::string(cudaGetErrorName(rc)));

      if (!s.context) {
        s.context = mEngine->createExecutionContext();
        if (!s.context)
          throw std::runtime_error("Can't create context!");

        rc = cudaStreamCreate(
            &s.stream); // Creates a new asynchronous stream. s.stream is
                        // pointer to new stream identifier.
        if (rc != cudaSuccess)
          throw std::runtime_error("cudaStreamCreate: " +
                                   std::string(cudaGetErrorName(rc)));
      }

      if (s.device_buffers.size() != mEngine->getNbBindings())
        throw std::runtime_error("Wrong number of bindings: " +
                                 std::to_string(mEngine->getNbBindings()));

      // Allocate inputs memory on device
      if (!s.device_buffers[input_index]) {
        rc = cudaMalloc(&s.device_buffers[input_index], input_byte_size);
        if (rc != cudaSuccess)
          throw std::runtime_error("Allocation failed: " +
                                   std::string(cudaGetErrorName(rc)));
      }

      // Allocate outputs memory on device
      if (!s.device_buffers[output_index]) {
        rc = cudaMalloc(&s.device_buffers[output_index], output_byte_size);
        if (rc != cudaSuccess)
          throw std::runtime_error("Allocation failed: " +
                                   std::string(cudaGetErrorName(rc)));
      }

      // Allocate outputs memory on host
      if (!s.host_buffers[output_index]) {
        rc = cudaMallocHost(&s.host_buffers[output_index], output_byte_size);
        if (rc != cudaSuccess)
          throw std::runtime_error("Allocation failed: " +
                                   std::string(cudaGetErrorName(rc)));
      }

      // Refill data on host
      rc = cudaMemcpyAsync(s.host_buffers[input_index], buf.data(),
                           input_byte_size, cudaMemcpyHostToHost, s.stream);
      if (rc != cudaSuccess)
        throw std::runtime_error("HostToHost: " +
                                 std::string(cudaGetErrorName(rc)));
    }

    int queued_stream_id = -1;
    int synced_stream_id = -1;
    std::vector<std::chrono::high_resolution_clock::time_point> streamStart(
        streams_data.size());

    auto start = std::chrono::high_resolution_clock::now(); // start timekeeping
                                                            // from this point.
    for (int i = 0; i < iterations; i++) {
      queued_stream_id =
          (int)((queued_stream_id + 1) %
                streams_data.size()); // this modulo operator makes sure that
                                      // queued_stream_id iterates through
                                      // number of streams.
                                      
      streamStart[queued_stream_id] = std::chrono::high_resolution_clock::now();                                
      // copy image from host to device.
      rc = cudaMemcpyAsync(
          streams_data.at(queued_stream_id).device_buffers[input_index],
          streams_data.at(queued_stream_id).host_buffers[input_index],
          input_byte_size, cudaMemcpyHostToDevice,
          streams_data.at(queued_stream_id).stream);

      if (rc != cudaSuccess)
        throw std::runtime_error("HostToDevice: " +
                                 std::string(cudaGetErrorName(rc)));

      
      // actual inference
      streams_data.at(queued_stream_id)
          .context->enqueue(
              batch_size,
              streams_data.at(queued_stream_id).device_buffers.data(),
              streams_data.at(queued_stream_id).stream, nullptr);

      // copy back data from device to host.
      rc = cudaMemcpyAsync(
          streams_data.at(queued_stream_id).host_buffers[output_index],
          streams_data.at(queued_stream_id).device_buffers[output_index],
          output_byte_size, cudaMemcpyDeviceToHost,
          streams_data.at(queued_stream_id).stream);
      if (rc != cudaSuccess)
        throw std::runtime_error("DeviceToHost: " +
                                 std::string(cudaGetErrorName(rc)));

      if (((synced_stream_id == queued_stream_id) ||
           ((synced_stream_id == -1) &&
            (((queued_stream_id + 1) % streams_data.size()) == 0)))) {
        synced_stream_id = (int)((synced_stream_id + 1) % streams_data.size());
        rc = cudaStreamSynchronize(streams_data.at(synced_stream_id).stream);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff =
            stop - streamStart[synced_stream_id];
        streamTimes.push_back(diff.count());
        if (rc != cudaSuccess)
          throw std::runtime_error("Can't synchronize stream " +
                                   std::to_string(synced_stream_id) +
                                   std::string(cudaGetErrorName(rc)));
      }
    }

    for (auto &s : streams_data) {
      // printf("entered to destroy context\n");
      s.context->destroy();
    }

    // Wait for all
    while (synced_stream_id != queued_stream_id) {
      synced_stream_id = (int)((synced_stream_id + 1) % streams_data.size());
      rc = cudaStreamSynchronize(streams_data.at(synced_stream_id).stream);
      auto stop = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = stop - streamStart[synced_stream_id];
      streamTimes.push_back(diff.count());
      if (rc != cudaSuccess)
        throw std::runtime_error("Can't synchronize stream " +
                                 std::to_string(synced_stream_id) +
                                 std::string(cudaGetErrorName(rc)));
    }
    // end timer here.
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;

    streamTimes.push_front(diff.count());

    // this loop is to destroy streams.
    for (auto &s : streams_data) {

      rc = cudaStreamDestroy(s.stream);
      // printf("destroyed stream\n");
      if (rc != cudaSuccess)
        throw std::runtime_error("cudaStreamDestroy: " +
                                 std::string(cudaGetErrorName(rc)));
    }

    // if user wants to verify accuracy, write output in files.
    if (filewrite == true) {
      printf("synchronization done. File writing begins\n");
      std::string mode("stream_");
      std::string fileExtension(".txt");
      std::ofstream out1;

      for (int b = 0; b < batch_size; b++) {
        if (batch_size > 1) // Need to split output buffer as per batches only
                            // if batchsize>1.
        {
          std::string fileNum = std::to_string(b);
          out1.open(mode + fileNum + fileExtension, std::ofstream::out);
        } else {
          out1.open(mode + fileExtension, std::ofstream::out);
        }
        for (int i = 0; i < 1000; i++) {
          out1 << ((float *)streams_data.at(0)
                       .host_buffers[output_index])[b * 1000 + i]
               << std::endl; // host_buffers[output_index] is a pointer.
        }
        out1.close();
      }
    }

    return streamTimes;
  } // infer_streams function ends.

  // write results in text file
  void write_results(const char *output_name, int batchsize) {
    std::string outFile(output_name);
    std::string mode("_throughput_");
    std::string fileExtension(".txt");
    std::ofstream out1;

    for (int b = 0; b < batchsize; b++) {
      if (batchsize >
          1) // Need to split output buffer as per batches only if batchsize>1.
      {
        std::string fileNum = std::to_string(b);
        out1.open(outFile + mode + fileNum + fileExtension, std::ofstream::out);
      } else {
        out1.open(outFile + fileExtension, std::ofstream::out);
      }
      for (int i = 0; i < 1000; i++) {
        out1 << (mOutputs[0].CUDA[b * 1000 + i]) << std::endl;
      }
      out1.close();
    }
  }

  // end of class Resnet50.
};

// Call these functions from python.
extern "C" {

Resnet50 *return_object() { return new Resnet50(); }

void create_trt(Resnet50 *obj, const char *uffName, const char *engineFileName,
                int batchsize, char *precision) {
  bool k = false;
  k = obj->create_engine(uffName, engineFileName, batchsize, precision);
  if (k == true) {
    printf("TensorRT engine file written successfully. \n");
  }
}

void prepare_memory_trt(Resnet50 *obj, int maxBatchSize) {
  bool k = false;
  k = obj->prepare_memory(maxBatchSize);
  if (k == false) {
    printf("Error while allocating memory");
  }
}

void deserialize_load_trt(Resnet50 *obj, const char *engine_name) {
  bool k = false;
  k = obj->deserialize_load_engine(engine_name);
  if (k == true) {
    printf("TensorRT engine deserialized and loaded successfully. \n");
  }
}

float infer_trt(Resnet50 *obj, const char *image_name, float *data,
                int batchsize) {
  float inference_time = obj->infer(image_name, data, batchsize);
  return inference_time;
}

void infer_stream_trt(Resnet50 *obj, float *data, int batch_size,
                      int max_requests_in_fly, int runs, float *results,
                      bool filewrite) {
  std::deque<float> inference_times = obj->inference_streams(
      data, batch_size, max_requests_in_fly, runs, filewrite);
  // Assumes results is malloc'd to # of iterations + 1
  for (int i = 0; i < inference_times.size(); ++i) {
    results[i] = inference_times[i];
  }
}

void write_results_trt(Resnet50 *obj, const char *output_name, int batchsize) {
  obj->write_results(output_name, batchsize);
}

void destroy_trt(Resnet50 *obj) { delete obj; }
} // end of extern c.
