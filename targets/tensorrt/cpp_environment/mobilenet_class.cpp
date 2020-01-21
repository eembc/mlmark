#include "loadImage.h"
#include "BatchStream.h"
#include "NvInfer.h"
#include "NvUffParser.h"
#include "cudaMappedMemory.h"
#include <chrono> //accurate clock.
#include <cstdint>
#include <deque>
#include <fstream> //file i/o.
#include <iostream>
#include <map>      //for std::map
#include <numeric>  //for accumulate
#include <sstream>  //strings.
#include <stdlib.h> //exit
#include <vector>
#include "warmup.h"

#define CUDA(x) cudaCheckError((x), #x, __FILE__, __LINE__)
#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

template <typename T>
struct destroyer
{
    void operator()(T* t) { t->destroy(); }
};

template <typename T> using unique_ptr_destroy = std::unique_ptr<T, destroyer<T> >;



// INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

// Required by TensorRT API. Standard practice.
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    if (severity <= Severity::kINFO /*|| mEnableDebug*/) {
      // printf("%s\n", msg);
    }
  }
} gLogger;

struct outputLayer {
  std::string name;
  nvinfer1::Dims3 dims;
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

// copied from sample Trt code. Todo: Understand.
static inline nvinfer1::Dims validateDims(const nvinfer1::Dims &dims) {
  if (dims.nbDims == nvinfer1::Dims::MAX_DIMS)
    return dims;

  nvinfer1::Dims dims_out = dims;
  for (int n = dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++)
    dims_out.d[n] = 1;

  return dims_out;
}

std::string locateFile(const std::string &input) {
  std::vector<std::string> dirs{"targets/tensorrt/utility/"};
  return locateFile(input, dirs);
}

class Mobilenet {
protected:
  nvinfer1::IRuntime *mInfer;
  nvinfer1::ICudaEngine *mEngine;
  nvinfer1::IExecutionContext *mContext;
  float *mInputCPU;
  float *mInputCUDA;
  std::vector<outputLayer> mOutputs;

public:
  // Constructor
  Mobilenet() {
    mEngine = NULL;
    mInfer = NULL;
    mContext = NULL;
    mInputCPU = NULL;
    mInputCUDA = NULL;
  }

  // Destructor
  ~Mobilenet() {
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
                     int maxBatchSize, char *precision, int cudaDevice, int DLACore = -1, bool fallback = true) {
    cudaSetDevice(cudaDevice);
    // create API root class - must span the lifetime of the engine usage
    unique_ptr_destroy<nvinfer1::IBuilder> builder{ nvinfer1::createInferBuilder(gLogger) };
    unique_ptr_destroy<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(0) };
    unique_ptr_destroy<nvuffparser::IUffParser> parser{ nvuffparser::createUffParser() };


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    std::string device_name(prop.name);
    std::string optimisation_precision(precision);

    char buff[500];
    GetCurrentDir(buff, 500);
    std::string current_working_dir(buff);
    std::string utility_folder =
        current_working_dir + "/targets/tensorrt/utility/";
    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES, "Mobilenet");
    Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH,
                                     utility_folder +
                                         "CalibrationTable_mobilenet");

    parser->registerInput(
        "input", nvinfer1::Dims3(3, 224, 224),
        nvuffparser::UffInputOrder::kNCHW); // input for frozen_model.
    parser->registerOutput(
        "MobilenetV1/Predictions/Reshape_1"); // Name of output layer.

    if (!parser->parse(modelFile.c_str(), *network,
                       nvinfer1::DataType::kFLOAT)) {
      printf("failed to parse uff model\n");
      return false;
    }

    // extract the dimensions of the network input blobs.
    std::map<std::string, nvinfer1::Dims3> inputDimensions;

    for (int i = 0, n = network->getNbInputs(); i < n; i++) {
      nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3 &&>(
          network->getInput(i)->getDimensions());
      inputDimensions.insert(
          std::make_pair(network->getInput(i)->getName(), dims));
      std::cout << "1/4. Retrieved Input tensor \""
                << network->getInput(i)->getName() << "\":  " << dims.d[0]
                << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    for (unsigned int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        // Set formats and data types of inputs
        auto input = network->getInput(i);
        input->setType(DataType::kFLOAT);
        input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    }

    for (unsigned int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        // Set formats and data types of outputs
        auto output = network->getOutput(i);
        output->setType(DataType::kFLOAT);
        output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    }

    // build the engine
    unique_ptr_destroy<IBuilderConfig> config{builder->createBuilderConfig()};
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(64 << 20); // 64MBs of space

    if (optimisation_precision == "int8") {
      if (builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(&calibrator);
      } else {
        printf("Selected device is:'%s'. This device does not support int8 "
               "optimization. Kindly use correct config file.\n\n",
               device_name.c_str());
        exit(EXIT_FAILURE);
      }
    } else if (optimisation_precision == "fp16") {
      config->setFlag(BuilderFlag::kFP16);
    } else if (optimisation_precision == "fp32") {
      // Dont do anything. Fp32 is default mode.
    }
    
    if (DLACore == -1)
    {
      config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
    }
    else
    {
        if (DLACore < builder->getNbDLACores())
        {
            config->setDefaultDeviceType(DeviceType::kDLA);
            config->setDLACore(DLACore);
            config->setFlag(BuilderFlag::kSTRICT_TYPES);

            if (fallback)
            {
                config->setFlag(BuilderFlag::kGPU_FALLBACK);
            }
            if (optimisation_precision != "int8")
            {
                config->setFlag(BuilderFlag::kFP16);
            }
        }
        else
        {
            printf("Cannot create DLA engine, %d not available\n", DLACore);
            return false;
        }
    }
    

    printf("initiated engine build for batchsize %d. Wait few minutes for "
           "%s to build the optimized engine.\n",
           maxBatchSize, device_name.c_str());
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    printf("2/4. Engine built successfully\n");

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

    return true;
  }

  // this function reads the optimized engine file and loads it in mEngine
  // variable.
  bool deserialize_load_engine(const char *engine_name, int cudaDevice) {
    cudaSetDevice(cudaDevice);
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    std::string engineName(engine_name);
    std::ifstream engine_exists(engineName);

    if (!engine_exists) {
      printf("Entered in engine building part\n");
    } else {
      printf("Loading network profile from engine ... %s\n",
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
  }

  // aim of this function is to allocate input and output buffers for engine.
  bool prepare_memory(int maxBatchSize) {
    const int numBindings = mEngine->getNbBindings();
    const int inputIndex = mEngine->getBindingIndex("input");
    nvinfer1::Dims inputDims =
        validateDims(mEngine->getBindingDimensions(inputIndex));
    size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) *
                       DIMS_W(inputDims) * sizeof(float);

    if (!cudaAllocMapped((void **)&mInputCPU, (void **)&mInputCUDA,
                         inputSize)) // allocate memory to hold the input buffer
    {
      printf("failed to alloc CUDA mapped memory for tensor input, %zu bytes\n",
             inputSize);
      return false;
    }
    // setup network output buffers
    const int numOutputs = 1;
    std::string output_blob = "MobilenetV1/Predictions/Reshape_1";
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

  // Inference function. Returns time taken in seconds to do image
  // loading+inference.
  float infer(const char *image_name, float *data, int batchsize) {
    float *imgCPU = NULL;
    float *imgCUDA = NULL;
    auto start = std::chrono::system_clock::now();
    if (!loadData(image_name, (float3 **)&imgCPU, (float3 **)&imgCUDA, data,
                  batchsize, 224, 224)) // load image.
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
                                      int max_requests_in_fly, int runs,
                                      bool filewrite) {
    std::string input_name = "input";
    std::string output_name = "MobilenetV1/Predictions/Reshape_1";
    int input_byte_size =
        batch_size * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    int output_byte_size =
        batch_size * 1001 *
        sizeof(float); // Number of output classes of mobilenet - 1001.
    int imgPixels = INPUT_W * INPUT_H;
    int volImg = INPUT_W * INPUT_H * INPUT_C;
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

    // Loop to create/allocate memories in advance.
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

    checkWarmUp(streams_data[0].stream, 5000);
    auto start = std::chrono::high_resolution_clock::now(); // start timekeeping
                                    // from this point.
    for (int i = 0; i < runs; i++) // runs are iterations. used in original code
                                   // for timing measurements.
    {
      queued_stream_id =
          (int)((queued_stream_id + 1) %
                streams_data.size()); // this modulo operator makes sure that
                                      // queued_stream_id  iterates through
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

    // destroy context
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
    auto stop = std::chrono::high_resolution_clock::now(); // end timer here.
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

    if (filewrite ==
        true) // if user wants to verify accuracy, write output in files.
    {
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
        for (int i = 0; i < 1001; i++) {
          out1 << ((float *)streams_data.at(0)
                       .host_buffers[output_index])[b * 1001 + i]
               << std::endl; // host_buffers[output_index] is a pointer.
        }
        out1.close();
      }
    }
    return streamTimes;
  } // infer_streams function ends.

  // write accuracy results in text files.
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
      for (int i = 0; i < 1001; i++) {
        out1 << (mOutputs[0].CUDA[b * 1001 + i]) << std::endl;
      }
      out1.close();
    }
  }
  // end of class Mobilenet.
};

// Call these functions from python.
extern "C" {

Mobilenet *return_object() { return new Mobilenet(); }

void create_trt(Mobilenet *obj, const char *uffName, const char *engineFileName,
                int batchsize, char *precision, int cudaDevice) {
  bool k = false;
  k = obj->create_engine(uffName, engineFileName, batchsize, precision, cudaDevice);
  if (k == true) {
    printf("TensorRT engine file written successfully. \n");
  }
}

void prepare_memory_trt(Mobilenet *obj, int maxBatchSize) {
  bool k = false;
  k = obj->prepare_memory(maxBatchSize);
  if (k == false) {
    printf("Error while input output buffer allocation. \n");
  }
}

void deserialize_load_trt(Mobilenet *obj, const char *engine_name, int cudaDevice) {
  bool k = false;
  k = obj->deserialize_load_engine(engine_name, cudaDevice);
  if (k == true) {
    printf("TensorRT engine deserialized and loaded successfully. \n");
  }
}

float infer_trt(Mobilenet *obj, const char *image_name, float *data,
                int batchsize) {
  float inference_time = obj->infer(image_name, data, batchsize);
  return inference_time;
}

void infer_stream_trt(Mobilenet *obj, float *data, int batch_size,
                      int max_requests_in_fly, int runs, float *results,
                      bool filewrite) {
  std::deque<float> inference_times = obj->inference_streams(
      data, batch_size, max_requests_in_fly, runs, filewrite);
  // Assumes results is malloc'd to # of iterations + 1
  for (int i = 0; i < inference_times.size(); ++i) {
    results[i] = inference_times[i];
  }
}

void write_results_trt(Mobilenet *obj, const char *output_name, int batchsize) {
  obj->write_results(output_name, batchsize);
}

void destroy_trt(Mobilenet *obj) {
  delete obj; // explicit calling of destructor is not required for dynamically
              // created object. delete calls destructor automatically.
}
} // end of extern c.
