#include "cudaMappedMemory.h"
#include <algorithm> //for std::max_element
#include <assert.h>
#include <chrono> // high precision clock.
#include <cstdint>
#include <cublas_v2.h> //for cublasHandle_t
#include <deque>
#include <fstream> //for file i/o.
#include <iostream>
#include <map>      //for std::map
#include <numeric>  //for accumulate
#include <sstream>  //since you have strings.
#include <stdlib.h> //exit
#include <vector>   //for loacte

#include "BatchStream_ssdmobilenet.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
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



#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cout << "Cuda failure: " << ret;                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// These are necessary for templated structure. Declared in
// Batchstream_ssdmobilenet.h
/*static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 300;
static constexpr int INPUT_W = 300;
*/

static constexpr int OUTPUT_CLS_SIZE = 91;

const char *OUTPUT_BLOB_NAME0 = "NMS";

nvinfer1::plugin::DetectionOutputParameters detectionOutputParam{
    true,
    false,
    0,
    OUTPUT_CLS_SIZE,
    100,
    100,
    0.3,
    0.6,
    nvinfer1::plugin::CodeTypeSSD::TF_CENTER,
    {0, 2, 1},
    true,
    true};

// INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

// Just a structure to hold data. Much useful for batchsizes>1.
template <int C, int H, int W> struct bImage { float buffer[C * H * W]; };

inline void *safeCudaMalloc(size_t memSize) {
  void *deviceMem;
  CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

std::vector<std::pair<int64_t, nvinfer1::DataType>>
calculateBindingBufferSizes(const nvinfer1::ICudaEngine &engine, int nbBindings,
                            int batchSize) {
  std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
  for (int i = 0; i < nbBindings; ++i) {
    nvinfer1::Dims dims = engine.getBindingDimensions(i);
    nvinfer1::DataType dtype = engine.getBindingDataType(i);

    int64_t eltCount = volume(dims) * batchSize;
    sizes.push_back(std::make_pair(eltCount, dtype));
  }

  return sizes;
}

class FlattenConcat : public nvinfer1::IPluginV2 {
public:
  FlattenConcat(int concatAxis, bool ignoreBatch)
      : mIgnoreBatch(ignoreBatch), mConcatAxisID(concatAxis) {
    assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
  }
  // clone constructor
  FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs,
                int outputConcatAxis, int *inputConcatAxis)
      : mIgnoreBatch(ignoreBatch), mConcatAxisID(concatAxis),
        mOutputConcatAxis(outputConcatAxis), mNumInputs(numInputs) {
    CHECK(cudaMallocHost((void **)&mInputConcatAxis, mNumInputs * sizeof(int)));
    for (int i = 0; i < mNumInputs; ++i)
      mInputConcatAxis[i] = inputConcatAxis[i];
  }

  FlattenConcat(const void *data, size_t length) {
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);
    CHECK(cudaMallocHost((void **)&mInputConcatAxis, mNumInputs * sizeof(int)));
    CHECK(cudaMallocHost((void **)&mCopySize, mNumInputs * sizeof(size_t)));

    std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs,
                  [&](int &inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::DimsCHW>(d);

    std::for_each(mCopySize, mCopySize + mNumInputs,
                  [&](size_t &inp) { inp = read<size_t>(d); });

    assert(d == a + length);
  }
  ~FlattenConcat() {
    if (mInputConcatAxis)
      CHECK(cudaFreeHost(mInputConcatAxis));
    if (mCopySize)
      CHECK(cudaFreeHost(mCopySize));
  }
  int getNbOutputs() const override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override {
    assert(nbInputDims >= 1);
    assert(index == 0);
    mNumInputs = nbInputDims;
    CHECK(cudaMallocHost((void **)&mInputConcatAxis, mNumInputs * sizeof(int)));
    mOutputConcatAxis = 0;
#ifdef SSD_INT8_DEBUG
    std::cout << " Concat nbInputs " << nbInputDims << "\n";
    std::cout << " Concat axis " << mConcatAxisID << "\n";
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 3; ++j)
        std::cout << " Concat InputDims[" << i << "]"
                  << "d[" << j << " is " << inputs[i].d[j] << "\n";
#endif
    for (int i = 0; i < nbInputDims; ++i) {
      int flattenInput = 0;
      assert(inputs[i].nbDims == 3);
      if (mConcatAxisID != 1)
        assert(inputs[i].d[0] == inputs[0].d[0]);
      if (mConcatAxisID != 2)
        assert(inputs[i].d[1] == inputs[0].d[1]);
      if (mConcatAxisID != 3)
        assert(inputs[i].d[2] == inputs[0].d[2]);
      flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
      mInputConcatAxis[i] = flattenInput;
      mOutputConcatAxis += mInputConcatAxis[i];
    }

    return nvinfer1::Dims3(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                             mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                             mConcatAxisID == 3 ? mOutputConcatAxis : 1);
  }

  int initialize() override {
    CHECK(cublasCreate(&mCublas));
    return 0;
  }

  void terminate() override { CHECK(cublasDestroy(mCublas)); }

  size_t getWorkspaceSize(int) const override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void **outputs, void *,
              cudaStream_t stream) override {
    int numConcats = 1;
    assert(mConcatAxisID != 0);
    numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1,
                                 std::multiplies<int>());

    if (!mIgnoreBatch)
      numConcats *= batchSize;

    float *output = reinterpret_cast<float *>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i) {
      const float *input = reinterpret_cast<const float *>(inputs[i]);
      float *inputTemp;
      CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

      CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize,
                            cudaMemcpyDeviceToDevice, stream));

      for (int n = 0; n < numConcats; ++n) {
        CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                          inputTemp + n * mInputConcatAxis[i], 1,
                          output + (n * mOutputConcatAxis + offset), 1));
      }
      CHECK(cudaFree(inputTemp));
      offset += mInputConcatAxis[i];
    }

    return 0;
  }

  size_t getSerializationSize() const override {
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) +
           sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
  }

  void serialize(void *buffer) const override {
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i) {
      write(d, mInputConcatAxis[i]);
    }
    write(d, mCHW);
    for (int i = 0; i < mNumInputs; ++i) {
      write(d, mCopySize[i]);
    }
    assert(d == a + getSerializationSize());
  }

  void configureWithFormat(const nvinfer1::Dims *inputs, int nbInputs,
                           const nvinfer1::Dims *outputDims, int nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) override {
    assert(nbOutputs == 1);
    mCHW = inputs[0];
    assert(inputs[0].nbDims == 3);
    CHECK(cudaMallocHost((void **)&mCopySize, nbInputs * sizeof(size_t)));
    for (int i = 0; i < nbInputs; ++i) {
      mCopySize[i] =
          inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
    }
  }

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == nvinfer1::PluginFormat::kNCHW);
  }
  const char *getPluginType() const override { return "FlattenConcat_TRT"; }

  const char *getPluginVersion() const override { return "1"; }

  void destroy() override { delete this; }

  nvinfer1::IPluginV2 *clone() const override {
    return new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs,
                             mOutputConcatAxis, mInputConcatAxis);
  }

  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

private:
  template <typename T> void write(char *&buffer, const T &val) const {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
  }

  template <typename T> T read(const char *&buffer) {
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
  }

  size_t *mCopySize = nullptr;
  bool mIgnoreBatch{false};
  int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
  int *mInputConcatAxis = nullptr;
  nvinfer1::Dims mCHW;
  cublasHandle_t mCublas;
  std::string mNamespace;
};

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    if (severity <= Severity::kINFO /*|| mEnableDebug*/) {
       // printf("%s\n", msg);
       // printf("%d\n", (int)severity);
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

class SsdMobilenet {
protected:
  nvinfer1::IRuntime *mInfer;
  nvinfer1::ICudaEngine *mEngine;
  nvinfer1::IExecutionContext *mContext;

  float *mInputCPU;
  float *mInputCUDA;

  std::vector<outputLayer> mOutputs;

public:
  // Constructor
  SsdMobilenet() {
    mEngine = NULL;
    mInfer = NULL;
    mContext = NULL;

    mInputCPU = NULL;
    mInputCUDA = NULL;
    initLibNvInferPlugins(&gLogger, "");
  }

  // Destructor
  ~SsdMobilenet() {
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
    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
    Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH,
                                     utility_folder +
                                         "CalibrationTable_ssdmobilenet");

    parser->registerInput(
        "Input", nvinfer1::Dims3(3, 300, 300),
        nvuffparser::UffInputOrder::kNCHW); // input for frozen_model.
    parser->registerOutput("MarkOutput_0");

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
    config->setMaxWorkspaceSize(1UL << 32);

    if (optimisation_precision == "int8") {
      if (builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(&calibrator);
      } else {
        printf("Selected device is:%s . This device does not support int8 "
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
    

    printf("initiated engine build for batchsize:%d. Wait few minutes for "
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
    std::string mCacheEnginePath(engine_name);
    std::ifstream cache(mCacheEnginePath);

    if (!cache) {
      printf("Entered in engine building part\n");
    } else {
      printf("Loading network profile from engine cache... %s\n",
             mCacheEnginePath.c_str());
      gieModelStream << cache.rdbuf();
      cache.close();
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

  float doInference(const char *image_name, float *data, int batchsize,
                    float *detectionOut, int *keepCount) {

    // Input and output buffer pointers that we pass to the engine - the engine
    // requires exactly IEngine::getNbBindings(), of these, but in this case we
    // know that there is exactly 1 input and 2 output.
    int nbBindings = mEngine->getNbBindings();
    const bool isOptimizedNMS = nbBindings == 2;

    std::vector<void *> buffers(nbBindings);
    std::vector<std::pair<int64_t, nvinfer1::DataType>> buffersSizes;

    // calculate Binding Buffer Sizes
    for (int i = 0; i < nbBindings; ++i) {
      nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
      nvinfer1::DataType dtype = mEngine->getBindingDataType(i);

      int64_t eltCount = volume(dims) * batchsize;
      buffersSizes.push_back(std::make_pair(eltCount, dtype));
    }

    // image data NHWC to NCHW converter
    int input_byte_size =
        batchsize * INPUT_C * INPUT_W * INPUT_H * sizeof(float);
    int imgPixels = INPUT_W * INPUT_H;
    int volImg = INPUT_W * INPUT_H * INPUT_C;

    // bImage is data structure to hold data for 1 image. Now create vector of
    // bImage.length equal to batchsize.
    std::vector<bImage<INPUT_C, INPUT_H, INPUT_W>> b_images(batchsize);

    // Fill in data from data pointer.
    for (int i = 0; i < batchsize; i++) {
      for (int j = 0; j < volImg; j++) {
        b_images[i].buffer[j] = data[i * volImg + j];
      }
    }

    std::vector<float> buf(input_byte_size, 0); // input image in NCHW format.

    // actual pixel format conversion.
    for (int i = 0; i < batchsize; ++i) // Batch N
    {
      for (int c = 0; c < INPUT_C; ++c) // Channel C
      {
        for (unsigned j = 0; j < imgPixels; ++j) // H*W
        {
          buf[i * volImg + c * imgPixels + j] =
              (2.0 / 255.0) * b_images[i].buffer[j * INPUT_C + c] - 1.0;
        }
      }
    }

    for (int i = 0; i < nbBindings; ++i) {
      auto bufferSizesOutput = buffersSizes[i];
      buffers[i] = safeCudaMalloc(
          bufferSizesOutput.first *
          getElementSize(bufferSizesOutput.second)); // sizeof float
    }

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings().
    int inputIndex = mEngine->getBindingIndex("Input"),
        outputIndex0 = mEngine->getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 =
            outputIndex0 + 1; // engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    const int output0_volume = 
        volume(mEngine->getBindingDimensions(outputIndex0));
    const int output0_bytes_size = batchsize * output0_volume * sizeof(float);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    auto t_start = std::chrono::high_resolution_clock::now();
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it
    // back:
    CHECK(
        cudaMemcpyAsync(buffers[inputIndex], &buf[0],
                        batchsize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));

    nvinfer1::IExecutionContext *context = mEngine->createExecutionContext();
    if (!context) {
      printf("failed to create execution context\n");
    }
    context->execute(batchsize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(t_end - t_start).count();

    if (!isOptimizedNMS) {
      CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0],
                          output0_bytes_size,
                          cudaMemcpyDeviceToHost, stream));
      CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1],
                          batchsize * sizeof(int), cudaMemcpyDeviceToHost,
                          stream));
    } else {
      std::vector<float> output(batchsize * output0_volume);  
      CHECK(cudaMemcpyAsync(&output[0], buffers[outputIndex0],
                          output0_bytes_size,
                          cudaMemcpyDeviceToHost, stream));
      float* floatPtr = (float*)&output[0];
      int* intPtr = (int*)&output[0] + detectionOutputParam.keepTopK*7;
      for (int b = 0; b < batchsize; ++b){
        for (int i =0; i < detectionOutputParam.keepTopK*7; i += 7) {
          // detectionOut: [image_id, label, confidence, xmin, ymin, xmax, ymax]
          // floatPtr:     [IMAGE_ID, YMIN, XMIN, YMAX, XMAX, CONFIDENCE, LABEL]
          detectionOut[i + 0] = floatPtr[i + 0];
          detectionOut[i + 1] = floatPtr[i + 6];
          detectionOut[i + 2] = floatPtr[i + 5];
          detectionOut[i + 3] = floatPtr[i + 2];
          detectionOut[i + 4] = floatPtr[i + 1];
          detectionOut[i + 5] = floatPtr[i + 4];
          detectionOut[i + 6] = floatPtr[i + 3];
        }
        keepCount[b] = intPtr[0];

        floatPtr += output0_volume;
        intPtr += output0_volume;
        detectionOut += detectionOutputParam.keepTopK*7;
      }
    }
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    if (!isOptimizedNMS) { 
      CHECK(cudaFree(buffers[outputIndex1])); 
    }

    context->destroy();

    return total_time;
  }

  // This function creates number of streams and makes inference.
  std::deque<float> inference_streams(float *data, int batch_size,
                                      int max_requests_in_fly, int iterations,
                                      bool filewrite) {

    // Input and output buffer pointers that we pass to the engine - the engine
    // requires exactly IEngine::getNbBindings(), of these, but in this case we
    // know that there is exactly 1 input and 2 output.
    int nbBindings = mEngine->getNbBindings();
    const bool isOptimizedNMS = nbBindings == 2;

    
    // image data NHWC to NCHW converter
    int input_byte_size =
        batch_size * INPUT_C * INPUT_W * INPUT_H * sizeof(float);
    int imgPixels = INPUT_W * INPUT_H;
    int volImg = INPUT_W * INPUT_H * INPUT_C;
    std::deque<float> streamTimes;

    // bImage is data structure to hold data for 1 image. Now create vector of
    // that data structure whoes length is equal to batchsize.
    std::vector<bImage<INPUT_C, INPUT_H, INPUT_W>> b_images(batch_size);

    // Empty vector of structures is available. Fill in data from data pointer.
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < volImg; j++) {
        b_images[i].buffer[j] = data[i * volImg + j];
      }
    }

    std::vector<float> buf(input_byte_size, 0); // input image in NCHW format.
    // actual pixel format conversion.
    for (int i = 0; i < batch_size; ++i) // Batch N
    {
      for (int c = 0; c < INPUT_C; ++c) // Channel C
      {
        for (unsigned j = 0; j < imgPixels; ++j) // H*W
        {
          buf[i * volImg + c * imgPixels + j] =
              (2.0 / 255.0) * b_images[i].buffer[j * INPUT_C + c] - 1.0;
        }
      }
    }

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings().
    int inputIndex = mEngine->getBindingIndex("Input"),
        outputIndex0 = mEngine->getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 =
            outputIndex0 + 1; // engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    const int output0_volume = volume(mEngine->getBindingDimensions(outputIndex0));
    const int output0_bytes_size = batch_size * output0_volume * sizeof(float);

    cudaError_t rc = cudaSuccess;
    size_t bindings_number = nbBindings;
    std::vector<TensorRTStreamData> streams_data(max_requests_in_fly);

    // this loop is to create/allocate memories in advance.
    for (auto &s : streams_data) // c++11 standard's range based for loop.
    {
      // printf("Stream %d created.\n",++i);
      // buffer is a vector of type(void*).
      s.host_buffers.resize(bindings_number);
      s.device_buffers.resize(bindings_number);

      for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        size_t bufferSize = batch_size * volume(dims) * getElementSize(dtype);
        
        // Allocate memory on host.
        rc = cudaMallocHost(&s.host_buffers[i], bufferSize); 
        if (rc != cudaSuccess)
          throw std::runtime_error("Allocation failed (Host): " +
                                  std::string(cudaGetErrorName(rc)));
        
        // Allocate memory on device
        rc = cudaMalloc(&s.device_buffers[i], bufferSize);
        if (rc != cudaSuccess)
          throw std::runtime_error("Allocation failed: (Device) " +
                                  std::string(cudaGetErrorName(rc)));
      }

      s.context = mEngine->createExecutionContext();
      if (!s.context)
        throw std::runtime_error("Can't create context!");

      rc = cudaStreamCreate(
          &s.stream); // Creates a new asynchronous stream. s.stream is
                      // pointer to new stream identifier.
      if (rc != cudaSuccess)
        throw std::runtime_error("cudaStreamCreate: " +
                                  std::string(cudaGetErrorName(rc)));
      // Refill data on host
      rc = cudaMemcpyAsync(s.host_buffers[inputIndex], buf.data(),
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
    // start timekeeping from this point.
    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; it++) {
      queued_stream_id =
          (int)((queued_stream_id + 1) %
                streams_data.size()); // this modulo operator makes sure that
                                      // queued_stream_id iterates through
                                      // number of streams.
                                      
      streamStart[queued_stream_id] = std::chrono::high_resolution_clock::now();                                

      rc = cudaMemcpyAsync(
          streams_data.at(queued_stream_id).device_buffers[inputIndex],
          streams_data.at(queued_stream_id).host_buffers[inputIndex],
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

      // copy back data from device to host. index0
      rc = cudaMemcpyAsync(
          streams_data.at(queued_stream_id).host_buffers[outputIndex0],
          streams_data.at(queued_stream_id).device_buffers[outputIndex0],
          output0_bytes_size,
          cudaMemcpyDeviceToHost, streams_data.at(queued_stream_id).stream);

      // copy back data from device to host. index1
      if (!isOptimizedNMS) {
        rc = cudaMemcpyAsync(
            streams_data.at(queued_stream_id).host_buffers[outputIndex1],
            streams_data.at(queued_stream_id).device_buffers[outputIndex1],
            batch_size * sizeof(int), cudaMemcpyDeviceToHost,
            streams_data.at(queued_stream_id).stream);
      }
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
    // destroy the context
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

      // ToDo: copy output data somewhere and make use of write file function.

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
        for (int i = 0; i < 100; i++) {
          float *det =
              ((float *)streams_data.at(0).host_buffers[outputIndex0]) +
              b * output0_volume + i * 7;
          out1 << det[1] << ";" << det[2] << ";" << det[3] << ";" << det[4]
               << ";" << det[5] << ";" << det[6] << std::endl;
        }
        out1.close();
      }
    }

    return streamTimes;
  } // infer_streams function ends.

  // end of class SsdMobilenet.
};

// Wrappers.Waiting to get called by python.
extern "C" {

SsdMobilenet *return_object() { return new SsdMobilenet(); }

void create_trt(SsdMobilenet *obj, const char *uffName,
                const char *engineFileName, int batchsize, char *precision, int cudaDevice) {
  bool k = false;
  k = obj->create_engine(uffName, engineFileName, batchsize, precision, cudaDevice);
  if (k == true) {
    printf("TensorRT engine file written successfully. \n");
  }
}

void deserialize_load_trt(SsdMobilenet *obj, const char *engine_name, int cudaDevice) {
  bool k = false;
  k = obj->deserialize_load_engine(engine_name, cudaDevice);
  if (k == true) {
    printf("TensorRT engine deserialized and loaded successfully. \n");
  }
}

float doInference_trt(SsdMobilenet *obj, const char *image_name, float *data,
                      const char *output_name, int batchsize, bool filewrite) {
  // Host memory for outputs.
  std::vector<float> detectionOut(batchsize * detectionOutputParam.keepTopK *
                                  7);
  std::vector<int> keepCount(batchsize);
  float inference_time = obj->doInference(image_name, data, batchsize,
                                          &detectionOut[0], &keepCount[0]);

  if (filewrite == true) {
    // write into a file
    std::string outFile(output_name);
    std::string fileExtension(".txt");
    std::ofstream out1;
    // std::cout<<outFile+fileExtension<<std::endl;   //print output filename
    out1.open(outFile + fileExtension, std::ofstream::out);

    for (int p = 0; p < batchsize; ++p) {
      for (int i = 0; i < keepCount[p]; ++i) {
        float *det =
            &detectionOut[0] + (p * detectionOutputParam.keepTopK + i) * 7;
        // Output format for each detection is stored in the below order
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]

        if (det[2] > 0.3)
        {
          out1 << det[1] << ";" << det[2] << ";" << det[3] << ";" << det[4]
               << ";" << det[5] << ";" << det[6] << std::endl;
        }
      }
    }
    out1.close();
  }
  return inference_time;
}

void infer_stream_trt(SsdMobilenet *obj, float *data, int batch_size,
                      int max_requests_in_fly, int iterations, float *results,
                      bool filewrite) {
  std::deque<float> inference_times = obj->inference_streams(
      data, batch_size, max_requests_in_fly, iterations, filewrite);
  // Assumes results is malloc'd to # of iterations + 1
  for (int i = 0; i < inference_times.size(); ++i) {
    results[i] = inference_times[i];
  }
}

void destroy_trt(SsdMobilenet *obj) { delete obj; }

} // end of extern c.
