#ifndef BATCH_STREAM_PPM_H
#define BATCH_STREAM_PPM_H
#include <vector>
#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <experimental/filesystem>
#include "NvInfer.h"
#include "common.h"
#include <unistd.h>
#define GetCurrentDir getcwd

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::string locateFile(const std::string& input);

const char* INPUT_BLOB_NAME = "input";

//Just a structure to hold data. Much useful for batchsizes>1.
template<int C, int H, int W>
struct CharImage
{
   float buffer[C * H * W];
};

//Function to load an image from file using stb library.
std::vector<uint8_t> loadImage(const char *fn, int width, int height, int channels) 
{
    std::vector<uint8_t> m_Data;
    
    using StbImageDataPtr = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>; //it is a smart pointer.
    StbImageDataPtr stbData(stbi_load(fn, &width, &height, &channels, 3), &stbi_image_free);
    int sizeInBytes = width * height * channels;
    m_Data.resize(sizeInBytes);
    memcpy(m_Data.data(), stbData.get(), sizeInBytes);
    return m_Data;
}

/*
inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}*/

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, std::string model)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
	, mModel(model)
    {
        mDims = nvinfer1::Dims4{batchSize, INPUT_C, INPUT_H, INPUT_W};
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
                return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        }
        mBatchCount++;
        return true;
    }

    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return mBatch.data(); }
    float* getLabels() { return mLabels.data(); }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    nvinfer1::Dims4 getDims() const { return mDims; }

private:
    float* getFileBatch() { return mFileBatch.data(); }
    float* getFileLabels() { return mFileLabels.data(); }

    bool update()
    {
        std::vector<std::string> fNames; //file names.

        std::ifstream file(locateFile("ILSVRC2012_list.txt"), std::ifstream::in);
        printf("located list\n");
        if (file)
        {
            std::cout << "Batch #" << mFileCount << "\n";
            file.seekg(((mBatchCount * mBatchSize)) * 29);  //29 characters are there in every filename.
        }
        
        //std::cout << "Current path is " << fs::current_path() << '\n';
        char buff[500];
        GetCurrentDir( buff, 500 );
        std::string current_working_dir(buff);
        std::string ilsvrc_folder = current_working_dir + "/datasets/ILSVRC2012/images/" ;

        for (int i = 1; i <= mBatchSize; i++)
        {
            std::string sName;
            std::getline(file, sName);
            std::string full_name= ilsvrc_folder + sName;
            std::cout << "Calibrating with file " << full_name << std::endl;
            
            fNames.emplace_back(full_name); //push_back
        } //get filenames equal to batchsize.
        mFileCount++;
        
        //charImage is data structure to hold data for 1 image. Now create vector of that data structure whoes length is equal to batchsize.
	std::vector<CharImage<INPUT_C, INPUT_H, INPUT_W>> b_images(mBatchSize);
        for (uint32_t i = 0; i < fNames.size(); ++i) //read each image and store 
        {
            std::vector<uint8_t> img=loadImage(fNames[i].c_str(), 224, 224, 3); 
            for(int j=0;j<(3*224*224)-2;j++)  //Mean subtraction
              {  
                 b_images[i].buffer[j]  =float(img[j])-123.68; 
		 b_images[i].buffer[j+1]=float(img[j+1]) -116.78;
		 b_images[i].buffer[j+2]=float(img[j+2]) -103.94;
              }
        }
        std::vector<float> data(volume(mDims));

        long int volChl = mDims.d[2] * mDims.d[3];
	if(mModel=="Resnet50")
	{
         for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i)
         {
            for (int c = 0; c < mDims.d[1]; ++c)
            {
                for (int j = 0; j < volChl; ++j)
                {
                    data[i * volImg + c * volChl + j] = float(b_images[i].buffer[j * mDims.d[1] + c]); //preprocessing is necessary.
                }//loop1
            }//loop2
          }//loop3
	}//if condition

	if(mModel=="Mobilenet")
	{
         for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i)
         {
            for (int c = 0; c < mDims.d[1]; ++c)
            {
                for (int j = 0; j < volChl; ++j)
                {
                    data[i * volImg + c * volChl + j] = float(b_images[i].buffer[j * mDims.d[1] + c])/128.0 ; //preprocessing is necessary.
                }//loop1
            }//loop2
          }//loop3
	}//if condition
	

        std::copy_n(data.data(), mDims.d[0] * mImageSize, getFileBatch());

        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    std::string mModel;

    int mFileCount{0}, mFileBatchPos{0};
    int mImageSize{0};

    nvinfer1::Dims4 mDims;
    std::vector<float> mBatch;
    std::vector<float> mLabels;
    std::vector<float> mFileBatch;
    std::vector<float> mFileLabels;
};

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator(BatchStream& stream, int firstBatch, std::string calibrationTableName, bool readCache = true)
        : mStream(stream)
        , mCalibrationTableName(std::move(calibrationTableName))
        , mReadCache(readCache)
    {
        nvinfer1::Dims4 dims = mStream.getDims();
        mInputCount = volume(dims);
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], INPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    BatchStream mStream;
    std::string mCalibrationTableName;
    bool mReadCache{true};

    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};
#endif
