#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace nvinfer1;
using namespace plugin;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

constexpr long double operator"" _GB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KB(long double val) { return val * (1 << 10); }

/*
// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};
*/

// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
                break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
                                                    [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << directoryList << std::endl;
        exit(EXIT_FAILURE);
    }
    return filepath;
}

inline int parseDLA(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (strncmp(argv[i], "--useDLACore=", 13) == 0)
            return stoi(argv[i] + 13);
    }
    return -1;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}



#endif // TENSORRT_COMMON_H
