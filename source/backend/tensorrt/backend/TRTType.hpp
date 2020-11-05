//
//  TRTType.hpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTType_H
#define MNN_TRTType_H

#include <MNN/MNNDefine.h>
#include <NvInfer.h>
#include <vector>
namespace MNN {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
        switch (severity) {
            case Severity::kINFO:
                // Ignore kINFO logs because it is too noisy.
                //MNN_PRINT("MNN kINFO : %s \n", msg);
                break;
            case Severity::kWARNING:
                MNN_PRINT("MNN kWARNING : %s \n", msg);
                break;
            case Severity::kINTERNAL_ERROR:
                MNN_PRINT("MNN kINTERNAL_ERROR : %s \n", msg);
            case Severity::kERROR:
                MNN_PRINT("MNN kERROR : %s \n", msg);
                break;
            default:
                break;
        }
    }

    nvinfer1::ILogger& getTRTLogger() {
        return *this;
    }
};

class TRTWeight {
public:
    TRTWeight() = default;
    TRTWeight(nvinfer1::DataType dtype, void* value, size_t num_elem) {
        w_.type   = dtype;
        w_.values = value;
        w_.count  = num_elem;
    }
    nvinfer1::Weights& get() {
        return w_;
    }

    std::vector<int64_t> dims;

private:
    nvinfer1::Weights w_;
};

// HANGXING TODO : not thread-safe.
template <typename T>
struct Singleton {
    static T& Global() {
        static T* x = new T;
        return *x;
    }

    Singleton()        = delete;
    Singleton& operator=(const Singleton&) = delete;
};
} // namespace MNN

#endif // MNN_TRTType_H
