//
//  CommonPlugin.hpp
//  MNN
//
//  Created by MNN on b'2020/08/13'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonPlugin_hpp
#define CommonPlugin_hpp
#include <cuda_runtime_api.h>
#include "../schema/current/MNNPlugin_generated.h"
#include "MNN_generated.h"
#include "NvInfer.h"
#include "cuda_fp16.h"
#include <MNN/MNNDefine.h>

namespace MNN {

#define CUASSERT(status_)                                                                                              \
        MNN_ASSERT(status_ == cudaSuccess)     

//only for debug
template <typename Dtype>
struct CpuBind
{
    size_t mSize;
    void* mPtr;

    CpuBind(size_t size, const void* gpuDataPtr)
    {
        mSize = size;
        mPtr = malloc(sizeof(Dtype) * mSize);
        auto status = cudaMemcpy(static_cast<void*>(mPtr), static_cast<const void*>(gpuDataPtr), sizeof(Dtype)*mSize, cudaMemcpyDeviceToHost);
        CUASSERT(status);
    }

    ~CpuBind()
    {
        if (mPtr != nullptr)
        {
            free(mPtr);
            mPtr = nullptr;
        }
    }
    void print(){
        printf("\n");
        for(int i = 0; i < mSize; i++){
            float* a = (float*)(mPtr);
            printf("%f ", a[i]);
        }
        printf("\n");
    }
}; 

template <typename Dtype>
struct CudaBind
{
    size_t mSize;
    void* mPtr;

    CudaBind(size_t size)
    {
        mSize = size;
        auto status = cudaMalloc(&mPtr, sizeof(Dtype) * mSize);
        CUASSERT(status);
    }

    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            auto status = cudaFree(mPtr);
            CUASSERT(status);
            mPtr = nullptr;
        }
    }
};      

class CommonPlugin : public nvinfer1::IPluginExt {
public:
    class Enqueue {
    public:
        Enqueue() {
        }
        virtual ~Enqueue() {
        }
        virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) = 0;
    };
    CommonPlugin(const void* buffer, size_t size);
    CommonPlugin(const Op* op, const MNNTRTPlugin::PluginT* plugin);
    virtual ~CommonPlugin() = default;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
    int initialize() override;
    void terminate() override;
    virtual int getNbOutputs() const override;
    size_t getWorkspaceSize(int) const override {
        return 0;
    }
    size_t getSerializationSize() override;
    void serialize(void* buffer) override;
    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* ptr, cudaStream_t stream) override {
        return mExe->onEnqueue(batchSize, inputs, outputs, ptr, mDataType, stream);
    }

    virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kINT32) && format == nvinfer1::PluginFormat::kNCHW; 
    }

    virtual void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                                     int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format,
                                     int maxBatchSize) override {
        mDataType = type;
    }

private:
    std::vector<int8_t> mOpBuffer;
    std::vector<int8_t> mPluginBuffer;
    std::shared_ptr<Enqueue> mExe;
    nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
};

#define CUDA_NUM_THREADS 512
inline int CAFFE_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

} // namespace MNN

#endif