//
//  ConvDepthWiseExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvDepthWiseExecution_hpp
#define ConvDepthWiseExecution_hpp

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "bf16/ConvDepthWiseBf16.cuh"
#include "MultiInputConvDepthWiseExecution.hpp"
namespace MNN {
namespace CUDA {

struct constBuffer {
    int pad[2];
    int kernelSize[2];
    int stride[2];
    int dilate[2];
    int inputSize[2];
    int outputSize[2];
    int channel;
    int total;
    int batch;
    float minValue = -65504.0f;
    float maxValue = 65504.0f;
} uConstant;

class ConvDepthWiseExecution : public Execution {
public:
    struct Resource {
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        void* mFilter;
        void* mBias;
    };
    ConvDepthWiseExecution(const Op *op, Backend *bn, std::shared_ptr<Resource> resource);
    virtual ~ConvDepthWiseExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    MemChunk mConstBuffer;
    const Op *mOp;
    int mTotalCount;
    constBuffer parameters;
    std::shared_ptr<Resource> mResource;
};

class DeconvDepthWiseExecution : public ConvDepthWiseExecution {
public:
    DeconvDepthWiseExecution(const Op *op, Backend *bn, std::shared_ptr<Resource> resource) : ConvDepthWiseExecution(op, bn, resource) {
        // Do nothing
    }
    virtual ~DeconvDepthWiseExecution() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace CUDA
} // namespace MNN
#endif