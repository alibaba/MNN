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
namespace MNN {
namespace CUDA {
class ConvDepthWiseExecution : public Execution {
public:
    ConvDepthWiseExecution(const Op *op, Backend *bn);
    virtual ~ConvDepthWiseExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    std::pair<void*, int> mConstBuffer;
    const Op *mOp;
    int mTotalCount;

    void* mFilter;
    void* mBias;
    std::shared_ptr<Tensor> weightTensor;
    std::shared_ptr<Tensor> biasTensor;
    bool use_bias_=false;
};

class DeconvDepthWiseExecution : public ConvDepthWiseExecution {
public:
    DeconvDepthWiseExecution(const Op *op, Backend *bn) : ConvDepthWiseExecution(op, bn) {
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