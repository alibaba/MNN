//
//  DepthwiseConvInt8Execution.hpp
//  MNN
//
//  Created by MNN on 2023/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#ifndef DepthwiseConvInt8Execution_hpp
#define DepthwiseConvInt8Execution_hpp

#include "ConvInt8CutlassExecution.hpp"
namespace MNN {
namespace CUDA {

class DepthwiseConvInt8Execution : public ConvInt8CutlassExecution {
public:
    DepthwiseConvInt8Execution(Backend *bn, const Op *op, std::shared_ptr<Resource> resource);
    virtual ~DepthwiseConvInt8Execution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    const Op* mOp = nullptr;
    std::shared_ptr<ConvInt8CutlassExecution::Resource> mResource;
    std::pair<int, int> mPads;
    std::pair<int, int> mStrides;
    std::pair<int, int> mDilates;
    std::pair<int, int> mKernels;
    std::pair<int8_t, int8_t> mClamps;
};

} // namespace CUDA
} // namespace MNN

#endif /* DepthwiseConvInt8Execution_hpp */
#endif