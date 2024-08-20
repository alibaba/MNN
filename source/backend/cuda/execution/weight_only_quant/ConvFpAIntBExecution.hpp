//
//  ConvFpAIntBExecution.hpp
//  MNN
//
//  Created by MNN on 2024/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvFpAIntBExecution_hpp
#define ConvFpAIntBExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "../CutlassGemmParam.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
#include "../cutlass_common/CutlassConvCommonExecution.hpp"

namespace MNN {
namespace CUDA {

class ConvFpAIntBExecution : public CutlassConvCommonExecution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mScale;
        void* mOffset;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> scaleTensor;
        std::shared_ptr<Tensor> offsetTensor;
        std::shared_ptr<Tensor> biasTensor;
        Backend* mBackend = nullptr;
        bool mIsWeightInt4 = false;
    };
    static bool isValid(const Convolution2D* conv, Backend* backend);
    ConvFpAIntBExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvFpAIntBExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;
};

} // namespace CUDA
} // namespace MNN

#endif /* ConvFpAIntBExecution */