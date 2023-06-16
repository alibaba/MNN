//
//  BinaryInt8Execution.hpp
//  MNN
//
//  Created by MNN on 2023/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef ENABLE_CUDA_QUANT

#ifndef BinaryInt8Execution_hpp
#define BinaryInt8Execution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {
class BinaryInt8Execution : public Execution {
public:
    BinaryInt8Execution(const MNN::Op* op, Backend *backend, int activationType = 0);
    virtual ~BinaryInt8Execution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mInput0ScalesTensor;
    std::shared_ptr<Tensor> mInput1ScalesTensor;
    std::shared_ptr<Tensor> mOutputScalesTensor;
    int mType;
    int mActivationType;
    bool mIsEltwiseInt8;

};
} // namespace CUDA
} // namespace MNN

#endif
#endif