//
//  Int8ToFloatExecution.hpp
//  MNN
//
//  Created by MNN on 2023/01/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#ifndef Int8ToFloatExecution_hpp
#define Int8ToFloatExecution_hpp

#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class Int8ToFloatExecution : public Execution {
public:
    Int8ToFloatExecution(Backend *backend, const std::vector<Tensor *> &inputs, const MNN::Op *param);
    virtual ~Int8ToFloatExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void* mScales;
    int8_t mZeroPoint;
    int mClipBits;
    bool mSingle = false;
    float mSingleScale;
    int mChannel;
    int mCount;
    int mArea;
    MemChunk mScaleStorage;
};

} // namespace CUDA
} // namespace MNN
#endif /* Int8ToFloatExecution_hpp */
#endif