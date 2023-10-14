//
//  FloatToInt8Execution.hpp
//  MNN
//
//  Created by MNN on 2023/01/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#ifndef FloatToInt8Execution_hpp
#define FloatToInt8Execution_hpp

#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "../CastExecution.hpp"

namespace MNN {
namespace CUDA {

class FloatToInt8Execution : public Execution {
public:
    FloatToInt8Execution(Backend *backend, const std::vector<Tensor *> &inputs, const MNN::Op *param);
    virtual ~FloatToInt8Execution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void* mScales;
    int8_t mZeroPoint;
    int8_t mClampMin;
    int8_t mClampMax;
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
#endif /* FloatToInt8Execution_hpp */
#endif