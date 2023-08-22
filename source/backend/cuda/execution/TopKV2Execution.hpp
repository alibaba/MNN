//
//  TopKV2Execution.hpp
//  MNN
//
//  Created by MNN on 2023/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#ifndef TopKV2Execution_hpp
#define TopKV2Execution_hpp

#include "core/Execution.hpp"
#include "core/Macro.h"
#include "backend/cuda/core/CUDABackend.hpp"
#include <memory>
#include <limits>
#include "cuda_fp16.h"

namespace MNN {
namespace CUDA {


class TopKV2Execution : public Execution {
public:
    TopKV2Execution(const Op * op, Backend * backend);
    virtual ~TopKV2Execution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    struct TopKV2Params {
        int mLengthRow;
        int mNumRow;
        int mDescendFlag = 1;
        void * mBufferIndices;
        void * mBufferValues;

        int mNumK;
        int mNumElePerRow;
        int mNumElePerThread;
        int mNumThreadPerBlock;
        int mNumElePerBlock;
        int mNumBlockPerRow;
        int mNumBlockTotal;
        int mNumBlockFinal;
        int mNumThreadFinal;

        float mMinFloat = std::numeric_limits<float>::lowest();
        half mMinHalf = __float2half(-65504.0f);
        int mMinInt = -std::numeric_limits<int>::max();
    };

    const Op * mOp;
    TopKV2Params mParams;
};


}
}

#endif





