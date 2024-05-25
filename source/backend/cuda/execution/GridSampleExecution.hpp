//
//  GridSampleExecution.hpp
//  MNN
//
//  Created by MNN on 2023/03/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GridSampleExecution_hpp
#define GridSampleExecution_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"
#include "backend/cuda/core/CUDABackend.hpp"
#include "MNNCUDAFunction.cuh"
#include "MNNCUDADefine.hpp"

namespace MNN {
namespace CUDA {
class GridSampleExecution : public Execution {
public:
    GridSampleExecution(Backend *backend, SampleMode mode, BorderMode paddingMode, bool alignCorners);
    virtual ~GridSampleExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    SampleMode mMode;
    BorderMode mPaddingMode;
    bool mAlignCorners;
    int mCount;
    int mBatch;
    int mChannel;
    int mInputHeight;
    int mInputWidth;
    int mOutputHeight;
    int mOutputWidth;
    int mChannelPack;
    int mInputDepth;
    int mOutputDepth;
};

} // namespace CUDA
} // namespace MNN
#endif /* SelectExecution_hpp */
