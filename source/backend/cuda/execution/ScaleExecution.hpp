//
//  ScaleExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScaleExecution_hpp
#define ScaleExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class ScaleExecution : public Execution {
public:
    ScaleExecution(const Scale* scale, Backend *backend);
    virtual ~ScaleExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    CUDARuntime *mRuntime;
    void *mDeviceBias = nullptr;
    void *mDeviceScale = nullptr;
    int mCount;
    int mChannel;
    int mArea;
    MemChunk mScaleBiasStorage;
};

} // namespace CUDA
} // namespace MNN
#endif /* ScaleExecution_hpp */
