//
//  GatherV2Execution.hpp
//  MNN
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GatherV2Execution_hpp
#define GatherV2Execution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
namespace MNN {
namespace CUDA {
class GatherV2Execution : public Execution {
public:
    GatherV2Execution(const Op* op, Backend *backend);
    virtual ~GatherV2Execution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Op* mOp;
    int mAxis;
    int mInside;
    int mOutside;
    int mInpNum;
    int mOutNum;
};
} // namespace CUDA
} // namespace MNN

#endif
