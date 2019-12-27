//
//  ZerosLikeExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ZerosLikeExecution_hpp
#define ZerosLikeExecution_hpp

#include "core/Execution.hpp"
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class ZerosLikeExecution : public Execution {
public:
    ZerosLikeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ZerosLikeExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ZerosLikeExecution_hpp */
