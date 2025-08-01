//
//  FuseExecution.cpp
//  MNN
//
//  Created by MNN on 2025/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/FuseBufExecution.hpp"
#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class FuseBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_Extra();
        if(param->type()->str() == "ExtraConvolution2DPrelu"){
            return new ConvBufExecution(inputs, outputs, op, backend, true);
        }
        return nullptr;
    }
};
REGISTER_OPENCL_OP_CREATOR(FuseBufCreator, OpType_Extra, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
