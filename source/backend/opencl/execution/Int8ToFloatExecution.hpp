//
//  Int8ToFloatExecution.hpp
//  MNN
//
//  Created by MNN on 2019/6/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Int8ToFloatExecution_hpp
#define Int8ToFloatExecution_hpp

#include "CommonExecution.hpp"
#include <MNN_generated.h>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class Int8ToFloatExecution : public Execution {
public:
    Int8ToFloatExecution(Backend *backend, const MNN::Op *param);
    virtual ~Int8ToFloatExecution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScales;
    std::shared_ptr<cl::Buffer> mScaleBuffer;
    OpenCLBackend *mOpenCLBackend;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
};
}
} // namespace MNN

#endif /* Int8ToFloatExecution_hpp */
