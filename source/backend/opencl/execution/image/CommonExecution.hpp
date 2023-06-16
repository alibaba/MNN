//
//  CommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonExecution_hpp
#define CommonExecution_hpp
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "backend/opencl/execution/image/CommonExtension.hpp"
namespace MNN {
namespace OpenCL {

class CommonExecution : public Execution, public CommonExtension {
public:
    CommonExecution(Backend *backend, const MNN::Op *Op);
    virtual ~CommonExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    struct Unit {
        cl::Kernel kernel;
        cl::NDRange globalWorkSize;
        cl::NDRange localWorkSize;
    };
    std::vector<Unit> mUnits;
    const MNN::Op *mOp;
    OpType mOpType;
};
} // namespace OpenCL
} // namespace MNN
#endif /* CommonExecution_hpp */
