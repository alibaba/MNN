//
//  FuseExecution.hpp
//  MNN
//
//  Created by MNN on 2022/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FuseExecution_hpp
#define FuseExecution_hpp

#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "backend/opencl/execution/image/CommonExtension.hpp"

namespace MNN {
namespace OpenCL {

class FuseExecution : public Execution, public CommonExtension {
public:
    FuseExecution(const std::vector<Tensor *> &inputs, Backend *backend, const Op* op);

    virtual ~FuseExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildFuseKernel(const Op* op);
private:
    std::string mKernelName;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
};
} // namespace OpenCL
} // namespace MNN
#endif /* FuseExecution_hpp */
