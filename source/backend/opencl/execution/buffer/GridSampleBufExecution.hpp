//
//  GridSampleBufExecution.hpp
//  MNN
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef GridSampleBufExecution_hpp
#define GridSampleBufExecution_hpp

#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {
class GridSampleBufExecution : public Execution {
public:
    GridSampleBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~GridSampleBufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    SampleMode mMode;
    BorderMode mPaddingMode;
    int mAlignCorners;

    std::vector<uint32_t> mGlobalWorkSize{ 0,0,0,0 };
    std::vector<uint32_t> mLocalWorkSize{ 0,0,0,0 };

    cl::Kernel mKernel;
    std::string	mKernelName;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
};
} // namespace OpenCL
} // namespace MNN

#endif // GridSampleBufExecution_hpp
#endif // MNN_OPENCL_BUFFER_CLOSED