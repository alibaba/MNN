//
//  GridSampleExecution.hpp
//  MNN
//
//  Created by MNN on 2021/08/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GridSampleExecution_hpp
#define GridSampleExecution_hpp

#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {
class GridSampleExecution : public Execution {
public:
    GridSampleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~GridSampleExecution() = default;

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

#endif // GridSampleExecution_hpp