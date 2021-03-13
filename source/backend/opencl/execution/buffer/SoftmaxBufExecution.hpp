//
//  SoftmaxBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef SoftmaxBufExecution_hpp
#define SoftmaxBufExecution_hpp

#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class SoftmaxBufExecution : public Execution {
public:
    SoftmaxBufExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend);

    virtual ~SoftmaxBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildSoftmaxKernel();
private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* SoftmaxBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
