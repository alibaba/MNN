//
//  MatmulBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef MatMulBufExecution_hpp
#define MatMulBufExecution_hpp

#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class MatMulBufExecution : public Execution {
public:
    MatMulBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, bool transposeA, bool transposeB);
    virtual ~MatMulBufExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mTransposeA;
    bool mTransposeB;
    cl::Kernel mKernel;
    std::string mKernelName;
    uint32_t mMaxWorkGroupSize;
    std::vector<int> mInput0Shape;
    std::vector<int> mInput1Shape;
    OpenCLBackend *mOpenCLBackend;
    std::vector<uint32_t> mGlobalWorkSize{1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1};
};

} // namespace OpenCL
} // namespace MNN

#endif
#endif /* MNN_OPENCL_BUFFER_CLOSED */
