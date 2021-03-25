//
//  InterpBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef InterpBufExecution_hpp
#define InterpBufExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class InterpBufExecution : public Execution {
public:
    InterpBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~InterpBufExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    cl::Kernel mKernel;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    uint32_t mMaxWorkGroupSize;
    std::string mKernelName;
    OpenCLBackend *mOpenCLBackend;
    float mCordTransform[4];
};

} // namespace OpenCL
} // namespace MNN
#endif /* InterpBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
