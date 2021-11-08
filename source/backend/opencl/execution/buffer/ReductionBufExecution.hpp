//
//  ReductionBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef ReductionBufExecution_hpp
#define ReductionBufExecution_hpp

#include "core/Execution.hpp"
#include <MNN_generated.h>
#include <vector>
#include <string.h>
#include <unordered_set>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ReductionBufExecution : public CommonExecution {
public:
    ReductionBufExecution(const MNN::Op* op, Backend* backend);
    virtual ~ReductionBufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    cl::Kernel mReduct1DKernel;
    std::string mKernelName;
    OpenCLBackend *mOpenCLBackend;
    MNN::DataType mdataType;
    int mReductType;
    std::vector<int> mAxis;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1};
    bool mUseLocal = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReductionBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
