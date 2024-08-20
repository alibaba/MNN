//
//  UnaryBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef UnaryBufExecution_hpp
#define UnaryBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class UnaryBufExecution : public CommonExecution {
public:
    UnaryBufExecution(const std::string &compute, const MNN::Op* op, Backend *backend);
    virtual ~UnaryBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    ErrorCode SubgrouponResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalSize      = {1, 1, 1};
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* UnaryBufExecution_hpp */
#endif/* MNN_OPENCL_BUFFER_CLOSED */
