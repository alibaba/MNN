//
//  BinaryBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef BinaryBufExecution_hpp
#define BinaryBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class BinaryBufExecution : public CommonExecution {
public:
    BinaryBufExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend);
    virtual ~BinaryBufExecution() = default;
    uint32_t realSize(const Tensor* tensor);
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    ErrorCode SubgroupOnResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    std::string mCompute;
    std::set<std::string> mBuildOptions;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize = {1, 1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* BinaryBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
