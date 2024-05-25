//
//  CastBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef CastBufExecution_hpp
#define CastBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class CastBufExecution : public CommonExecution {
public:
    CastBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::string &compute, const MNN::Op* Op, Backend *backend);
    virtual ~CastBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalSize      = {1, 1, 1};
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* CastBufExecution_hpp */
#endif/* MNN_OPENCL_BUFFER_CLOSED */
