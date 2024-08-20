//
//  ArgMaxBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef ArgMaxBufExecution_hpp
#define ArgMaxBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ArgMaxBufExecution : public CommonExecution {
public:
    ArgMaxBufExecution(const std::string &compute, const MNN::Op *op, Backend *backend, const int axis);
    virtual ~ArgMaxBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    int getLocalSize(int size, int maxGroupSize);
private:
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalSize      = {1, 1, 1};
    std::set<std::string> mBuildOptions;
    int mAxis;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ArgMaxBufExecution_hpp */
#endif/* MNN_OPENCL_BUFFER_CLOSED */
