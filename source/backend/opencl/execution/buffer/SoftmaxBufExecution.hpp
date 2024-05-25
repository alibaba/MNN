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

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class SoftmaxBufExecution : public CommonExecution {
public:
    SoftmaxBufExecution(const std::vector<Tensor *> &inputs, int axis, const MNN::Op* Op, Backend *backend);

    virtual ~SoftmaxBufExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildSoftmaxKernel(int localSize);
private:
    int getLocalSize(int size, int maxGroupSize);
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
