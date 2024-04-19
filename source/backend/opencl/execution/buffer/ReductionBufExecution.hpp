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

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ReductionBufExecution : public CommonExecution {
public:
    ReductionBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op* op, Backend* backend);
    virtual ~ReductionBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    int getLocalSize(int size, int maxGroupSize);
    OpenCLBackend *mOpenCLBackend;
    MNN::DataType mdataType;
    int mReductType;
    int mAxis;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1};
    bool mUseLocal = false;
    uint32_t mMaxWorkGroupSize;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReductionBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
