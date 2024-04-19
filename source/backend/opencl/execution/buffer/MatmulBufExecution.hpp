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

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class MatMulBufExecution : public CommonExecution {
public:
    MatMulBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, bool transposeA, bool transposeB);
    virtual ~MatMulBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mTransposeA;
    bool mTransposeB;
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
