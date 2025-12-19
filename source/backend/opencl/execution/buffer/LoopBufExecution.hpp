//
//  LoopBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef LoopBufExecution_hpp
#define LoopBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class LoopBufExecution : public CommonExecution{
public:
    LoopBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn);
    virtual ~LoopBufExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode InitCommandOnEncode();
    ErrorCode LoopGather(const Tensor *output, int cmdIndex, int iter);
    ErrorCode LoopBatchMatMul(const Tensor *output, int cmdIndex, int iter);
    ErrorCode LoopBinary(const Tensor *outputs, int cmdIndex, int iter);
    ErrorCode LoopCumsum(const Tensor *output);
    ErrorCode FuseOutput(int iter, int* inputStride, int sizeZ, int sizeY, int SizeX, int n, int n_offset);
private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::shared_ptr<Tensor> mFuseTensor;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LoopBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
