//
//  LoopExecution.hpp
//  MNN
//
//  Created by MNN on 2023/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#ifndef LoopExecution_hpp
#define LoopExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class LoopExecution : public CommonExecution{
public:
    LoopExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn);
    virtual ~LoopExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    void ImageToBufferAllTensor();
    void BufferToImageOutputTensor(const std::vector<Tensor *> &outputs);
    ErrorCode InitCommandOnEncode();
    ErrorCode LoopGather(int cmdIndex, int iter);
    ErrorCode LoopBatchMatMul(int cmdIndex, int iter);
    ErrorCode LoopBinary(int cmdIndex, int iter);
    ErrorCode LoopCumsum();
    ErrorCode FuseOutput(int iter, int* inputStride, int sizeZ, int sizeY, int SizeX, int n, int n_offset);
private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    cl::Buffer* mFuseBuffer;
    std::map<const Tensor*, cl::Buffer*> mTmpBuffers;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LoopExecution_hpp */
