#ifndef _MUSA_DECONV_EXECUTION_HPP_
#define _MUSA_DECONV_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class DeconvExecution : public Execution {
public:
    DeconvExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~DeconvExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::Convolution2D* mOp;
    
    int mBatch;
    int mInChannels;
    int mOutChannels;
    int mInHeight;
    int mInWidth;
    int mOutHeight;
    int mOutWidth;
    int mKernelH;
    int mKernelW;
    int mStrideH;
    int mStrideW;
    int mPadH;
    int mPadW;
    int mDilationH;
    int mDilationW;
    int mGroup;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
