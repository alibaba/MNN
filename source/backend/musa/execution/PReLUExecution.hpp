#ifndef _MUSA_PRELU_EXECUTION_HPP_
#define _MUSA_PRELU_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class PReLUExecution : public Execution {
public:
    PReLUExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~PReLUExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::PReLU* mOp;
    
    int mTotalSize;
    int mChannels;
    int mInnerDims;
    int mSlopeSize;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
