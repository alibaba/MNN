#ifndef _MUSA_TRANSPOSE_EXECUTION_HPP_
#define _MUSA_TRANSPOSE_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class TransposeExecution : public Execution {
public:
    TransposeExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~TransposeExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::Transpose* mOp;
    
    int mDims;
    int mTotalSize;
    std::vector<int> mPerm;
    std::vector<int> mInputStrides;
    std::vector<int> mOutputStrides;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
