#ifndef _MUSA_CAST_EXECUTION_HPP_
#define _MUSA_CAST_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace CUDA {

class CastExecution : public Execution {
public:
    CastExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~CastExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::Cast* mOp;
    
    int mTotalSize;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace CUDA
} // namespace MNN

#endif
