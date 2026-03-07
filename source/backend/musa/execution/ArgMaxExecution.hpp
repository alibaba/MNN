#ifndef _MUSA_ARGMAX_EXECUTION_HPP_
#define _MUSA_ARGMAX_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace CUDA {

class ArgMaxExecution : public Execution {
public:
    ArgMaxExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~ArgMaxExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::ArgMax* mOp;
    
    int mAxis;
    int mOuterSize;
    int mAxisSize;
    int mInnerSize;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace CUDA
} // namespace MNN

#endif
