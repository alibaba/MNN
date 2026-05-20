#ifndef _MUSA_ARGMIN_EXECUTION_HPP_
#define _MUSA_ARGMIN_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class ArgMinExecution : public Execution {
public:
    ArgMinExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~ArgMinExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::ArgMin* mOp;
    
    int mAxis;
    int mOuterSize;
    int mAxisSize;
    int mInnerSize;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
