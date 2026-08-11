#ifndef _MUSA_INTERP_EXECUTION_HPP_
#define _MUSA_INTERP_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class InterpExecution : public Execution {
public:
    InterpExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~InterpExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::Interp* mOp;
    
    int mInBatch;
    int mInChannels;
    int mInHeight;
    int mInWidth;
    int mOutHeight;
    int mOutWidth;
    float mHeightScale;
    float mWidthScale;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
