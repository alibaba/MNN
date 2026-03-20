#ifndef _MUSA_GATHERV2_EXECUTION_HPP_
#define _MUSA_GATHERV2_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class GatherV2Execution : public Execution {
public:
    GatherV2Execution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~GatherV2Execution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::GatherV2* mOp;
    
    int mAxis;
    int mOuterDims;
    int mIndicesCount;
    int mInnerDims;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
