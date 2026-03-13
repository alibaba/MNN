#ifndef _MUSA_LAYERNORM_EXECUTION_HPP_
#define _MUSA_LAYERNORM_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class LayerNormExecution : public Execution {
public:
    LayerNormExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~LayerNormExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::LayerNorm* mOp;
    
    float mEpsilon;
    int mOuterSize;
    int mInnerSize;
    int mGammaSize;
    int mBetaSize;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
