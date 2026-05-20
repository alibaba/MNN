#ifndef _MUSA_TOPKV2_EXECUTION_HPP_
#define _MUSA_TOPKV2_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class TopKV2Execution : public Execution {
public:
    TopKV2Execution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~TopKV2Execution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::TopKV2* mOp;
    
    int mAxis;
    int mK;
    int mOuterSize;
    int mInnerSize;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
