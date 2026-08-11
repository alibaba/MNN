//
//  FusedProjExecution.hpp
//  MNN
//
//  CUDA composite execution for the fused projection op (OpType_FusedLinear).
//  Mirrors the OpenCL FusedProjBufExecution: the member conv1x1 / binary
//  RMSNorm / MUL_SILU ops are driven as child executions inside one execution,
//  which is byte-for-byte the work the geometry decomposition would have
//  emitted. Keeping the op whole is what lets a later change collapse those
//  dispatches.
//

#ifndef FusedProjExecution_hpp
#define FusedProjExecution_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "core/Execution.hpp"
#include "core/AutoStorage.h"
#include "MNN_generated.h"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

// The serialized member sub-ops. Shared between an execution and its clones so
// the flatbuffer bytes the children reference outlive every copy.
struct FusedProjSubOps {
    std::vector<std::shared_ptr<BufferStorage>> convs;
    std::shared_ptr<BufferStorage> mulSilu;
    std::shared_ptr<BufferStorage> layerNorm;
};

class FusedProjExecution : public Execution {
public:
    FusedProjExecution(const MNN::Op* op, Backend* backend);
    FusedProjExecution(std::shared_ptr<FusedProjSubOps> subOps, const MNN::Op* op, Backend* backend);
    virtual ~FusedProjExecution() = default;

    bool valid() const {
        return mValid;
    }
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    bool _createConvs(Backend* backend);
    bool _createRest(Backend* backend, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode _resize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    const FusedLinearParam* mParam = nullptr;
    bool mValid    = true;
    bool mIsGateUp = false;
    bool mHasLn    = false;
    int mNumConvs   = 0;
    int mNumProjOut = 0;

    std::shared_ptr<FusedProjSubOps> mSubOps;
    std::vector<std::shared_ptr<Execution>> mConvs;
    std::shared_ptr<Execution> mMulSilu;
    std::shared_ptr<Execution> mLn;

    // Intermediates, re-acquired from the dynamic pool on every onResize.
    std::shared_ptr<Tensor> mNormalized;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
};

} // namespace CUDA
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* FusedProjExecution_hpp */
