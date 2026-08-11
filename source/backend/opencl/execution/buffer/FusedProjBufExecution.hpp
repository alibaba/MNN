//
//  FusedProjBufExecution.hpp
//  MNN
//
//  OpenCL (buffer mode) execution for the export-time fused projection op
//  (OpType_FusedLinear): both the gate/up flavour (act_silu_mul, 2 convs)
//  and the QKV flavour (3-4 convs writing straight to the outputs).
//
//  See FusedProjBufExecution.cpp for why this container exists.
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef FusedProjBufExecution_hpp
#define FusedProjBufExecution_hpp

#include <vector>
#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/AutoStorage.h"

namespace MNN {
namespace OpenCL {

// The synthetic member ops. Shared across clones: each clone's child
// executions keep raw `const Op*` pointers into these buffers, and
// re-serializing them per clone would copy every folded weight blob again.
struct FusedProjSubOps {
    std::vector<std::shared_ptr<BufferStorage>> convs;
    std::shared_ptr<BufferStorage> mulSilu;
    std::shared_ptr<BufferStorage> layerNorm;
};

class FusedProjBufExecution : public CommonExecution {
public:
    FusedProjBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                          const MNN::Op *op, Backend *backend);
    FusedProjBufExecution(std::shared_ptr<FusedProjSubOps> subOps, const MNN::Op *op, Backend *backend);
    virtual ~FusedProjBufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend *bn, const Op *op, Execution **dst) override;

private:
    bool _createConvs(Backend *backend);
    bool _createRest(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode _resize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    // Declared before the children so it outlives them: the child executions
    // hold raw `const Op*` pointers into these buffers.
    std::shared_ptr<FusedProjSubOps> mSubOps;
    std::vector<std::shared_ptr<Execution>> mConvs;
    // gate/up only: the SiLU-mul child and the two projection intermediates.
    std::shared_ptr<Execution> mMulSilu;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
    std::shared_ptr<Execution> mLn;
    std::shared_ptr<Tensor> mNormalized;
    const FusedLinearParam *mParam = nullptr;
    bool mIsGateUp = false;
    bool mHasLn    = false;
    int mNumConvs  = 0;
    // Projection outputs the group produces: 1 for gate/up (the SiLU-mul
    // result), otherwise one per conv. Also the index of residual_out.
    int mNumProjOut = 0;
};

} // namespace OpenCL
} // namespace MNN

#endif /* FusedProjBufExecution_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
