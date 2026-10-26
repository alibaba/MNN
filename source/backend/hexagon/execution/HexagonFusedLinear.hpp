//
//  HexagonFusedLinear.hpp
//  MNN
//
//  Composite execution for the export-time fused projection op (OpType_FusedLinear).
//  The member conv1x1 / binary RMSNorm / MUL_SILU ops are the regular Hexagon executions, so
//  prefill is exactly the work of the geometry decomposition. Decode (M = 1) replaces the member
//  GEMVs with one DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED launch over their existing weight blobs:
//  QKV writes straight to the group outputs, gate/up applies up * silu(gate) in the epilogue.
//

#ifndef HexagonFusedLinear_hpp
#define HexagonFusedLinear_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <memory>
#include <vector>

#include "HexagonConvolution.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {

class HexagonFusedLinear : public Execution {
public:
    static HexagonFusedLinear* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs);
    ~HexagonFusedLinear() override = default;

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    // Immutable state shared by clones: serialized member ops (children reference into them) and
    // the conv resources the fused decode launch reads.
    struct Shared {
        std::vector<std::shared_ptr<BufferStorage>> convOps;
        std::shared_ptr<BufferStorage> mulSiluOp;
        std::shared_ptr<BufferStorage> layerNormOp;
        std::vector<std::shared_ptr<HexagonConvolution::Resource>> convResources; // empty: no fused decode
        bool gateUp = false;
        bool hasLn = false;
        int inputCount = 0;
    };

    HexagonFusedLinear(Backend* backend, std::shared_ptr<Shared> shared);
    bool fusedDecodeOk(int rows) const;

    std::shared_ptr<Shared> mShared;
    std::vector<std::shared_ptr<Execution>> mConvs;
    std::shared_ptr<Execution> mMulSilu;
    std::shared_ptr<Execution> mLn;

    std::shared_ptr<Tensor> mNormalized;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
    bool mUseFused = false;
    HexagonCommand mFusedCmd;
};

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* HexagonFusedLinear_hpp */
