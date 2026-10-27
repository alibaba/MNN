//
//  HexagonLinearAttention.hpp
//  MNN
//
//  OpType_LinearAttention on the DSP (DSP_OP_LINEAR_ATTN): gated_delta_rule (Qwen3.5) and short_conv
//  (LFM2) in the packed C4 LLM layout. Parameters the kernel does not cover are refused at creation, so
//  the op stays on the CPU backup. Layouts cannot be refused there: StaticModule pre-creates this op
//  with no tensors (see _tensorsSupported), so that check runs in onBuildCmd and fails resize instead.
//

#ifndef HexagonLinearAttention_hpp
#define HexagonLinearAttention_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <memory>
#include <vector>

#include "HexagonExecution.hpp"
#include "htp_command.h"

namespace MNN {
struct KVMeta;

class HexagonLinearAttention : public HexagonExecution {
public:
    static HexagonLinearAttention* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs);
    ~HexagonLinearAttention() override = default;

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    // Persistent per-layer state, shared by the prefill and decode clones (like the CPU StateCache).
    // These buffers are STATIC, so they must be handed back explicitly. That happens in ~State rather than
    // in ~HexagonLinearAttention because the clones share one State and only the last reference may free
    // it. allocBackend is whichever backend actually allocated them (the first execution to resize).
    struct State {
        ~State();
        Backend* allocBackend = nullptr;
        std::shared_ptr<Tensor> convState;
        std::shared_ptr<Tensor> rnnState;
        std::shared_ptr<Tensor> gateConst;
        bool needZero = true; // freshly allocated memory is undefined
    };

    HexagonLinearAttention(Backend* backend, const DSPLinearAttnParam& param, std::vector<float> gateConst,
                           std::shared_ptr<State> state);
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    DSPLinearAttnParam mParam;
    std::vector<float> mGateConstHost; // [coef[Hv], bias[Hv]] when gate_fold
    std::shared_ptr<State> mState;
    std::vector<HexagonCommand> mZeroCmd;
    std::shared_ptr<Tensor> mWorkspace;
    KVMeta* mMeta = nullptr;
};

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* HexagonLinearAttention_hpp */
