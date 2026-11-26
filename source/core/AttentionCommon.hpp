//
//  AttentionCommon.hpp
//  MNN
//
//  Conventions shared by the LLM engine (producer) and every Attention / LinearAttention backend (consumer).
//

#ifndef MNNAttentionCommon_hpp
#define MNNAttentionCommon_hpp

#include <cstddef>

namespace MNN {

// The Attention op's mask input is classified by shape only:
//   absent / rank 0       causal when kv_cache=true, otherwise no masking
//   rank 1, one element   bidirectional, nothing masked
//   anything else         dense additive mask, added to QK as-is
// Backends must never read the mask value to classify it: on Metal the host->device upload
// has not run yet at encode time, so the contents are stale.

// Conventional values written into a one-element mask. Debug aid only; never read.
constexpr float kAttentionMaskCausal        = 0.0f;
constexpr float kAttentionMaskBidirectional = 1.0f;
// Rank of the bidirectional one-element mask.
constexpr int kAttentionMaskBidirectionalDims = 1;

// `dims` = Tensor::dimensions(), `elems` = Tensor::elementSize(); `hasMask` is false when
// the op has no mask input.
inline bool IsAttentionMaskCausalSentinel(bool hasMask, int dims) {
    return !hasMask || dims == 0;
}
inline bool IsAttentionMaskBidirectionalSentinel(bool hasMask, int dims, int elems) {
    return hasMask && dims == kAttentionMaskBidirectionalDims && elems == 1;
}
// Whether the causal triangle applies. kv_cache=false ops (encoders, cross attention, fused
// FMHA without mask) must not be made causal.
inline bool IsAttentionMaskCausal(bool hasMask, int dims, bool kvCache) {
    return IsAttentionMaskCausalSentinel(hasMask, dims) && kvCache;
}

// ---- LinearAttention input layout ----
// Index of the short-conv weight among LinearAttention inputs; -1 when the variant has none.
//   inputs.size() >= 4 : index 3
//   inputs.size() == 2 : index 1 for the short-conv variant, otherwise -1
inline int linearAttentionWeightIndex(bool shortConv, std::size_t inputCount) {
    if (inputCount >= 4) {
        return 3;
    }
    return shortConv && inputCount == 2 ? 1 : -1;
}

} // namespace MNN

#endif /* MNNAttentionCommon_hpp */
