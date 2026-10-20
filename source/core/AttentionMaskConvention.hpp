//
//  AttentionMaskConvention.hpp
//  MNN
//
//  Mask contract shared by the LLM engine (producer) and every Attention backend (consumer).
//

#ifndef MNNAttentionMaskConvention_hpp
#define MNNAttentionMaskConvention_hpp

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

} // namespace MNN

#endif /* MNNAttentionMaskConvention_hpp */
