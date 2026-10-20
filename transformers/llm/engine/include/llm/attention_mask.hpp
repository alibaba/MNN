//
//  attention_mask.hpp
//  MNN
//
//  Single producer of the attention_mask input fed to exported LLM / vision / audio graphs.
//  Callers describe the mask (AttentionMaskType + shape); this file decides whether a backend can omit
//  the materialised plane. When it can, a one-element mask is passed instead -- rank-0 for
//  Causal, rank-1 for Bidirectional -- and the Attention op derives the masking itself. The
//  encoding is defined in source/core/AttentionMaskConvention.hpp.
//
//  Created by MNN on 2026/10/13.
//

#ifndef LLM_ATTENTION_MASK_hpp
#define LLM_ATTENTION_MASK_hpp

#include <map>
#include <string>
#include <vector>

#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace Transformer {

enum class AttentionMaskType {
    Causal,               // query i sees keys j <= i (right-aligned over history)
    Bidirectional,        // every query sees every key
    BlockDiagonal,        // each query sees only its own block (cuSeqlens)
    Tree,                 // speculative draft tree, explicit boolean matrix
    KeyPadding,           // trailing keys of each batch item are invisible (validLens)
    SlidingWindow,        // causal, limited to `slidingWindow` keys behind the query
    CausalBlockDiagonal,  // causal, confined to own block beyond a shared prefix (reranker)
    AllOnes,              // constant 1.0f plane, the embedding path's non-float mask
};

// Mask description. Common fields first; the rest are read only by the AttentionMaskType named.
// Pointer fields are not owned and must outlive the call.
//
// `rank` must match the attention_mask input the graph was exported with:
//   4 -> {1, 1, seqLen, kvSeqLen}        3 -> {1, seqLen, kvSeqLen}
//   KeyPadding: 4 -> {batch, 1, 1, kvSeqLen}, 2 -> {batch, kvSeqLen}
// A one-element mask ignores `rank`. `forceDense` keeps the plane even where it could be omitted.
struct MaskRequest {
    AttentionMaskType type   = AttentionMaskType::Causal;
    int seqLen      = 0;
    int kvSeqLen    = 0;  // 0 -> seqLen
    int rank        = 4;
    bool forceDense = false;

    const std::vector<int>* cuSeqlens = nullptr;           // BlockDiagonal, CausalBlockDiagonal
    int sharedPrefix                  = 0;                 // CausalBlockDiagonal
    const std::vector<std::vector<bool>>* tree = nullptr;  // Tree: rows x (visiblePrefix + cols)
    int visiblePrefix                 = 0;                 // Tree
    int batch                         = 1;                 // KeyPadding
    const std::vector<int>* validLens = nullptr;           // KeyPadding
    int slidingWindow                 = 0;                 // SlidingWindow
};

// One plane of a two-plane mask, see buildTwoPlaneMask.
struct MaskPlane {
    AttentionMaskType type                     = AttentionMaskType::Causal;
    const std::vector<int>* cuSeqlens = nullptr;  // BlockDiagonal
    int slidingWindow                 = 0;        // SlidingWindow
};

// ---------------------------------------------------------------------------------------------
// Building masks
// ---------------------------------------------------------------------------------------------

// Whether `backendType` can omit the plane for `type`. Only Causal and Bidirectional can. A
// one-element Causal mask is causal only when the op has kv_cache=true.
MNN_PUBLIC bool canOmitMaskPlane(AttentionMaskType type, const std::string& backendType);

// One-element mask when the plane can be omitted and not forceDense, otherwise a new dense
// plane. An empty `backendType` always yields a plane.
MNN_PUBLIC Express::VARP buildAttentionMask(const MaskRequest& req, const std::string& backendType = std::string());

// plane0 stacked before plane1 on a new leading axis: rank 4 -> {2,1,1,q,kv}, rank 3 -> {2,1,q,kv}.
// Reads only seqLen / kvSeqLen / rank from `req`.
MNN_PUBLIC Express::VARP buildTwoPlaneMask(const MaskRequest& req, const MaskPlane& plane0, const MaskPlane& plane1);

// ---------------------------------------------------------------------------------------------
// Filling caller-owned planes ([seqLen, kvSeqLen], additive: 0 visible, lowest() masked)
// ---------------------------------------------------------------------------------------------

MNN_PUBLIC void fillCausal(float* ptr, int seqLen, int kvSeqLen);
MNN_PUBLIC void fillSlidingWindow(float* ptr, int seqLen, int kvSeqLen, int window);
// Block b is [cuSeqlens[b], cuSeqlens[b+1]); out-of-range boundaries are clamped.
MNN_PUBLIC void fillBlockDiagonal(float* ptr, int seqLen, int kvSeqLen, const std::vector<int>& cuSeqlens);
MNN_PUBLIC void fillCausalBlockDiagonal(float* ptr, int seqLen, int kvSeqLen, int sharedPrefix,
                                        const std::vector<int>& cuSeqlens);
// Int plane for attention_mask == "int": 1 where key j <= i + rowOffset, else 0.
MNN_PUBLIC void fillCausalInt(int* ptr, int seqLen, int kvSeqLen, int rowOffset);
// Two consecutive planes, as laid out by buildTwoPlaneMask.
MNN_PUBLIC void fillTwoPlane(float* ptr, const MaskRequest& req, const MaskPlane& plane0, const MaskPlane& plane1);

// {0, blockSize, 2*blockSize, ..., seqLen}. blockSize <= 0 gives one block; seqLen <= 0 gives {0}.
MNN_PUBLIC std::vector<int> uniformBoundaries(int seqLen, int blockSize);

// ---------------------------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------------------------

// Reuses the mask VARP for repeated shapes. Only Causal and Bidirectional are cached, since the
// other types carry per-call contents. Use one cache per module; cleared when it exceeds
// kMaxEntries.
class MNN_PUBLIC MaskCache {
public:
    Express::VARP obtain(const MaskRequest& req, const std::string& backendType = std::string());
    void clear() {
        mVars.clear();
    }

private:
    static const size_t kMaxEntries = 8;
    std::map<std::string, Express::VARP> mVars;
};

}  // namespace Transformer
}  // namespace MNN

#endif /* LLM_ATTENTION_MASK_hpp */
