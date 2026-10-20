//
//  attention_mask.cpp
//  MNN
//
//  Created by MNN on 2026/10/13.
//

#include "llm/attention_mask.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <MNN/expr/ExprCreator.hpp>

#include "core/AttentionMaskConvention.hpp"

namespace MNN {
namespace Transformer {

const size_t MaskCache::kMaxEntries;

bool canOmitMaskPlane(AttentionMaskType type, const std::string& backendType) {
    // Keep in sync with source/core/AttentionMaskConvention.hpp.
    // cuda / vulkan read an absent or rank-0 mask as fully visible regardless of kv_cache, and
    // index a rank-1 mask as a dense plane, so they always need the plane.
    if (type == AttentionMaskType::Causal) {
        // hexagon: not re-verified, kept because Llm has always sent it a one-element mask.
        return backendType == "cpu" || backendType == "hexagon" || backendType == "metal" || backendType == "opencl";
    }
    if (type == AttentionMaskType::Bidirectional) {
        return backendType == "cpu" || backendType == "metal" || backendType == "opencl";
    }
    return false;
}

static const std::vector<int>& orEmpty(const std::vector<int>* v) {
    static const std::vector<int> kEmpty;
    return v == nullptr ? kEmpty : *v;
}

static void fillTree(float* ptr, int rows, int cols, const MaskRequest& req) {
    const float lowest = std::numeric_limits<float>::lowest();
    for (int i = 0; i < rows * cols; ++i) {
        ptr[i] = lowest;
    }
    const auto& tree = *req.tree;
    for (int i = 0; i < rows; ++i) {
        float* row = ptr + (size_t)cols * i;
        for (int j = 0; j < req.visiblePrefix && j < cols; ++j) {
            row[j] = 0.0f;
        }
        const auto& line = tree[i];
        for (int j = 0; j < (int)line.size() && req.visiblePrefix + j < cols; ++j) {
            row[req.visiblePrefix + j] = line[j] ? 0.0f : lowest;
        }
    }
}

static void fillKeyPadding(float* ptr, int cols, const MaskRequest& req) {
    const float lowest = std::numeric_limits<float>::lowest();
    for (int b = 0; b < req.batch; ++b) {
        int valid = cols;
        if (req.validLens != nullptr && b < (int)req.validLens->size()) {
            valid = std::max(0, std::min((*req.validLens)[b], cols));
        }
        float* row = ptr + (size_t)cols * b;
        for (int j = 0; j < cols; ++j) {
            row[j] = j < valid ? 0.0f : lowest;
        }
    }
}

Express::VARP buildAttentionMask(const MaskRequest& req, const std::string& backendType) {
    using namespace MNN::Express;
    const int kvSeqLen = req.kvSeqLen > 0 ? req.kvSeqLen : req.seqLen;

    if (!req.forceDense && canOmitMaskPlane(req.type, backendType)) {
        // Backends classify by shape only: rank-1 one element = bidirectional, rank-0 = causal.
        const bool bidirectional = req.type == AttentionMaskType::Bidirectional;
        auto mask = bidirectional ? _Input({1}, NCHW, halide_type_of<float>())
                                  : _Input({}, NCHW, halide_type_of<float>());
        mask->writeMap<float>()[0] = bidirectional ? kAttentionMaskBidirectional : kAttentionMaskCausal;
        return mask;
    }

    int rows = req.seqLen, cols = kvSeqLen;
    if (req.type == AttentionMaskType::Tree) {
        if (req.tree == nullptr || req.tree->empty() || (*req.tree)[0].empty()) {
            return nullptr;
        }
        rows = (int)req.tree->size();
        cols = req.visiblePrefix + (int)(*req.tree)[0].size();
    }
    std::vector<int> dims;
    if (req.type == AttentionMaskType::KeyPadding) {
        dims = req.rank == 2 ? std::vector<int>{req.batch, cols} : std::vector<int>{req.batch, 1, 1, cols};
    } else {
        dims = req.rank == 3 ? std::vector<int>{1, rows, cols} : std::vector<int>{1, 1, rows, cols};
    }
    auto mask = _Input(dims, NCHW, halide_type_of<float>());
    auto ptr  = mask->writeMap<float>();
    if (nullptr == ptr) {
        return nullptr;
    }
    switch (req.type) {
    case AttentionMaskType::Bidirectional:
        ::memset(ptr, 0, (size_t)rows * (size_t)cols * sizeof(float));
        break;
    case AttentionMaskType::BlockDiagonal:
        fillBlockDiagonal(ptr, rows, cols, orEmpty(req.cuSeqlens));
        break;
    case AttentionMaskType::Tree:
        fillTree(ptr, rows, cols, req);
        break;
    case AttentionMaskType::KeyPadding:
        fillKeyPadding(ptr, cols, req);
        break;
    case AttentionMaskType::SlidingWindow:
        fillSlidingWindow(ptr, rows, cols, req.slidingWindow);
        break;
    case AttentionMaskType::CausalBlockDiagonal:
        fillCausalBlockDiagonal(ptr, rows, cols, req.sharedPrefix, orEmpty(req.cuSeqlens));
        break;
    case AttentionMaskType::AllOnes:
        std::fill(ptr, ptr + (size_t)rows * cols, 1.0f);
        break;
    case AttentionMaskType::Causal:
    default:
        fillCausal(ptr, rows, cols);
        break;
    }
    return mask;
}

void fillBlockDiagonal(float* ptr, int seqLen, int kvSeqLen, const std::vector<int>& cuSeqlens) {
    if (nullptr == ptr || seqLen <= 0 || kvSeqLen <= 0) {
        return;
    }
    const float lowest = std::numeric_limits<float>::lowest();
    std::fill(ptr, ptr + (size_t)seqLen * kvSeqLen, lowest);
    for (size_t b = 1; b < cuSeqlens.size(); ++b) {
        const int begin  = std::max(cuSeqlens[b - 1], 0);
        const int rowEnd = std::min(cuSeqlens[b], seqLen);
        const int colEnd = std::min(cuSeqlens[b], kvSeqLen);
        for (int i = begin; i < rowEnd; ++i) {
            float* row = ptr + (size_t)kvSeqLen * i;
            for (int j = begin; j < colEnd; ++j) {
                row[j] = 0.0f;
            }
        }
    }
}

std::vector<int> uniformBoundaries(int seqLen, int blockSize) {
    const int len = std::max(seqLen, 0);
    std::vector<int> boundaries(1, 0);
    if (blockSize <= 0) {
        if (len > 0) {
            boundaries.push_back(len);
        }
        return boundaries;
    }
    for (int cur = blockSize; cur < len; cur += blockSize) {
        boundaries.push_back(cur);
    }
    if (boundaries.back() != len) {
        boundaries.push_back(len);
    }
    return boundaries;
}

void fillCausal(float* ptr, int seqLen, int kvSeqLen) {
    if (nullptr == ptr || seqLen <= 0 || kvSeqLen <= 0) {
        return;
    }
    // Same expression as the previous LLM / embedding code, so those planes stay byte-identical.
    const float lowest = std::numeric_limits<float>::lowest();
    const int offset   = kvSeqLen - seqLen;
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < kvSeqLen; ++j) {
            ptr[(size_t)kvSeqLen * i + j] = (j > i + offset) * lowest;
        }
    }
}

void fillSlidingWindow(float* ptr, int seqLen, int kvSeqLen, int window) {
    if (nullptr == ptr || seqLen <= 0 || kvSeqLen <= 0) {
        return;
    }
    const float lowest = std::numeric_limits<float>::lowest();
    const int offset   = kvSeqLen - seqLen;
    for (int i = 0; i < seqLen; ++i) {
        const int queryPos = i + offset;
        for (int j = 0; j < kvSeqLen; ++j) {
            const bool allowed            = (j <= queryPos) && (j > queryPos - window);
            ptr[(size_t)kvSeqLen * i + j] = allowed ? 0.0f : lowest;
        }
    }
}

void fillCausalBlockDiagonal(float* ptr, int seqLen, int kvSeqLen, int sharedPrefix,
                             const std::vector<int>& cuSeqlens) {
    if (nullptr == ptr || seqLen <= 0 || kvSeqLen <= 0) {
        return;
    }
    const float lowest = std::numeric_limits<float>::lowest();
    const int offset   = kvSeqLen - seqLen;
    for (int i = 0; i < seqLen; ++i) {
        const int queryPos = i + offset;
        // Rows outside every block (shared prefix, gaps) stay plainly causal.
        int blockBegin = 0, blockEnd = -1;
        for (size_t b = 1; b < cuSeqlens.size(); ++b) {
            if (queryPos >= cuSeqlens[b - 1] && queryPos < cuSeqlens[b]) {
                blockBegin = cuSeqlens[b - 1];
                blockEnd   = cuSeqlens[b];
                break;
            }
        }
        float* row = ptr + (size_t)kvSeqLen * i;
        for (int j = 0; j < kvSeqLen; ++j) {
            const bool causal  = j <= queryPos;
            const bool visible = blockEnd < 0 ? causal
                                              : (causal && (j < sharedPrefix || (j >= blockBegin && j < blockEnd)));
            row[j] = visible ? (j > queryPos) * lowest : lowest;
        }
    }
}

void fillCausalInt(int* ptr, int seqLen, int kvSeqLen, int rowOffset) {
    if (nullptr == ptr) {
        return;
    }
    for (int i = 0; i < seqLen; ++i) {
        const int row = i + rowOffset;
        for (int j = 0; j < kvSeqLen; ++j) {
            ptr[kvSeqLen * i + j] = (int)(j <= row);
        }
    }
}

static void fillPlane(float* ptr, int seqLen, int kvSeqLen, const MaskPlane& plane) {
    switch (plane.type) {
    case AttentionMaskType::Bidirectional:
        ::memset(ptr, 0, (size_t)seqLen * (size_t)kvSeqLen * sizeof(float));
        return;
    case AttentionMaskType::BlockDiagonal:
        fillBlockDiagonal(ptr, seqLen, kvSeqLen, orEmpty(plane.cuSeqlens));
        return;
    case AttentionMaskType::SlidingWindow:
        fillSlidingWindow(ptr, seqLen, kvSeqLen, plane.slidingWindow);
        return;
    case AttentionMaskType::Causal:
    default:
        fillCausal(ptr, seqLen, kvSeqLen);
        return;
    }
}

void fillTwoPlane(float* ptr, const MaskRequest& req, const MaskPlane& plane0, const MaskPlane& plane1) {
    const int kvSeqLen = req.kvSeqLen > 0 ? req.kvSeqLen : req.seqLen;
    if (nullptr == ptr || req.seqLen <= 0 || kvSeqLen <= 0) {
        return;
    }
    const size_t planeElems = (size_t)req.seqLen * (size_t)kvSeqLen;
    fillPlane(ptr, req.seqLen, kvSeqLen, plane0);
    fillPlane(ptr + planeElems, req.seqLen, kvSeqLen, plane1);
}

Express::VARP buildTwoPlaneMask(const MaskRequest& req, const MaskPlane& plane0, const MaskPlane& plane1) {
    using namespace MNN::Express;
    const int kvSeqLen = req.kvSeqLen > 0 ? req.kvSeqLen : req.seqLen;
    std::vector<int> dims;
    if (req.rank == 3) {
        dims = {2, 1, req.seqLen, kvSeqLen};
    } else {
        dims = {2, 1, 1, req.seqLen, kvSeqLen};
    }
    auto mask = _Input(dims, NCHW, halide_type_of<float>());
    auto ptr  = mask->writeMap<float>();
    if (nullptr == ptr) {
        return nullptr;
    }
    fillTwoPlane(ptr, req, plane0, plane1);
    return mask;
}

Express::VARP MaskCache::obtain(const MaskRequest& req, const std::string& backendType) {
    if (req.type != AttentionMaskType::Causal && req.type != AttentionMaskType::Bidirectional) {
        return buildAttentionMask(req, backendType);
    }
    const int kvSeqLen = req.kvSeqLen > 0 ? req.kvSeqLen : req.seqLen;
    char key[160];
    ::snprintf(key, sizeof(key), "%s|%d|%d|%d|%d|%d", backendType.c_str(), (int)req.type, req.seqLen, kvSeqLen,
               req.rank, req.forceDense ? 1 : 0);
    auto it = mVars.find(key);
    if (it != mVars.end()) {
        return it->second;
    }
    if (mVars.size() >= kMaxEntries) {
        mVars.clear();
    }
    auto var = buildAttentionMask(req, backendType);
    if (nullptr != var) {
        mVars[key] = var;
    }
    return var;
}

}  // namespace Transformer
}  // namespace MNN
