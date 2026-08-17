#ifndef HexagonAttentionUtils_hpp
#define HexagonAttentionUtils_hpp

namespace MNN {

static inline int visionAttentionWorkerSlots(int maxThreads) {
    return maxThreads > 0 ? maxThreads : 1;
}

// rank < 2 is treated as no mask. A 2-D mask is valid only for batch 1; batched
// attention must provide [batch, tokens, tokens].
static inline bool validateVisionAttentionMaskShape(int batch, int tokens, int rank, const int* dimensions) {
    if (rank < 2) {
        return true;
    }
    if (dimensions == nullptr || batch <= 0 || tokens <= 0) {
        return false;
    }
    if (rank == 2) {
        return batch == 1 && dimensions[0] == tokens && dimensions[1] == tokens;
    }
    if (rank == 3) {
        return dimensions[0] == batch && dimensions[1] == tokens && dimensions[2] == tokens;
    }
    return false;
}

} // namespace MNN

#endif
