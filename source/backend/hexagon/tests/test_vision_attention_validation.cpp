#include "source/backend/hexagon/execution/HexagonAttentionUtils.hpp"

#include <cstdio>

using MNN::validateVisionAttentionMaskShape;
using MNN::visionAttentionWorkerSlots;

static bool expect(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL: %s\n", name);
        return false;
    }
    return true;
}

int main() {
    bool ok = true;
    const int rank1[] = {64};
    const int valid2D[] = {64, 64};
    const int valid3D[] = {2, 64, 64};
    const int wrongBatch[] = {1, 64, 64};
    const int wrongQuery[] = {2, 63, 64};
    const int wrongStride[] = {2, 64, 63};
    const int rank4[] = {1, 2, 64, 64};

    ok &= expect(validateVisionAttentionMaskShape(2, 64, 0, nullptr), "no mask");
    ok &= expect(validateVisionAttentionMaskShape(2, 64, 1, rank1), "rank-1 treated as no mask");
    ok &= expect(validateVisionAttentionMaskShape(1, 64, 2, valid2D), "2-D batch-1 mask");
    ok &= expect(!validateVisionAttentionMaskShape(2, 64, 2, valid2D), "2-D batched mask rejected");
    ok &= expect(validateVisionAttentionMaskShape(2, 64, 3, valid3D), "3-D batched mask");
    ok &= expect(!validateVisionAttentionMaskShape(2, 64, 3, wrongBatch), "wrong batch rejected");
    ok &= expect(!validateVisionAttentionMaskShape(2, 64, 3, wrongQuery), "wrong query extent rejected");
    ok &= expect(!validateVisionAttentionMaskShape(2, 64, 3, wrongStride), "wrong stride rejected");
    ok &= expect(!validateVisionAttentionMaskShape(2, 64, 4, rank4), "rank-4 rejected");
    ok &= expect(visionAttentionWorkerSlots(6) == 6, "all physical worker slots reserved");
    ok &= expect(visionAttentionWorkerSlots(0) == 1, "invalid worker count falls back to one slot");

    if (ok) {
        std::printf("PASS: Vision attention mask validation\n");
    }
    return ok ? 0 : 1;
}
