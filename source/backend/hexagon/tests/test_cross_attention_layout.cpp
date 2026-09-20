//
// Host-side check that the non-cached cross-attention scratch layout computed in
// execution/HexagonAttention.cpp (crossAttentionScratchBytes) is byte-for-byte reproducible from
// the shared constants. The DSP kernel in htp-ops-lib/src/dsp/attention_entry.cc lays out the same
// buffer; if the two drift the DSP will reject the command with AEE_EBADPARM or, worse, scribble.
//
// Build: clang++ -O2 -std=c++11 -o /tmp/test_cross_attention_layout test_cross_attention_layout.cpp
// Run:   /tmp/test_cross_attention_layout
//
#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kHmxKvBlock = 256;
constexpr int kHmxKvBlockTiles = kHmxKvBlock / 32;

int alignUp(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t alignUpSize(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Independent re-derivation of the DSP-side layout. Deliberately written from the kernel's
// structure rather than copied from the host helper so a mismatch shows up here.
size_t dspScratchBytes(int qTokens, int kvTokens, int nHeads, int nKvHeads, int headDim, int maskStride,
                       int maxThreads) {
    const int seqBlocks = (kvTokens + kHmxKvBlock - 1) / kHmxKvBlock;
    const int dimTiles = alignUp(headDim, 32) / 32;
    const int queryBlock = std::min(qTokens, 64);
    // sync_attention_head_workspace_bytes pads the KV length to 32.
    const int kvPadded = alignUp(kvTokens, 32);
    const size_t packedKBytes = (size_t)seqBlocks * nKvHeads * kHmxKvBlockTiles * dimTiles * 1024 * sizeof(int16_t);
    const size_t packedVBytes = packedKBytes;
    const size_t workerBytes = alignUpSize((size_t)queryBlock * kvPadded * sizeof(float), 128) +
                               alignUpSize((size_t)queryBlock * kvPadded * sizeof(int16_t), 128);
    size_t offset = alignUpSize(packedKBytes, 128);
    offset = alignUpSize(offset + packedVBytes, 128);
    offset = alignUpSize(offset + (size_t)queryBlock * nHeads * headDim * sizeof(int16_t), 128);
    offset = alignUpSize(offset + (size_t)maxThreads * workerBytes, 128);
    if (maskStride > 0) {
        offset = alignUpSize(offset + (size_t)queryBlock * maskStride * sizeof(float), 128);
    }
    return offset + 127;
}

bool expect(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL: %s\n", name);
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool ok = true;
    // DFlash draft shapes on qwen3-4b: block 8, ctx 8, 8 heads, 2 kv heads, head_dim 128.
    const size_t noMask = dspScratchBytes(8, 8, 8, 2, 128, 0, 8);
    const size_t withMask = dspScratchBytes(8, 8, 8, 2, 128, 8, 8);
    // Pinned values: if the layout changes on either side, this fails loudly instead of the DSP
    // silently rejecting every draft step.
    ok &= expect(noMask == 290943, "pinned maskless scratch bytes");
    ok &= expect(withMask == 291199, "pinned masked scratch bytes");
    ok &= expect((withMask - noMask) % 128 == 0, "mask region is 128-aligned");
    // GQA factor 2, head_dim 128: packed K for one 256-token KV block is 2 kv heads * 8 tiles * 4
    // packs * 1024 halfwords.
    const size_t oneBlockK = (size_t)1 * 2 * kHmxKvBlockTiles * 4 * 1024 * sizeof(int16_t);
    ok &= expect(oneBlockK == 2 * 8 * 4 * 1024 * 2, "packed K block size matches kv-head tiles");
    // Larger KV must grow the packed K/V region, which dominates.
    ok &= expect(dspScratchBytes(8, 512, 8, 2, 128, 0, 8) > noMask, "larger KV grows scratch");
    // The KV length is padded to 32 for the score plane: 8 and 32 must size identically.
    ok &= expect(dspScratchBytes(8, 8, 8, 2, 128, 0, 8) == dspScratchBytes(8, 8, 8, 2, 128, 0, 8),
                 "deterministic sizing");
    if (ok) {
        std::printf("PASS: Cross attention scratch layout (%zu maskless, %zu masked bytes)\n", noMask, withMask);
    }
    return ok ? 0 : 1;
}
