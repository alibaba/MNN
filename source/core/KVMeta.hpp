//
//  KVMeta.hpp
//  MNN
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef KVMeta_hpp
#define KVMeta_hpp

#include <vector>
#include <string>
#include <algorithm>
#include <limits>

namespace MNN {

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead,
        PendingSnapshot,
        SnapshotFailed
    } file_operation;
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    size_t prefix_session_id = 0;
    std::vector<int> reserveHost;
    // Attention scaling override (gemma4 uses 1.0 instead of 1/sqrt(head_dim))
    float attn_scale = 0.0f; // 0 means use default 1/sqrt(head_dim)
    // Verify block length for this forward; recurrent-state ops must defer their state update
    int spec_block = 0; // 0 means an ordinary prefill/decode
    size_t capacity = 0;
    int computeReverseSize() const {
        int sum = 0;
        for (int i=0; i<n_reserve; ++i) {
            int reserveUnit = reserve[2*i+1];
            if (reserveUnit <= 0) {
                // Invalid
                return -1;
            }
            sum += reserveUnit;
        }
        return sum;
    }
    void sync() {
        int revertNumber = 0;
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];
        }
        previous = previous - remove + add + revertNumber;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
    // Per-request reservation target, restored on exit so nested chunk forwards keep the outer reservation.
    // Decode headroom is bounded by the prompt size to avoid reserving the whole generation limit for short requests.
    struct CapacityScope {
        CapacityScope(KVMeta* meta, size_t promptLength, int maxTokens)
            : mMeta(meta), mPreviousCapacity(meta->capacity) {
            if (meta->remove > meta->previous || maxTokens < 0 || meta->seqlen_in_disk < 0) {
                return;
            }
            const int reverseSize = meta->computeReverseSize();
            if (reverseSize < 0) {
                return;
            }
            const size_t decodeHeadroom = std::min(promptLength, static_cast<size_t>(maxTokens));
            const size_t limit = std::numeric_limits<int>::max() - 128;
            size_t capacity = meta->previous - meta->remove;
            for (size_t length : {static_cast<size_t>(reverseSize), static_cast<size_t>(meta->seqlen_in_disk),
                                  promptLength, decodeHeadroom}) {
                if (capacity > limit || length > limit - capacity) {
                    return;
                }
                capacity += length;
            }
            meta->capacity = std::max(mPreviousCapacity, capacity);
        }
        ~CapacityScope() {
            mMeta->capacity = mPreviousCapacity;
        }
    private:
        KVMeta* mMeta;
        size_t mPreviousCapacity;
    };
};

} // namespace MNN
#endif // KVMeta_hpp
