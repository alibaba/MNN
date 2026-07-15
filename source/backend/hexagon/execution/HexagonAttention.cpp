#include "HexagonRuntime.hpp"
#include "HexagonAttention.hpp"
#include "backend/hexagon/backend/HexagonBackend.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/hexagon/htp-ops-lib/include/htp_command.h"
#include <algorithm>
#include <string.h>
#define HEXAGON_KV_PAGE_SIZE 256
#define HEXAGON_ATTN_FIXED_WORKSPACE_KV 2048
#define HEXAGON_ATTN_PREFILL_SEGMENT_Q 64
namespace MNN {

static constexpr int kFlashAttnPageTableInputIndex = 4;

static int groupedCausalWorkspaceRows(int qoLen, int nHeads, int nKvHeads, int headDim, bool hasExplicitMask) {
    if (hasExplicitMask || qoLen <= 8 || headDim % 64 != 0 ||
        nKvHeads <= 0 || nHeads % nKvHeads != 0) {
        return 0;
    }
    const int gqaFactor = nHeads / nKvHeads;
    if (gqaFactor <= 1 || gqaFactor > 4) {
        return 0;
    }
    const int qRows = std::min(qoLen, 64);
    return gqaFactor * qRows;
}

struct FlashAttnParam {
    int qo_len;
    int seq_current;
    int seq_add;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    float scale;
    int mask_stride;
    int max_kv_len;
    int page_count;
    int page_size;
    int page_table_capacity;
    int value_c4;
};

struct FlashAttentionBlockParam {
    int batch;
    int heads;
    int tokens;
    int chunk;
    int head_dim;
    float scale;
};

static int flashAttentionBlockAlignUp(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static size_t flashAttentionBlockAlignUpSize(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t flashAttentionBlockWorkspaceBytes(int batch, int heads, int chunk, int headDim) {
    const int chunkPadded = flashAttentionBlockAlignUp(chunk, 32);
    if (chunkPadded > 32 || headDim != 64) {
        return 0;
    }
    constexpr int kHmxKvBlock = 256;
    constexpr int kHmxKvBlockTiles = kHmxKvBlock / 32;
    const int seqBlocks = (chunkPadded + kHmxKvBlock - 1) / kHmxKvBlock;
    const int kIcP = flashAttentionBlockAlignUp(headDim, 32) / 32;
    const int vOcP = flashAttentionBlockAlignUp(headDim, 32) / 32;
    const size_t packedKBytes = (size_t)seqBlocks * heads * kHmxKvBlockTiles * kIcP * 1024 * sizeof(int16_t);
    const size_t packedVBytes = (size_t)seqBlocks * heads * vOcP * kHmxKvBlockTiles * 1024 * sizeof(int16_t);
    size_t headWorkspaceBytes = 0;
    headWorkspaceBytes = flashAttentionBlockAlignUpSize(headWorkspaceBytes + 256 * 32 * sizeof(float), 128);
    headWorkspaceBytes = flashAttentionBlockAlignUpSize(headWorkspaceBytes + 256 * 32 * sizeof(int16_t), 128);
    headWorkspaceBytes = flashAttentionBlockAlignUpSize(headWorkspaceBytes + 256 * 64 * sizeof(float), 128);
    headWorkspaceBytes = flashAttentionBlockAlignUpSize(headWorkspaceBytes + 256 * sizeof(float), 128);
    headWorkspaceBytes = flashAttentionBlockAlignUpSize(headWorkspaceBytes + 256 * sizeof(float), 128);

    size_t offset = 0;
    offset = flashAttentionBlockAlignUpSize(offset + (size_t)batch * packedKBytes, 128);
    offset = flashAttentionBlockAlignUpSize(offset + (size_t)batch * packedVBytes, 128);
    offset = flashAttentionBlockAlignUpSize(offset + (size_t)batch * heads * headWorkspaceBytes, 128);
    return offset;
}

HexagonAttention::HexagonAttention(Backend *backend, const Op *op) : HexagonExecution(backend) {
    auto param = op->main_as_AttentionParam();
    if (param) {
        mAttnScale = param->attnScale();
    }
    MNN::KVCacheManager::KVCacheConfig kvconfig;
    mKVCacheManager.reset(new HexagonKVCacheManager(backend, kvconfig));
    mMeta = (KVMeta*)(backend->getMetaPtr());
}

HexagonAttention::~HexagonAttention() {
}

bool HexagonAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new HexagonAttention(bn, op);
    if (bn->getMetaPtr() == mMeta && mMeta != nullptr) {
        exe->mKVCacheManager = mKVCacheManager;
    }
    *dst = exe;
    return true;
}

ErrorCode HexagonAttention::ensurePageTableCapacity(int pageCount) {
    if (pageCount <= mPageTableCapacity && mPageTable != nullptr) {
        return NO_ERROR;
    }
    int capacity = std::max(1, mPageTableCapacity);
    while (capacity < pageCount) {
        capacity *= 2;
    }
    if (capacity < 8) {
        capacity = 8;
    }
    if (mPageTable) {
        backend()->onReleaseBuffer(mPageTable.get(), Backend::STATIC);
    }
    mPageTable.reset(Tensor::createDevice<int32_t>({capacity * 4}));
    if (!backend()->onAcquireBuffer(mPageTable.get(), Backend::STATIC)) {
        mPageTable.reset();
        mPageTableCapacity = 0;
        mSyncedPageGeneration = (uint64_t)-1;
        return OUT_OF_MEMORY;
    }
    ::memset(HexagonBackend::getPtr(mPageTable.get()), 0, (size_t)capacity * 4 * sizeof(int32_t));
    mPageTableCapacity = capacity;
    mSyncedPageGeneration = (uint64_t)-1;
    return NO_ERROR;
}

void HexagonAttention::updatePageTable() {
    if (mPageTable == nullptr) {
        return;
    }
    auto pageTable = reinterpret_cast<int32_t*>(HexagonBackend::getPtr(mPageTable.get()));
    const int pageCount = mKVCacheManager->pageCount();
    const auto& keyPages = mKVCacheManager->keyPages();
    const auto& valuePages = mKVCacheManager->valuePages();
    for (int i = 0; i < pageCount; ++i) {
        auto keyDev = HexagonBackend::getDevicePtr(keyPages[i].get());
        auto valueDev = HexagonBackend::getDevicePtr(valuePages[i].get());
        pageTable[4 * i + 0] = keyDev.first;
        pageTable[4 * i + 1] = keyDev.second;
        pageTable[4 * i + 2] = valueDev.first;
        pageTable[4 * i + 3] = valueDev.second;
    }
    static_cast<HexagonBackend*>(backend())->markHostInput(mPageTable.get());
    mSyncedPageGeneration = mKVCacheManager->pageGeneration();
}

ErrorCode HexagonAttention::onBuildCmd(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                       std::vector<HexagonCommand>& dst) {
    const bool streamingState = inputs.size() == 7 && outputs.size() == 3;
    if (mMeta != nullptr && streamingState) {
        return NOT_SUPPORT;
    }
    if (streamingState) {
        auto Q = inputs[0];
        auto K = inputs[1];
        auto V = inputs[2];
        if (Q == nullptr || K == nullptr || V == nullptr ||
            inputs[4] == nullptr || inputs[5] == nullptr || inputs[6] == nullptr) {
            return INPUT_DATA_ERROR;
        }
        if (Q->dimensions() != 4 || K->dimensions() != 4 || V->dimensions() != 4) {
            return NOT_SUPPORT;
        }
        for (auto output : outputs) {
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            if (HexagonBackend::getBytes(output) != 2) {
                return NOT_SUPPORT;
            }
        }
        for (int i : {0, 1, 2, 4, 5, 6}) {
            if (HexagonBackend::getBytes(inputs[i]) != 2) {
                return NOT_SUPPORT;
            }
        }
        const int batch = Q->length(0);
        const int heads = Q->length(1);
        const int tokens = Q->length(2);
        const int headDim = Q->length(3);
        const int chunk = K->length(1);
        if (batch != K->length(0) || batch != V->length(0) ||
            heads != K->length(2) || heads != V->length(2) ||
            headDim != K->length(3) || headDim != V->length(3) ||
            chunk != V->length(1)) {
            return NOT_SUPPORT;
        }
        const float scale = (mAttnScale == 0.0f) ? (1.0f / sqrt(headDim)) : mAttnScale;
        FlashAttentionBlockParam params = {batch, heads, tokens, chunk, headDim, scale};

        std::vector<Tensor*> commandOutputs;
        commandOutputs.reserve(outputs.size() + 1);
        std::vector<std::pair<int, int>> outputFds;
        outputFds.reserve(outputs.size() + 1);
        for (auto output : outputs) {
            outputFds.emplace_back(HexagonBackend::getDevicePtr(output));
            commandOutputs.emplace_back(output);
        }
        const size_t workspaceBytes = flashAttentionBlockWorkspaceBytes(batch, heads, chunk, headDim);
        if (workspaceBytes > 0) {
            mWorkspace.reset(Tensor::createDevice<int8_t>({(int)workspaceBytes}));
            if (!backend()->onAcquireBuffer(mWorkspace.get(), Backend::DYNAMIC)) {
                mWorkspace.reset();
                return OUT_OF_MEMORY;
            }
            outputFds.emplace_back(HexagonBackend::getDevicePtr(mWorkspace.get()));
            commandOutputs.emplace_back(mWorkspace.get());
        }

        std::vector<std::pair<int, int>> inputFds;
        inputFds.reserve(inputs.size());
        std::vector<Tensor*> commandInputs;
        commandInputs.reserve(inputs.size());
        for (auto input : inputs) {
            inputFds.emplace_back(HexagonBackend::getDevicePtr(input));
            commandInputs.emplace_back(input);
        }
        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_FLASH_ATTENTION_BLOCK,
                         &params, sizeof(params), inputFds, outputFds, commandInputs, commandOutputs);
        if (mWorkspace) {
            backend()->onReleaseBuffer(mWorkspace.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }
    if (inputs.size() < 3 || outputs.empty()) {
        return NOT_SUPPORT;
    }
    TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];
    if (Q == nullptr || K == nullptr || V == nullptr ||
        Q->dimensions() != 4 || K->dimensions() != 4 || V->dimensions() < 2) {
        return NOT_SUPPORT;
    }
    if (TensorUtils::getDescribe(Q)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(K)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(V)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return NOT_SUPPORT;
    }
    if (HexagonBackend::getBytes(Q) != 2 || HexagonBackend::getBytes(K) != 2 ||
        HexagonBackend::getBytes(V) != 2 || HexagonBackend::getBytes(outputs[0]) != 2) {
        return NOT_SUPPORT;
    }
    if (inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->dimensions() > 2 &&
        HexagonBackend::getBytes(inputs[3]) != 2) {
        return NOT_SUPPORT;
    }

    int qo_len = Q->length(1);
    int head_dim = Q->length(3);
    int insert_len = K->length(1);

    int M = qo_len;
    int K_dim = head_dim;

    int n_heads = Q->length(2);
    int n_kv_heads = K->length(2);
    if (V->length(0) != insert_len || V->length(1) != n_kv_heads * head_dim) {
        return NOT_SUPPORT;
    }

    mKVCacheManager->onResize(n_kv_heads, head_dim);

    if (mMeta != nullptr) {
        if (mMeta->previous == mMeta->remove) {
            mKVCacheManager->onClear();
            mMeta->block = HEXAGON_KV_PAGE_SIZE;
            mKVCacheManager->onAlloc(mMeta, insert_len);
        } else {
            mMeta->block = HEXAGON_KV_PAGE_SIZE;
            mKVCacheManager->onRealloc(mMeta);
        }
    } else {
        mKVCacheManager->onClear();
        mKVCacheManager->onAlloc(nullptr, insert_len);
    }
    if (!mKVCacheManager->valid()) {
        return OUT_OF_MEMORY;
    }
    int max_kv_len = mKVCacheManager->maxLength();
    int page_count = mKVCacheManager->pageCount();
    int page_size = mKVCacheManager->pageSize();
    mMaxKVLen = max_kv_len;
    auto tableCode = ensurePageTableCapacity(page_count);
    if (tableCode != NO_ERROR) {
        return tableCode;
    }
    if (mSyncedPageGeneration != mKVCacheManager->pageGeneration()) {
        updatePageTable();
    }

    const bool hasExplicitMask = (inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->dimensions() > 2);
    int workspaceRows = M;
    if (!hasExplicitMask && M > HEXAGON_ATTN_PREFILL_SEGMENT_Q) {
        workspaceRows = HEXAGON_ATTN_PREFILL_SEGMENT_Q;
    }
    if (M <= 8 && head_dim % 64 == 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0) {
        int gqaFactor = n_heads / n_kv_heads;
        int groupedRows = gqaFactor * M;
        if (groupedRows > workspaceRows) {
            workspaceRows = groupedRows;
        }
    }
    int groupedCausalRows = groupedCausalWorkspaceRows(M, n_heads, n_kv_heads, head_dim, hasExplicitMask);
    if (groupedCausalRows > workspaceRows) {
        workspaceRows = groupedCausalRows;
    }
    const bool useFixedPageWorkspace = (page_size > 0) && (head_dim % 64 == 0);
    const int workspaceKvLen = useFixedPageWorkspace ? HEXAGON_ATTN_FIXED_WORKSPACE_KV : max_kv_len;
    const int workspaceKvLenPadded = (workspaceKvLen + 31) / 32 * 32;
    const int head_dim_padded = (head_dim + 31) / 32 * 32;
    size_t total_bytes = 0;
    total_bytes += workspaceRows * workspaceKvLenPadded * sizeof(float);
    total_bytes = (total_bytes + 127) & ~127;
    total_bytes += workspaceRows * workspaceKvLenPadded * sizeof(int16_t);
    total_bytes = (total_bytes + 127) & ~127;
    total_bytes += workspaceRows * head_dim_padded * sizeof(int16_t);
    total_bytes = (total_bytes + 127) & ~127;
    if (useFixedPageWorkspace) {
        total_bytes += workspaceRows * head_dim_padded * sizeof(int16_t);
        total_bytes = (total_bytes + 127) & ~127;
        total_bytes += 3 * workspaceRows * sizeof(float);
        total_bytes = (total_bytes + 127) & ~127;
    }
    int maxThreads = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info().maxThreads;
    size_t maskBytes = 0;
    if (inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->dimensions() > 2) {
        int maskLen = inputs[3]->length(3);
        if (useFixedPageWorkspace && maskLen > HEXAGON_ATTN_FIXED_WORKSPACE_KV) {
            maskLen = HEXAGON_ATTN_FIXED_WORKSPACE_KV;
        }
        maskBytes = (size_t)qo_len * maskLen * sizeof(float);
        maskBytes = (maskBytes + 127) & ~((size_t)127);
    }
    mWorkspace.reset(Tensor::createDevice<int8_t>({(int)(total_bytes * maxThreads + maskBytes)}));
    bool res = backend()->onAcquireBuffer(mWorkspace.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }

    float scale = (mAttnScale == 0.0f) ? (1.0f / sqrt(head_dim)) : mAttnScale;

    auto kDev = HexagonBackend::getDevicePtr(K);
    auto vDev = HexagonBackend::getDevicePtr(V);
    auto qDev = HexagonBackend::getDevicePtr(Q);
    auto oDev = HexagonBackend::getDevicePtr(outputs[0]);
    auto workspaceDev = HexagonBackend::getDevicePtr(mWorkspace.get());

    int seq_current = 0;
    int seq_add = insert_len;
    if (mMeta != nullptr) {
        seq_current = mMeta->previous - mMeta->remove;
        seq_add = mMeta->add;
    }
    Tensor* maskTensor = nullptr;
    int mask_stride = 0;
    if (inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->dimensions() > 2) {
        maskTensor = inputs[3];
        mask_stride = inputs[3]->length(3);
        mUseGeneratedCausalMask = false;
    } else if (qo_len == 1) {
        mUseGeneratedCausalMask = false;
        mask_stride = 0;
    } else {
        mUseGeneratedCausalMask = true;
        maskTensor = nullptr;
        mask_stride = -1;
    }
    auto maskDev = maskTensor != nullptr ? HexagonBackend::getDevicePtr(maskTensor) : std::make_pair(-1, 0);
    int maskFd = maskDev.first;
    int maskOffset = maskDev.second;

    FlashAttnParam params = {qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim, scale, mask_stride,
                             max_kv_len, page_count, page_size, mPageTableCapacity, 1};

    auto pageTableDev = HexagonBackend::getDevicePtr(mPageTable.get());
    std::vector<std::pair<int, int>> attnInputFds = {qDev, kDev, vDev, {maskFd, maskOffset}, pageTableDev};
    std::vector<Tensor*> attnInputs = {Q, K, V, maskTensor, mPageTable.get()};
    std::vector<std::pair<int, int>> attnOutputFds = {oDev, workspaceDev};

    std::vector<Tensor*> attnOutputs = {outputs[0], mWorkspace.get()};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_FLASH_ATTN, &params, sizeof(params),
                     attnInputFds, attnOutputFds, attnInputs, attnOutputs);

    backend()->onReleaseBuffer(mWorkspace.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode HexagonAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (!mValid) {
        return NOT_SUPPORT;
    }
    if (mMeta == nullptr) {
        return HexagonExecution::onExecute(inputs, outputs);
    }
    const int seq_current = (int)mMeta->previous - (int)mMeta->remove;
    const int seq_add = (int)mMeta->add;
    mKVCacheManager->onRealloc(mMeta);
    if (!mKVCacheManager->valid()) {
        return OUT_OF_MEMORY;
    }
    const int oldPageTableCapacity = mPageTableCapacity;
    auto tableCode = ensurePageTableCapacity(mKVCacheManager->pageCount());
    if (tableCode != NO_ERROR) {
        return tableCode;
    }
    if (mPageTableCapacity != oldPageTableCapacity) {
        for (auto& cmd : mCmd) {
            cmd.setInputTensor(mPageTable.get(), kFlashAttnPageTableInputIndex);
        }
    }
    if (mSyncedPageGeneration != mKVCacheManager->pageGeneration()) {
        updatePageTable();
    }
    for (auto& cmd : mCmd) {
        auto params = static_cast<FlashAttnParam*>(cmd.getParam());
        if (params != nullptr) {
            params->seq_current = seq_current;
            params->seq_add = seq_add;
            if (mUseGeneratedCausalMask) {
                params->mask_stride = -1;
            }
            params->max_kv_len = mKVCacheManager->maxLength();
            params->page_count = mKVCacheManager->pageCount();
            params->page_size = mKVCacheManager->pageSize();
            params->page_table_capacity = mPageTableCapacity;
        }
        cmd.execute(true);
    }
    return NO_ERROR;
}
} // namespace MNN
