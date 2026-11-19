// Copyright © 2026, Alibaba Group Holding Limited
#include <riscv_vector.h>
#include <stddef.h>
#include "MNNRvvPrefill.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPULinearAttention.hpp"
#include "core/Concurrency.h"
#include "core/MNNFileUtils.h"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <climits>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef MNN_USE_SPACEMIT_IME2
#include "backend/cpu/riscv/spacemit_ime2/MNNSpacemitIme2Half.hpp"
extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
#endif
#endif

// Match the RISC-V scalar fallback's explicit FMA contract: round the rank-one
// product before fusing decay, and fuse v - decay * outK before multiplying beta.

void MNNRankOneUpdate_RVV(float* S, const float* k, const float* delta, size_t dk, size_t dv) {
    for (size_t i = 0; i < dk; ++i) {
        const float kValue = k[i];
        float* row = S + i * dv;
        size_t j = 0;
        while (j < dv) {
            const size_t vl = __riscv_vsetvl_e32m8(dv - j);
            vfloat32m8_t state = __riscv_vle32_v_f32m8(row + j, vl);
            const vfloat32m8_t deltaValue = __riscv_vle32_v_f32m8(delta + j, vl);
            state = __riscv_vfmacc_vf_f32m8(state, kValue, deltaValue, vl);
            __riscv_vse32_v_f32m8(row + j, state, vl);
            j += vl;
        }
    }
}

void MNNDualMatVec_RVV(const float* S, const float* k, const float* q, float* out_k, float* out_q, size_t dk,
                       size_t dv) {
    size_t j = 0;
    while (j < dv) {
        const size_t vl = __riscv_vsetvl_e32m8(dv - j);
        __riscv_vse32_v_f32m8(out_k + j, __riscv_vfmv_v_f_f32m8(0.0f, vl), vl);
        __riscv_vse32_v_f32m8(out_q + j, __riscv_vfmv_v_f_f32m8(0.0f, vl), vl);
        j += vl;
    }

    for (size_t i = 0; i < dk; ++i) {
        const float kValue = k[i];
        const float qValue = q[i];
        const float* row = S + i * dv;
        j = 0;
        while (j < dv) {
            const size_t vl = __riscv_vsetvl_e32m8(dv - j);
            const vfloat32m8_t state = __riscv_vle32_v_f32m8(row + j, vl);
            vfloat32m8_t outK = __riscv_vle32_v_f32m8(out_k + j, vl);
            vfloat32m8_t outQ = __riscv_vle32_v_f32m8(out_q + j, vl);
            outK = __riscv_vfmacc_vf_f32m8(outK, kValue, state, vl);
            outQ = __riscv_vfmacc_vf_f32m8(outQ, qValue, state, vl);
            __riscv_vse32_v_f32m8(out_k + j, outK, vl);
            __riscv_vse32_v_f32m8(out_q + j, outQ, vl);
            j += vl;
        }
    }
}

void MNNDecayRankOneUpdate_RVV(float* S, const float* k, const float* delta, float decay, size_t dk, size_t dv) {
    for (size_t i = 0; i < dk; ++i) {
        const float kValue = k[i];
        float* row = S + i * dv;
        size_t j = 0;
        while (j < dv) {
            const size_t vl = __riscv_vsetvl_e32m8(dv - j);
            vfloat32m8_t state = __riscv_vle32_v_f32m8(row + j, vl);
            const vfloat32m8_t deltaValue = __riscv_vle32_v_f32m8(delta + j, vl);
            const vfloat32m8_t update = __riscv_vfmul_vf_f32m8(deltaValue, kValue, vl);
            state = __riscv_vfmacc_vf_f32m8(update, decay, state, vl);
            __riscv_vse32_v_f32m8(row + j, state, vl);
            j += vl;
        }
    }
}

void MNNFusedGatedDelta_RVV(float* S, const float* k, const float* q, const float* v, float* out, float decay,
                            float beta, float kq, size_t dk, size_t dv) {
    size_t j = 0;
    while (j < dv) {
        const size_t vl = __riscv_vsetvl_e32m8(dv - j);
        vfloat32m8_t outK = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t outQ = __riscv_vfmv_v_f_f32m8(0.0f, vl);

        for (size_t i = 0; i < dk; ++i) {
            const vfloat32m8_t state = __riscv_vle32_v_f32m8(S + i * dv + j, vl);
            outK = __riscv_vfmacc_vf_f32m8(outK, k[i], state, vl);
            outQ = __riscv_vfmacc_vf_f32m8(outQ, q[i], state, vl);
        }

        const vfloat32m8_t value = __riscv_vle32_v_f32m8(v + j, vl);
        vfloat32m8_t delta = __riscv_vfnmsac_vf_f32m8(value, decay, outK, vl);
        delta = __riscv_vfmul_vf_f32m8(delta, beta, vl);

        vfloat32m8_t outValue = __riscv_vfmul_vf_f32m8(delta, kq, vl);
        outValue = __riscv_vfmacc_vf_f32m8(outValue, decay, outQ, vl);
        __riscv_vse32_v_f32m8(out + j, outValue, vl);

        for (size_t i = 0; i < dk; ++i) {
            float* row = S + i * dv + j;
            vfloat32m8_t state = __riscv_vle32_v_f32m8(row, vl);
            const vfloat32m8_t update = __riscv_vfmul_vf_f32m8(delta, k[i], vl);
            state = __riscv_vfmacc_vf_f32m8(update, decay, state, vl);
            __riscv_vse32_v_f32m8(row, state, vl);
        }
        j += vl;
    }
}

void MNNGatedDeltaChunk_RVV(float* state, const float* q, const float* k, float* v, float* projection, float* output,
                            const float* decay, const float* beta, size_t outputStride, bool projected) {
    constexpr size_t count = MNN::kRvvGatedDeltaChunk;
    constexpr size_t dk = MNN::kRvvGatedDeltaHeadDim, dv = MNN::kRvvGatedDeltaHeadDim;
    static_assert(count > 0 && count % 2 == 0 && dk % 4 == 0, "RVV processes two tokens and four state rows at a time");
    float prefix[count];
    float keyWeight[count][count];
    float queryWeight[count][count];
    float carry = 1.0f;
    for (size_t t = 0; t < count; ++t) {
        carry *= decay[t];
        prefix[t] = carry;
        float between = 1.0f;
        for (size_t j = t + 1; j > 0; --j) {
            const size_t i = j - 1;
            float kk = 0.0f;
            float qk = 0.0f;
            for (size_t d = 0; d < dk;) {
                const size_t vl = __riscv_vsetvl_e32m4(dk - d);
                const auto ki = __riscv_vle32_v_f32m4(k + i * dk + d, vl);
                const auto kt = __riscv_vle32_v_f32m4(k + t * dk + d, vl);
                const auto qt = __riscv_vle32_v_f32m4(q + t * dk + d, vl);
                const auto zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                kk += __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(ki, kt, vl), zero, vl));
                qk += __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(ki, qt, vl), zero, vl));
                d += vl;
            }
            keyWeight[t][i] = between * kk;
            queryWeight[t][i] = between * qk;
            between *= decay[i];
        }
    }

    // Project pairs of Q/K rows against the same state loads. The entry state is
    // read-only until the residuals for the whole chunk have been solved.
    for (size_t col = 0; col < dv;) {
        const size_t vl = __riscv_vsetvl_e32m4(dv - col);
        const auto zero = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t t = 0; !projected && t < count; t += 2) {
            auto k0 = zero;
            auto q0 = zero;
            auto k1 = zero;
            auto q1 = zero;
            const float* a = k + t * dk;
            const float* b = q + t * dk;
            for (size_t d = 0; d < dk; ++d) {
                const auto row = __riscv_vle32_v_f32m4(state + d * dv + col, vl);
                k0 = __riscv_vfmacc_vf_f32m4(k0, a[d], row, vl);
                q0 = __riscv_vfmacc_vf_f32m4(q0, b[d], row, vl);
                k1 = __riscv_vfmacc_vf_f32m4(k1, a[dk + d], row, vl);
                q1 = __riscv_vfmacc_vf_f32m4(q1, b[dk + d], row, vl);
            }
            __riscv_vse32_v_f32m4(projection + t * dv + col, k0, vl);
            __riscv_vse32_v_f32m4(projection + (t + 1) * dv + col, k1, vl);
            __riscv_vse32_v_f32m4(output + t * outputStride + col, q0, vl);
            __riscv_vse32_v_f32m4(output + (t + 1) * outputStride + col, q1, vl);
        }

        // Solve the lower-triangular recurrence over the residuals in-place.
        for (size_t t = 0; t < count; ++t) {
            auto residual = __riscv_vle32_v_f32m4(v + t * dv + col, vl);
            auto pred = __riscv_vle32_v_f32m4(projection + t * dv + col, vl);
            pred = __riscv_vfmul_vf_f32m4(pred, prefix[t], vl);
            auto result = __riscv_vle32_v_f32m4(output + t * outputStride + col, vl);
            result = __riscv_vfmul_vf_f32m4(result, prefix[t], vl);
            for (size_t j = 0; j < t; ++j) {
                const auto previous = __riscv_vle32_v_f32m4(v + j * dv + col, vl);
                pred = __riscv_vfmacc_vf_f32m4(pred, keyWeight[t][j], previous, vl);
                result = __riscv_vfmacc_vf_f32m4(result, queryWeight[t][j], previous, vl);
            }
            residual = __riscv_vfsub_vv_f32m4(residual, pred, vl);
            residual = __riscv_vfmul_vf_f32m4(residual, beta[t], vl);
            result = __riscv_vfmacc_vf_f32m4(result, queryWeight[t][t], residual, vl);
            __riscv_vse32_v_f32m4(v + t * dv + col, residual, vl);
            __riscv_vse32_v_f32m4(output + t * outputStride + col, result, vl);
        }
        col += vl;
    }

    if (projected)
        return;
    float suffix[count];
    carry = 1.0f;
    for (size_t j = count; j > 0; --j) {
        suffix[j - 1] = carry;
        carry *= decay[j - 1];
    }
    // One rank-chunk update per state tile amortizes stores across all 16 tokens.
    for (size_t col = 0; col < dv;) {
        const size_t vl = __riscv_vsetvl_e32m4(dv - col);
        for (size_t d = 0; d < dk; d += 4) {
            float* row = state + d * dv + col;
            auto s0 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(row, vl), prefix[count - 1], vl);
            auto s1 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(row + dv, vl), prefix[count - 1], vl);
            auto s2 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(row + 2 * dv, vl), prefix[count - 1], vl);
            auto s3 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(row + 3 * dv, vl), prefix[count - 1], vl);
            for (size_t t = 0; t < count; ++t) {
                auto residual = __riscv_vle32_v_f32m4(v + t * dv + col, vl);
                residual = __riscv_vfmul_vf_f32m4(residual, suffix[t], vl);
                const float* kt = k + t * dk + d;
                s0 = __riscv_vfmacc_vf_f32m4(s0, kt[0], residual, vl);
                s1 = __riscv_vfmacc_vf_f32m4(s1, kt[1], residual, vl);
                s2 = __riscv_vfmacc_vf_f32m4(s2, kt[2], residual, vl);
                s3 = __riscv_vfmacc_vf_f32m4(s3, kt[3], residual, vl);
            }
            __riscv_vse32_v_f32m4(row, s0, vl);
            __riscv_vse32_v_f32m4(row + dv, s1, vl);
            __riscv_vse32_v_f32m4(row + 2 * dv, s2, vl);
            __riscv_vse32_v_f32m4(row + 3 * dv, s3, vl);
        }
        col += vl;
    }
}

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace MNN {
namespace {
bool isC4(const Tensor* t) {
    return TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

bool supportsLinearInputs(const std::vector<Tensor*>& inputs, int64_t channels) {
    if (inputs.size() < 4 || inputs[3]->dimensions() < 3 ||
        inputs[0]->dimensions() < (isC4(inputs[0]) ? 2 : 3)) return false;
    const auto qkv = inputs[0];
    return qkv->length(0) > 0 && qkv->length(1) == channels && inputs[3]->length(2) > 0 &&
           (isC4(qkv) || qkv->length(2) > 0);
}

class RvvLinearAttention final : public Execution {
public:
    RvvLinearAttention(CPUBackend* cpu, std::shared_ptr<const LinearAttentionParamT> p, std::shared_ptr<StateCache> state)
        : Execution(cpu), mParam(std::move(p)), mState(std::move(state)) {}

    bool onClone(Backend* bn, const Op*, Execution** dst) override {
        if (!dst) {
            return true;
        }
        if (!bn || bn->type() != MNN_FORWARD_CPU) {
            return false;
        }
        auto cpu = static_cast<CPUBackend*>(bn);
        if (!cpu->functions()->supportRVV || cpu->functions()->bytes != 4 || cpu->functions()->pack != 4) {
            return false;
        }
        *dst = new RvvLinearAttention(cpu, mParam, mState);
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    bool useChunks() const {
        const size_t workers = std::min(mBatch * mParam->num_v_heads, static_cast<CPUBackend*>(backend())->threadNumber());
        const size_t panelFloats = workers * kRvvGatedDeltaChunk * 4 * kRvvGatedDeltaHeadDim;
        return mParam->use_qk_l2norm && mParam->head_k_dim == kRvvGatedDeltaHeadDim &&
               mParam->head_v_dim == kRvvGatedDeltaHeadDim && mLength >= kRvvGatedDeltaChunk &&
               size_t(mBatch) * mChannels * (mLength + mKernel - 1) >= panelFloats;
    }
    void convolve(const Tensor* qkv, const float* weight, float* conv, float* work);
    void gates(const Tensor* gate, const Tensor* beta, float* decay, float* betaValues);
    void runHeads(const float* conv, const float* beta, const float* decay, float* output, float* panels, Tensor* out);
    ErrorCode prefix(const std::string& path, bool load);
    std::shared_ptr<const LinearAttentionParamT> mParam;
    std::shared_ptr<StateCache> mState;
    MemChunk mWork;
    size_t mConvBytes = 0, mRecurrentBytes = 0;
    int mBatch = 0, mLength = 0, mChannels = 0, mKernel = 0;
};

// Withdraw .v then .k, and publish the staged pair in the same order. A public .k and .v always match.
ErrorCode RvvLinearAttention::prefix(const std::string& path, bool load) {
    const auto slash = path.rfind('/');
    const auto directory = slash == std::string::npos ? "." : path.substr(0, slash + 1);
    struct DirectoryLock {
        int fd;
        ~DirectoryLock() { if (fd >= 0) ::close(fd); }
    } lock{::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
    if (lock.fd < 0 || ::flock(lock.fd, load ? LOCK_SH : LOCK_EX) != 0) {
        return FILE_OPEN_FAILED;
    }
    const std::string files[] = {path + ".k", path + ".v"};
    const std::string backup[] = {files[0] + ".prev", files[1] + ".prev"};
    const size_t sizes[] = {mConvBytes, mRecurrentBytes};
    const Tensor* states[] = {mState->mConvState.get(), mState->mRecurrentState.get()};
    auto missing = [](const std::string& name) {
        return ::access(name.c_str(), F_OK) != 0 && errno == ENOENT;
    };
    if (load) {
        const bool keyWithdrawn = missing(files[0]);
        const bool useBackup[] = {keyWithdrawn, keyWithdrawn || missing(files[1])};
        std::shared_ptr<void> mappings[2];
        for (int i = 0; i < 2; ++i) {
            const auto& name = useBackup[i] ? backup[i] : files[i];
            auto fd = MNNOpenFile(name.c_str(), MNN_FILE_READ);
            if (fd == INVALID_FILE) return FILE_OPEN_FAILED;
            const auto bytes = sizes[i];
            void* map = MNNGetFileSize(fd) == bytes ? MNNMmapFile(fd, bytes, true) : nullptr;
            MNNCloseFile(fd);
            if (!map) return INPUT_DATA_ERROR;
            mappings[i].reset(map, [bytes](void* p) { MNNUnmapFile(p, bytes); });
        }
        for (int i = 0; i < 2; ++i) {
            ::memcpy(states[i]->host<void>(), mappings[i].get(), sizes[i]);
        }
        return NO_ERROR;
    }

    auto removeIfPresent = [](const std::string& name) {
        return ::unlink(name.c_str()) == 0 || errno == ENOENT;
    };
    auto moveIfPresent = [](const std::string& from, const std::string& to) {
        return ::rename(from.c_str(), to.c_str()) == 0 || errno == ENOENT;
    };
    auto restore = [&]() {
        // Reverse publication one rename at a time, so readers can also recover during rollback.
        if (missing(files[0]) && (!removeIfPresent(files[1]) || !moveIfPresent(backup[0], files[0]))) return false;
        return !missing(files[1]) || moveIfPresent(backup[1], files[1]);
    };
    if ((!missing(backup[0]) || !missing(backup[1])) && !restore()) return FILE_CREATE_FAILED;
    for (const auto& file : files) {
        struct stat st;
        if (::lstat(file.c_str(), &st) == 0 ? !S_ISREG(st.st_mode) : errno != ENOENT) return FILE_CREATE_FAILED;
    }
    if (!removeIfPresent(backup[0]) || !removeIfPresent(backup[1])) return FILE_CREATE_FAILED;

    const std::string temp[] = {files[0] + ".tmp", files[1] + ".tmp"};
    struct RemoveTemps {
        const std::string* paths;
        ~RemoveTemps() { for (int i = 0; i < 2; ++i) ::unlink(paths[i].c_str()); }
    } cleanup{temp};
    for (int i = 0; i < 2; ++i) {
        if (!removeIfPresent(temp[i])) return FILE_CREATE_FAILED;
        int fd = ::open(temp[i].c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (fd < 0) return FILE_CREATE_FAILED;
        const bool written = MNNWriteFile(fd, states[i]->host<void>(), sizes[i]) == sizes[i];
        if (MNNCloseFile(fd) != NO_ERROR || !written) return FILE_CREATE_FAILED;
    }
    if (!moveIfPresent(files[1], backup[1])) return FILE_CREATE_FAILED;
    if (!moveIfPresent(files[0], backup[0]) || ::rename(temp[1].c_str(), files[1].c_str()) != 0 ||
        ::rename(temp[0].c_str(), files[0].c_str()) != 0) {
        restore();
        return FILE_CREATE_FAILED;
    }
    removeIfPresent(backup[0]);
    removeIfPresent(backup[1]);
    return NO_ERROR;
}

ErrorCode RvvLinearAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const int heads = mParam->num_v_heads, dk = mParam->head_k_dim, dv = mParam->head_v_dim;
    if (!supportsLinearInputs(inputs, 2LL * mParam->num_k_heads * dk + int64_t(heads) * dv)) {
        return INPUT_DATA_ERROR;
    }
    auto qkv = inputs[0];
    const bool packed = isC4(qkv);
    mBatch = packed ? 1 : qkv->length(0);
    mLength = packed ? qkv->length(0) : qkv->length(2);
    mChannels = qkv->length(1);
    mKernel = inputs[3]->length(2);
    const size_t tokens = size_t(mBatch) * mLength;
    const size_t convBytes = std::max<size_t>(1, size_t(mBatch) * mChannels * (mKernel - 1) * sizeof(float));
    const size_t recurrentBytes = size_t(mBatch) * heads * dk * dv * sizeof(float);
    auto allocState = [&](const std::shared_ptr<Tensor>& old, size_t bytes) -> std::shared_ptr<Tensor> {
        if (old && old->usize() >= bytes) {
            return old;
        }
        if (bytes > INT_MAX) {
            return nullptr;
        }
        std::shared_ptr<Tensor> tensor(Tensor::createDevice<uint8_t>({static_cast<int>(bytes)}));
        if (!backend()->onAcquireBuffer(tensor.get(), Backend::STATIC)) {
            return nullptr;
        }
        ::memset(tensor->host<void>(), 0, bytes);
        if (old) {
            ::memcpy(tensor->host<void>(), old->host<void>(), old->usize());
        }
        return tensor;
    };
    auto convState = allocState(mState->mConvState, convBytes);
    auto recurrentState = allocState(mState->mRecurrentState, recurrentBytes);
    if (!convState || !recurrentState) {
        return OUT_OF_MEMORY;
    }
    auto cpu = static_cast<CPUBackend*>(backend());
    const int threads = cpu->threadNumber();
    const int panelLength = useChunks() ? kRvvGatedDeltaChunk : 1;
    const size_t panels = size_t(std::min(mBatch * heads, threads)) * 2 * (dk + dv) * panelLength;
    const size_t work = std::max(panels, size_t(threads) * (mLength + mKernel - 1));
    const size_t floats = tokens * (mChannels + 2 * heads + (isC4(outputs[0]) ? heads * dv : 0)) + work;
    auto alloc = cpu->getBufferAllocator();
    mWork = alloc->alloc(floats * sizeof(float));
    if (mWork.invalid()) {
        return OUT_OF_MEMORY;
    }
    // Close this command's scratch interval in the CPU resize plan. The
    // MemChunk remains addressable during onExecute; all its workers join
    // before the next command can reuse this range (as in CPURaster).
    alloc->free(mWork);
    mConvBytes = convBytes;
    mRecurrentBytes = recurrentBytes;
    mState->mConvState = std::move(convState);
    mState->mRecurrentState = std::move(recurrentState);
    return NO_ERROR;
}

void RvvLinearAttention::convolve(const Tensor* qkv, const float* weight, float* conv, float* work) {
    const auto in = qkv->host<float>();
    auto history = mState->mConvState->host<float>();
    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();
    const int length = mLength, carry = mKernel - 1;
    const bool packed = isC4(qkv);
    const bool packedKernel = packed && mChannels % 4 == 0 && mKernel == 4 && length >= kRvvGatedDeltaChunk;
    MNN_CONCURRENCY_BEGIN(t, threads) {
        if (length == 1 && mKernel == 4) {
            // A single token is contiguous across channels in both supported layouts.
            // Give each worker adjacent output and history entries.
            const int count = mBatch * mChannels;
            const int begin = size_t(count) * t / threads;
            const int end = size_t(count) * (t + 1) / threads;
            for (int ch = begin; ch < end; ++ch) {
                auto state = history + size_t(ch) * 3;
                const auto w = weight + size_t(ch % mChannels) * 4;
                const float x = in[ch];
                const float sum = state[0] * w[0] + state[1] * w[1] + state[2] * w[2] + x * w[3];
                state[0] = state[1];
                state[1] = state[2];
                state[2] = x;
                conv[ch] = sum * (1.0f / (1.0f + expf(-sum)));
            }
        } else if (packedKernel) {
            for (int c = 4 * t; c < mChannels; c += 4 * threads) {
                MNNConv4SiLuC4_RVV(conv + size_t(c) * length, in + size_t(c) * length, history + c * 3,
                                   weight + c * 4, length);
            }
        } else {
            auto padded = work + size_t(t) * (length + carry);
            for (int ch = t; ch < mBatch * mChannels; ch += threads) {
                auto state = history + size_t(ch) * carry;
                auto out = conv + size_t(ch) * length;
                auto w = weight + (ch % mChannels) * mKernel;
                ::memcpy(padded, state, carry * sizeof(float));
                if (packed && length > 1) {
                    auto src = in + size_t(ch / 4) * length * 4 + ch % 4;
                    for (int i = 0; i < length; ++i) {
                        padded[carry + i] = src[i * 4];
                    }
                } else {
                    ::memcpy(padded + carry, in + size_t(ch) * length, length * sizeof(float));
                }
                ::memcpy(state, padded + length, carry * sizeof(float));
                for (int i = 0; i < length; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < mKernel; ++k) {
                        sum += padded[i + k] * w[k];
                    }
                    // Later windows start at i+1. Reuse the consumed input for
                    // the convolution sum so SiLU has disjoint input and output.
                    padded[i] = sum;
                }
                if (length == 1) {
                    out[0] = padded[0] * (1.0f / (1.0f + expf(-padded[0])));
                } else {
                    MNNSiLu(out, padded, length);
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
}

void RvvLinearAttention::gates(const Tensor* gate, const Tensor* beta, float* decay, float* betaValues) {
    auto cpu = static_cast<CPUBackend*>(backend());
    auto core = cpu->functions();
    const int heads = mParam->num_v_heads;
    const int count = mBatch * mLength * heads;
    int offsets[] = {mBatch * mLength, heads};
    auto unpack = [&](const Tensor* tensor, float* dst) {
        if (isC4(tensor)) {
            core->MNNUnpackCUnitTranspose(dst, tensor->host<float>(), offsets[0], heads, offsets);
        } else {
            ::memcpy(dst, tensor->host<float>(), count * sizeof(float));
        }
    };
    unpack(gate, decay);
    unpack(beta, betaValues);
    if (mParam->gate_fold) {
        auto unary = [&](UnaryOpOperation op, float* dst) {
            auto data = reinterpret_cast<int8_t*>(dst);
            core->MNNSelectUnaryFunctionForFloat(op, cpu->precisionMode())(data, data, count);
        };
        for (int token = 0; token < offsets[0]; ++token) {
            for (int h = 0; h < heads; ++h) {
                decay[token * heads + h] += mParam->gate_bias[h];
            }
        }
        unary(UnaryOpOperation_EXP, decay);
        for (int i = 0; i < count; ++i) {
            decay[i] += 1.0f;
        }
        unary(UnaryOpOperation_LOG, decay);
        for (int token = 0; token < offsets[0]; ++token) {
            for (int h = 0; h < heads; ++h) {
                decay[token * heads + h] *= mParam->gate_coef[h];
            }
        }
        unary(UnaryOpOperation_SIGMOID, betaValues);
    }
    if (mLength == 1) {
        for (int i = 0; i < count; ++i) {
            decay[i] = expf(decay[i]);
        }
    } else {
        float offset[] = {1.0f, 0.0f, 0.0f, 0.0f};
        MNNExp(decay, decay, offset, count);
    }
}

ErrorCode RvvLinearAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto meta = static_cast<KVMeta*>(backend()->getMetaPtr());
    const bool hasFile = meta && !meta->file_name.empty();
    const bool load = hasFile && meta->file_flag == KVMeta::PendingRead;
    const bool save = hasFile && meta->file_flag == KVMeta::PendingWrite;
    const bool reset = meta ? meta->previous == meta->remove : mLength > 1;
    if (!load && meta && meta->remove > 0 && (!reset || meta->n_reserve > 0)) {
        return NOT_SUPPORT;
    }
    std::string path;
    int layer = mState->mPrefixLayerIndex;
    if (load || save) {
        if (reset) {
            if (meta->layer_nums <= 0) {
                return INPUT_DATA_ERROR;
            }
            layer = meta->layer_index;
        }
        if (layer >= 0) {
            const auto& dir = backend()->getRuntime()->hint().prefixcacheDirPath;
            if (save && !MNNCreateDir(dir.c_str())) {
                return FILE_CREATE_FAILED;
            }
            path = MNNFilePathConcat(dir, meta->file_name) + "_" + std::to_string(layer);
        }
    }
    if (load && !path.empty()) {
        auto status = prefix(path, true);
        if (status != NO_ERROR) {
            return status;
        }
    } else if (!load && reset) {
        ::memset(mState->mConvState->host<void>(), 0, mState->mConvState->usize());
        ::memset(mState->mRecurrentState->host<void>(), 0, mState->mRecurrentState->usize());
    }
    const size_t tokens = size_t(mBatch) * mLength;
    const int heads = mParam->num_v_heads;
    auto conv = reinterpret_cast<float*>(mWork.ptr());
    auto decay = conv + tokens * mChannels;
    auto beta = decay + tokens * heads;
    auto work = beta + tokens * heads;
    auto out = outputs[0]->host<float>();
    if (isC4(outputs[0])) {
        out = work;
        work += tokens * heads * mParam->head_v_dim;
    }
    convolve(inputs[0], inputs[3]->host<float>(), conv, work);
    gates(inputs[1], inputs[2], decay, beta);
    runHeads(conv, beta, decay, out, work, outputs[0]);
    if (save && !path.empty()) {
        auto status = prefix(path, false);
        if (status != NO_ERROR) {
            return status;
        }
    }
    if ((load || save) && reset) {
        mState->mPrefixLayerIndex = layer;
        meta->layer_index = (layer + 1) % meta->layer_nums;
    }
    return NO_ERROR;
}

// Transpose 128 channels into 16 consecutive token rows. Four contiguous input rows
// fit in registers; the segmented store interleaves their channels in the panel.
static void packPanel(float* dst, const float* src, int length) {
    constexpr int dim = kRvvGatedDeltaHeadDim;
    for (int c = 0; c < dim; c += 4) {
        const float* row = src + size_t(c) * length;
        for (int i = 0; i < kRvvGatedDeltaChunk;) {
            size_t vl = __riscv_vsetvl_e32m1(kRvvGatedDeltaChunk - i);
            auto a = __riscv_vle32_v_f32m1(row + i, vl);
            auto b = __riscv_vle32_v_f32m1(row + length + i, vl);
            auto c2 = __riscv_vle32_v_f32m1(row + 2 * length + i, vl);
            auto d = __riscv_vle32_v_f32m1(row + 3 * length + i, vl);
            auto rows = __riscv_vcreate_v_f32m1x4(a, b, c2, d);
            __riscv_vssseg4e32_v_f32m1x4(dst + i * dim + c, dim * sizeof(float), rows, vl);
            i += vl;
        }
    }
}

void RvvLinearAttention::runHeads(const float* conv, const float* beta, const float* decay, float* output,
                                  float* panels, Tensor* outTensor) {
    const int batch = mBatch, channels = mChannels, length = mLength, heads = mParam->num_v_heads;
    const int dk = mParam->head_k_dim, dv = mParam->head_v_dim;
    const auto cpu = static_cast<CPUBackend*>(backend());
    const auto core = cpu->functions();
    const int threads = cpu->threadNumber();
    const int workers = std::min(batch * heads, threads);
    constexpr int chunk = kRvvGatedDeltaChunk;
    const bool chunked = useChunks();
    const int panelLength = chunked ? chunk : 1;
    const size_t panelFloats = panelLength * (2 * dk + 2 * dv);
    const int stride = heads * dv;
    const float qScale = 1.0f / std::sqrt(float(dk));
    auto runHeads = [&](int worker, void* scratch, size_t bytes) {
        const size_t stateBytes = size_t(dk) * dv * sizeof(float);
        const size_t usedBytes = stateBytes + panelFloats * sizeof(float);
        const bool staged = scratch && bytes >= usedBytes;
        auto q = staged ? static_cast<float*>(scratch) + dk * dv : panels + worker * panelFloats;
        auto k = q + panelLength * dk;
        auto v = k + panelLength * dk;
        auto projection = v + panelLength * dv;
        for (int idx = worker; idx < batch * heads; idx += workers) {
            const int h = idx % heads, b = idx / heads;
            const int keyHead = h / (heads / mParam->num_k_heads);
            const size_t tokenBase = size_t(b) * length;
            auto convBase = conv + tokenBase * channels;
            const auto qSrc = convBase + size_t(keyHead) * dk * length;
            const auto kSrc = qSrc + size_t(mParam->num_k_heads) * dk * length;
            const auto vSrc = convBase + size_t(2 * mParam->num_k_heads * dk + h * dv) * length;
            auto persistent = mState->mRecurrentState->host<float>() + size_t(idx) * dk * dv;
            auto recurrent = staged ? static_cast<float*>(scratch) : persistent;
            if (staged) {
                ::memcpy(recurrent, persistent, stateBytes);
            }
            int t = 0;
            for (; chunked && t + chunk <= length; t += chunk) {
                float decays[chunk], betas[chunk];
                for (int i = 0; i < chunk; ++i) {
                    decays[i] = decay[(tokenBase + t + i) * heads + h];
                    betas[i] = beta[(tokenBase + t + i) * heads + h];
                }
                packPanel(q, qSrc + t, length);
                packPanel(k, kSrc + t, length);
                packPanel(v, vSrc + t, length);
                for (int i = 0; i < chunk; ++i) {
                    core->MNNNormalizeQKAndDot(q + i * dk, k + i * dk, qScale, true, dk);
                }
                auto out = output + (tokenBase + t) * stride + h * dv;
#ifdef MNN_USE_SPACEMIT_IME2
                if (staged && MNNIme2GatedPrefill(recurrent, q, k, v, projection, out, decays, betas, stride,
                                                  static_cast<uint8_t*>(scratch) + usedBytes, bytes - usedBytes)) {
                    continue;
                }
#endif
                MNNGatedDeltaChunk_RVV(recurrent, q, k, v, projection, out, decays, betas, stride, false);
            }
            for (; t < length; ++t) {
                for (int c = 0; c < dk; ++c) {
                    q[c] = qSrc[c * length + t];
                    k[c] = kSrc[c * length + t];
                }
                for (int c = 0; c < dv; ++c) {
                    v[c] = vSrc[c * length + t];
                }
                const auto kq = core->MNNNormalizeQKAndDot(q, k, qScale, mParam->use_qk_l2norm, dk);
                auto out = output + (tokenBase + t) * stride + h * dv;
                const float g = decay[(tokenBase + t) * heads + h], b = beta[(tokenBase + t) * heads + h];
                if (length == 1) {
                    core->MNNDualMatVec(recurrent, k, q, projection, out, dk, dv);
                    for (int i = 0; i < dv; ++i) {
                        v[i] = b * (v[i] - g * projection[i]);
                        out[i] = g * out[i] + kq * v[i];
                    }
                    core->MNNDecayRankOneUpdate(recurrent, k, v, g, dk, dv);
                } else {
                    core->MNNFusedGatedDelta(recurrent, k, q, v, out, g, b, kq, dk, dv);
                }
            }
            if (staged) {
                ::memcpy(persistent, recurrent, stateBytes);
            }
        }
    };
    bool dispatched = false;
#ifdef MNN_USE_SPACEMIT_IME2
    if (chunked && length >= 64 && cpu->precisionMode() == BackendConfig::Precision_Low) {
        auto task = [](size_t id, void* scratch, size_t bytes, void* ctx) {
            (*static_cast<decltype(runHeads)*>(ctx))(static_cast<int>(id), scratch, bytes);
        };
        dispatched = MNNSpacemitIme2RunTcmTasks(workers, task, &runHeads) == size_t(workers);
    }
#endif
    if (!dispatched) {
        MNN_CONCURRENCY_BEGIN(worker, workers) {
            runHeads(worker, nullptr, 0);
        }
        MNN_CONCURRENCY_END();
    }
    if (TensorUtils::getDescribe(outTensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        const int tokens = batch * length * heads;
        int offsets[2] = {dv, tokens};
        MNN_CONCURRENCY_BEGIN(worker, threads) {
            const int begin = tokens * worker / threads;
            const int count = tokens * (worker + 1) / threads - begin;
            if (count > 0) {
                core->MNNPackCUnitTranspose(outTensor->host<float>() + size_t(begin) * core->pack,
                                            output + size_t(begin) * dv, count, dv, offsets);
            }
        }
        MNN_CONCURRENCY_END();
    }
}
} // namespace

Execution* MNNRvvCreateLinearAttention(CPUBackend* backend, const Op* op, const std::vector<Tensor*>& inputs) {
    const auto p = op->main_as_LinearAttentionParam();
    if (!p || !p->attn_type() || p->attn_type()->str() != "gated_delta_rule" ||
        p->head_k_dim() <= 0 || p->head_v_dim() <= 0 || p->num_k_heads() <= 0 ||
        p->num_v_heads() < p->num_k_heads() || p->num_v_heads() % p->num_k_heads() ||
        backend->functions()->bytes != 4 || backend->functions()->pack != 4) {
        return nullptr;
    }
    // StaticModule pre-creates the state-owning execution without tensors.
    // Its clones validate real shapes in onResize.
    if (!inputs.empty() && !supportsLinearInputs(inputs, 2LL * p->num_k_heads() * p->head_k_dim() +
                                                int64_t(p->num_v_heads()) * p->head_v_dim())) return nullptr;
    if (p->gate_fold() && (!p->gate_coef() || !p->gate_bias() || p->gate_coef()->size() != p->num_v_heads() ||
                          p->gate_bias()->size() != p->num_v_heads())) {
        return nullptr;
    }
    return new RvvLinearAttention(backend, std::shared_ptr<const LinearAttentionParamT>(p->UnPack()),
                                  std::make_shared<StateCache>());
}
} // namespace MNN
#endif
