// Copyright © 2026, Alibaba Group Holding Limited
#include <riscv_vector.h>
#include <stddef.h>
#include "MNNRvvPrefill.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "RVVBackend.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPULinearAttention.hpp"
#include "core/Concurrency.h"
#include "core/MNNFileUtils.h"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <atomic>
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
extern "C" size_t MNNSpacemitIme2RunDecodeTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
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
    ErrorCode onExecuteWithCallBack(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mAllowHandoff = false;
        auto code = onExecute(inputs, outputs);
        mAllowHandoff = true;
        return code;
    }

private:
    bool useChunks() const {
        const size_t workers = std::min(mBatch * mParam->num_v_heads, static_cast<CPUBackend*>(backend())->threadNumber());
        const size_t panelFloats = workers * kRvvGatedDeltaChunk * 4 * kRvvGatedDeltaHeadDim;
        return mParam->use_qk_l2norm && mParam->head_k_dim == kRvvGatedDeltaHeadDim &&
               mParam->head_v_dim == kRvvGatedDeltaHeadDim && mLength >= kRvvGatedDeltaChunk &&
               size_t(mBatch) * mChannels * (mLength + mKernel - 1) >= panelFloats;
    }
    bool convolve(const Tensor* qkv, const float* weight, float* conv, float* work);
    void gates(const Tensor* gate, const Tensor* beta, float* decay, float* betaValues);
    void runHeads(float* conv, bool convIsC4, const Tensor* fusedQkvTensor, const float* convWeight,
                  float* convState, const float* beta, const float* decay, float* output, float* panels, Tensor* out);
    ErrorCode prefix(const std::string& path, bool load);
    std::shared_ptr<const LinearAttentionParamT> mParam;
    std::shared_ptr<StateCache> mState;
    MemChunk mWork;
    size_t mConvBytes = 0, mRecurrentBytes = 0;
    int mBatch = 0, mLength = 0, mChannels = 0, mKernel = 0;
    bool mAllowHandoff = true;
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

bool RvvLinearAttention::convolve(const Tensor* qkv, const float* weight, float* conv, float* work) {
    const auto in = qkv->host<float>();
    auto history = mState->mConvState->host<float>();
    const auto cpu = static_cast<CPUBackend*>(backend());
    const int threads = cpu->threadNumber();
    const bool low = cpu->precisionMode() == BackendConfig::Precision_Low;
    const int length = mLength, carry = mKernel - 1;
    const bool packed = isC4(qkv);
    const bool packedKernel = packed && mChannels % 4 == 0 && mKernel == 4 && length >= kRvvGatedDeltaChunk;
    const bool keepC4 = packedKernel && mParam->head_k_dim % 4 == 0 && mParam->head_v_dim % 4 == 0;
#ifdef MNN_USE_SPACEMIT_IME2
    if (keepC4 && length >= 64 && low && threads <= 8) {
        struct ConvCtx {
            float* conv;
            const float* in;
            float* history;
            const float* weight;
            int channels;
            int length;
            int threads;
        } ctx{conv, in, history, weight, mChannels, length, threads};
        auto task = [](size_t t, void*, size_t, void* data) {
            auto& p = *static_cast<ConvCtx*>(data);
            for (int c = 4 * static_cast<int>(t); c < p.channels; c += 4 * p.threads) {
                MNNConv4SiLuC4KeepC4_RVV(p.conv + size_t(c) * p.length, p.in + size_t(c) * p.length, p.history + c * 3,
                                         p.weight + c * 4, p.length, true);
            }
        };
        if (MNNSpacemitIme2RunTcmTasks(threads, task, &ctx) == size_t(threads)) {
            return true;
        }
    }
#endif
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
        } else if (keepC4) {
            for (int c = 4 * t; c < mChannels; c += 4 * threads) {
                MNNConv4SiLuC4KeepC4_RVV(conv + size_t(c) * length, in + size_t(c) * length, history + c * 3,
                                         weight + c * 4, length, low);
            }
        } else if (packedKernel) {
            for (int c = 4 * t; c < mChannels; c += 4 * threads) {
                MNNConv4SiLuC4_RVV(conv + size_t(c) * length, in + size_t(c) * length, history + c * 3,
                                   weight + c * 4, length, low);
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
    return keepC4;
}

void RvvLinearAttention::gates(const Tensor* gate, const Tensor* beta, float* decay, float* betaValues) {
    auto cpu = static_cast<CPUBackend*>(backend());
    auto core = cpu->functions();
    const int heads = mParam->num_v_heads;
    const int totalTokens = mBatch * mLength;
    const int count = totalTokens * heads;
    auto expFn = core->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_EXP, cpu->precisionMode());
    auto logFn = core->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_LOG, cpu->precisionMode());
    auto sigFn = core->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_SIGMOID, cpu->precisionMode());
    const bool gateC4 = isC4(gate), betaC4 = isC4(beta);
    const float* gateSrc = gate->host<float>();
    const float* betaSrc = beta->host<float>();
    const bool fold = mParam->gate_fold;
    const float* biasPtr = fold ? mParam->gate_bias.data() : nullptr;
    const float* coefPtr = fold ? mParam->gate_coef.data() : nullptr;

    auto processSlice = [&](int tokBegin, int tokEnd) {
        const int sliceTokens = tokEnd - tokBegin;
        const int sliceCount = sliceTokens * heads;
        float* dOut = decay + size_t(tokBegin) * heads;
        float* bOut = betaValues + size_t(tokBegin) * heads;
        auto unpackSlice = [&](const float* src, bool c4, float* dst) {
            if (c4) {
                int off[] = {totalTokens, heads};
                core->MNNUnpackCUnitTranspose(dst, src + size_t(tokBegin) * core->pack, sliceTokens, heads, off);
            } else {
                ::memcpy(dst, src + size_t(tokBegin) * heads, sliceCount * sizeof(float));
            }
        };
        unpackSlice(gateSrc, gateC4, dOut);
        unpackSlice(betaSrc, betaC4, bOut);
        if (fold) {
            for (int t = 0; t < sliceTokens; ++t) {
                for (int h = 0; h < heads; ++h) {
                    dOut[t * heads + h] += biasPtr[h];
                }
            }
            expFn(reinterpret_cast<int8_t*>(dOut), reinterpret_cast<int8_t*>(dOut), sliceCount);
            for (int i = 0; i < sliceCount; ++i) {
                dOut[i] += 1.0f;
            }
            logFn(reinterpret_cast<int8_t*>(dOut), reinterpret_cast<int8_t*>(dOut), sliceCount);
            for (int t = 0; t < sliceTokens; ++t) {
                for (int h = 0; h < heads; ++h) {
                    dOut[t * heads + h] *= coefPtr[h];
                }
            }
            sigFn(reinterpret_cast<int8_t*>(bOut), reinterpret_cast<int8_t*>(bOut), sliceCount);
        }
        if (mLength == 1) {
            for (int i = 0; i < sliceCount; ++i) {
                dOut[i] = expf(dOut[i]);
            }
        } else {
            float offset[] = {1.0f, 0.0f, 0.0f, 0.0f};
            MNNExp(dOut, dOut, offset, sliceCount);
        }
    };

    const int threads = cpu->threadNumber();
    if (mLength >= 32 && count >= 1024 && threads > 1) {
        const int workers = std::min(threads, totalTokens / 16);
        MNN_CONCURRENCY_BEGIN(w, workers) {
            const int begin = totalTokens * w / workers;
            const int end = totalTokens * (w + 1) / workers;
            if (begin < end) {
                processSlice(begin, end);
            }
        }
        MNN_CONCURRENCY_END();
    } else {
        processSlice(0, totalTokens);
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
    bool fusedConv = false;
#ifdef MNN_USE_SPACEMIT_IME2
    {
        auto cpu = static_cast<CPUBackend*>(backend());
        const bool low = cpu->precisionMode() == BackendConfig::Precision_Low;
        // Single-token decode: run conv4+SiLU and gated-delta state update on balanced A100 cores
        // so mRecurrentState stays off X100 L2 (avoiding cross-cluster writeback stalls on A100 GEMV).
        if (mLength == 1 && mBatch == 1 && mKernel == 4 && mParam->num_k_heads == heads &&
            mParam->head_k_dim == kRvvGatedDeltaHeadDim && mParam->head_v_dim == kRvvGatedDeltaHeadDim && low) {
            gates(inputs[1], inputs[2], decay, beta);
            RvvHandoffContext* handoff = mAllowHandoff ? MNNRvvGetHandoffContext() : nullptr;
            const bool wantHeadMajor = handoff && handoff->backend == backend() && isC4(outputs[0]) &&
                                       handoff->linearAttnToGatedRms.count(outputs[0]) != 0;
            float* dstBase = wantHeadMajor ? outputs[0]->host<float>() : out;
            const int decodeWorkers = std::min(8, heads);
            struct DecodeCtx {
                const CoreFunctions* core;
                const float* qkvIn;
                const float* convW;
                float* convHist;
                float* recState;
                const float* decay;
                const float* beta;
                float* dst;
                int heads;
                int workers;
                bool useL2Norm;
                float qScale;
            } dctx{cpu->functions(), inputs[0]->host<float>(), inputs[3]->host<float>(),
                   mState->mConvState->host<float>(), mState->mRecurrentState->host<float>(),
                   decay, beta, dstBase, heads, decodeWorkers, mParam->use_qk_l2norm,
                   1.0f / std::sqrt(float(kRvvGatedDeltaHeadDim))};
            auto decodeTask = [](size_t w, void*, size_t, void* user) {
                auto& p = *static_cast<DecodeCtx*>(user);
                constexpr int dim = kRvvGatedDeltaHeadDim;
                const bool fastM4 = (__riscv_vsetvl_e32m4(dim) == dim);
                auto siluM2 = [](vfloat32m2_t x, size_t vl) -> vfloat32m2_t {
                    auto z = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(__riscv_vfneg_v_f32m2(x, vl), -87.0f, vl),
                                                    87.0f, vl);
                    auto n = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(z, 1.4426950408889634f, vl), vl);
                    auto r = __riscv_vfnmsac_vf_f32m2(z, 0.6931471805599453f, __riscv_vfcvt_f_x_v_f32m2(n, vl), vl);
                    auto y = __riscv_vfmv_v_f_f32m2(1.0f / 720.0f, vl);
                    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.0f / 120.0f, vl), vl);
                    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.0f / 24.0f, vl), vl);
                    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.0f / 6.0f, vl), vl);
                    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(0.5f, vl), vl);
                    const auto one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
                    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
                    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
                    auto pw = __riscv_vreinterpret_v_i32m2_f32m2(
                        __riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(n, 127, vl), 23, vl));
                    auto denom = __riscv_vfmacc_vv_f32m2(one, y, pw, vl);
                    auto rcp = __riscv_vfrec7_v_f32m2(denom, vl);
                    rcp = __riscv_vfmul_vv_f32m2(rcp, __riscv_vfnmsac_vv_f32m2(__riscv_vfmv_v_f_f32m2(2.0f, vl), denom, rcp, vl), vl);
                    rcp = __riscv_vfmul_vv_f32m2(rcp, __riscv_vfnmsac_vv_f32m2(__riscv_vfmv_v_f_f32m2(2.0f, vl), denom, rcp, vl), vl);
                    return __riscv_vfmul_vv_f32m2(x, rcp, vl);
                };
                alignas(64) float tmp[dim];
                auto convOne = [&](size_t chBase, float* dstVec) {
                    const float* x = p.qkvIn + chBase;
                    float* h = p.convHist + chBase * 3;
                    const float* cw = p.convW + chBase * 4;
                    if (fastM4) {
                        for (int off = 0; off < dim; off += 64) {
                            const size_t vl = 64;
                            auto hs = __riscv_vlseg3e32_v_f32m2x3(h + off * 3, vl);
                            auto ws = __riscv_vlseg4e32_v_f32m2x4(cw + off * 4, vl);
                            auto xv = __riscv_vle32_v_f32m2(x + off, vl);
                            auto h0 = __riscv_vget_v_f32m2x3_f32m2(hs, 0);
                            auto h1 = __riscv_vget_v_f32m2x3_f32m2(hs, 1);
                            auto h2 = __riscv_vget_v_f32m2x3_f32m2(hs, 2);
                            auto s = __riscv_vfmul_vv_f32m2(h0, __riscv_vget_v_f32m2x4_f32m2(ws, 0), vl);
                            s = __riscv_vfmacc_vv_f32m2(s, h1, __riscv_vget_v_f32m2x4_f32m2(ws, 1), vl);
                            s = __riscv_vfmacc_vv_f32m2(s, h2, __riscv_vget_v_f32m2x4_f32m2(ws, 2), vl);
                            s = __riscv_vfmacc_vv_f32m2(s, xv, __riscv_vget_v_f32m2x4_f32m2(ws, 3), vl);
                            auto nextH = __riscv_vcreate_v_f32m2x3(h1, h2, xv);
                            __riscv_vsseg3e32_v_f32m2x3(h + off * 3, nextH, vl);
                            __riscv_vse32_v_f32m2(dstVec + off, siluM2(s, vl), vl);
                        }
                        return;
                    }
                    for (int c = 0; c < dim; ++c) {
                        float* hs = h + c * 3;
                        const float* w4 = cw + c * 4;
                        const float xc = x[c];
                        const float s = hs[0] * w4[0] + hs[1] * w4[1] + hs[2] * w4[2] + xc * w4[3];
                        hs[0] = hs[1];
                        hs[1] = hs[2];
                        hs[2] = xc;
                        tmp[c] = s;
                    }
                    MNNSiLu(dstVec, tmp, dim);
                };
                alignas(64) float q[dim], k[dim], v[dim];
                const int hBegin = p.heads * static_cast<int>(w) / p.workers;
                const int hEnd = p.heads * (static_cast<int>(w) + 1) / p.workers;
                for (int h = hBegin; h < hEnd; ++h) {
                    convOne(size_t(h) * dim, q);
                    convOne(size_t(p.heads + h) * dim, k);
                    convOne(size_t(2 * p.heads + h) * dim, v);
                    float* S = p.recState + size_t(h) * dim * dim;
                    float* dstH = p.dst + size_t(h) * dim;
                    const float g = p.decay[h], b = p.beta[h];
                    if (fastM4) {
                        auto vZero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                        auto qv = __riscv_vle32_v_f32m4(q, dim);
                        auto kv = __riscv_vle32_v_f32m4(k, dim);
                        if (p.useL2Norm) {
                            const float qss = __riscv_vfmv_f_s_f32m1_f32(
                                __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(qv, qv, dim), vZero, dim));
                            const float kss = __riscv_vfmv_f_s_f32m1_f32(
                                __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(kv, kv, dim), vZero, dim));
                            qv = __riscv_vfmul_vf_f32m4(qv, p.qScale / std::sqrt(qss + 1e-6f), dim);
                            kv = __riscv_vfmul_vf_f32m4(kv, 1.0f / std::sqrt(kss + 1e-6f), dim);
                        } else {
                            qv = __riscv_vfmul_vf_f32m4(qv, p.qScale, dim);
                        }
                        const float kq = __riscv_vfmv_f_s_f32m1_f32(
                            __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(qv, kv, dim), vZero, dim));
                        __riscv_vse32_v_f32m4(q, qv, dim);
                        __riscv_vse32_v_f32m4(k, kv, dim);
                        // 4-row unrolled e32m4 (vl=128 == dv) keeps outK/outQ in registers across all 128 rows.
                        vfloat32m4_t outK = __riscv_vfmv_v_f_f32m4(0.0f, dim);
                        vfloat32m4_t outQ = __riscv_vfmv_v_f_f32m4(0.0f, dim);
                        for (int i = 0; i < dim; i += 4) {
                            const float* r0 = S + size_t(i) * dim;
                            auto s0 = __riscv_vle32_v_f32m4(r0, dim);
                            auto s1 = __riscv_vle32_v_f32m4(r0 + dim, dim);
                            auto s2 = __riscv_vle32_v_f32m4(r0 + 2 * dim, dim);
                            auto s3 = __riscv_vle32_v_f32m4(r0 + 3 * dim, dim);
                            outK = __riscv_vfmacc_vf_f32m4(outK, k[i], s0, dim);
                            outQ = __riscv_vfmacc_vf_f32m4(outQ, q[i], s0, dim);
                            outK = __riscv_vfmacc_vf_f32m4(outK, k[i + 1], s1, dim);
                            outQ = __riscv_vfmacc_vf_f32m4(outQ, q[i + 1], s1, dim);
                            outK = __riscv_vfmacc_vf_f32m4(outK, k[i + 2], s2, dim);
                            outQ = __riscv_vfmacc_vf_f32m4(outQ, q[i + 2], s2, dim);
                            outK = __riscv_vfmacc_vf_f32m4(outK, k[i + 3], s3, dim);
                            outQ = __riscv_vfmacc_vf_f32m4(outQ, q[i + 3], s3, dim);
                        }
                        auto vv = __riscv_vle32_v_f32m4(v, dim);
                        auto delta = __riscv_vfmul_vf_f32m4(__riscv_vfnmsac_vf_f32m4(vv, g, outK, dim), b, dim);
                        auto outV = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(delta, kq, dim), g, outQ, dim);
                        __riscv_vse32_v_f32m4(dstH, outV, dim);
                        for (int i = 0; i < dim; i += 4) {
                            float* r0 = S + size_t(i) * dim;
                            auto s0 = __riscv_vle32_v_f32m4(r0, dim);
                            auto s1 = __riscv_vle32_v_f32m4(r0 + dim, dim);
                            auto s2 = __riscv_vle32_v_f32m4(r0 + 2 * dim, dim);
                            auto s3 = __riscv_vle32_v_f32m4(r0 + 3 * dim, dim);
                            s0 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(delta, k[i], dim), g, s0, dim);
                            s1 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(delta, k[i + 1], dim), g, s1, dim);
                            s2 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(delta, k[i + 2], dim), g, s2, dim);
                            s3 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(delta, k[i + 3], dim), g, s3, dim);
                            __riscv_vse32_v_f32m4(r0, s0, dim);
                            __riscv_vse32_v_f32m4(r0 + dim, s1, dim);
                            __riscv_vse32_v_f32m4(r0 + 2 * dim, s2, dim);
                            __riscv_vse32_v_f32m4(r0 + 3 * dim, s3, dim);
                        }
                    } else {
                        const float kq = p.core->MNNNormalizeQKAndDot(q, k, p.qScale, p.useL2Norm, dim);
                        MNNFusedGatedDelta_RVV(S, k, q, v, dstH, g, b, kq, dim, dim);
                    }
                }
            };
            const size_t ran = decodeWorkers <= 4 ? MNNSpacemitIme2RunDecodeTasks(decodeWorkers, decodeTask, &dctx)
                                                  : MNNSpacemitIme2RunTcmTasks(decodeWorkers, decodeTask, &dctx);
            if (ran == size_t(decodeWorkers)) {
                if (wantHeadMajor) {
                    auto& info = handoff->activeLinearAttn[outputs[0]];
                    info.ptr = outputs[0]->host<float>();
                    info.batch = 1;
                    info.length = 1;
                    info.heads = heads;
                    info.dv = kRvvGatedDeltaHeadDim;
                } else {
                    if (handoff && handoff->backend == backend()) {
                        handoff->activeLinearAttn.erase(outputs[0]);
                    }
                    if (isC4(outputs[0])) {
                        int offsets[2] = {kRvvGatedDeltaHeadDim, heads};
                        cpu->functions()->MNNPackCUnitTranspose(outputs[0]->host<float>(), out, heads,
                                                                kRvvGatedDeltaHeadDim, offsets);
                    }
                }
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
        }
        fusedConv = isC4(inputs[0]) && mKernel == 4 && mChannels % 4 == 0 && useChunks() &&
                    mParam->num_k_heads == mParam->num_v_heads && mLength >= kRvvGatedDeltaChunk &&
                    cpu->threadNumber() <= 8 && low;
    }
#endif
    bool convIsC4 = false;
    if (!fusedConv) {
        convIsC4 = convolve(inputs[0], inputs[3]->host<float>(), conv, work);
    }
    gates(inputs[1], inputs[2], decay, beta);
    runHeads(conv, convIsC4, fusedConv ? inputs[0] : nullptr, inputs[3]->host<float>(),
             mState->mConvState->host<float>(), beta, decay, out, work, outputs[0]);
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

// Transpose 128 channels into Chunk consecutive token rows. Four contiguous input rows
// fit in registers; the segmented store interleaves their channels in the panel.
template <int Chunk>
static void packPanelN(float* dst, const float* src, int length) {
    constexpr int dim = kRvvGatedDeltaHeadDim;
    for (int c = 0; c < dim; c += 4) {
        const float* row = src + size_t(c) * length;
        for (int i = 0; i < Chunk;) {
            size_t vl = __riscv_vsetvl_e32m1(Chunk - i);
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

// Pack from NC4HW4 [dim/4, length, 4] into [Chunk, dim] with sequential 16-byte loads.
template <int Chunk>
static void packPanelC4(float* dst, const float* src, int length, int t) {
    constexpr int dim = kRvvGatedDeltaHeadDim;
    for (int c = 0; c < dim; c += 4) {
        const float* s = src + size_t(c) * length + size_t(t) * 4;
        float* d = dst + c;
        for (int i = 0; i < Chunk; ++i) {
            ::memcpy(d + i * dim, s + i * 4, 16);
        }
    }
}

static inline void normalizeQKRow_RVV(float* q, float* k, float qScale, size_t dk) {
    const size_t vl0 = __riscv_vsetvl_e32m4(dk);
    auto vZero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    if (vl0 == dk) {
        auto vq = __riscv_vle32_v_f32m4(q, vl0);
        auto vk = __riscv_vle32_v_f32m4(k, vl0);
        const float qSumSq = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vq, vq, vl0), vZero, vl0));
        const float kSumSq = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vk, vk, vl0), vZero, vl0));
        __riscv_vse32_v_f32m4(q, __riscv_vfmul_vf_f32m4(vq, qScale / std::sqrt(qSumSq + 1e-6f), vl0), vl0);
        __riscv_vse32_v_f32m4(k, __riscv_vfmul_vf_f32m4(vk, 1.0f / std::sqrt(kSumSq + 1e-6f), vl0), vl0);
        return;
    }
    float qSumSq = 0.0f, kSumSq = 0.0f;
    for (size_t d = 0; d < dk;) {
        const size_t vl = __riscv_vsetvl_e32m4(dk - d);
        auto vq = __riscv_vle32_v_f32m4(q + d, vl);
        auto vk = __riscv_vle32_v_f32m4(k + d, vl);
        qSumSq += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vq, vq, vl), vZero, vl));
        kSumSq += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vk, vk, vl), vZero, vl));
        d += vl;
    }
    const float qNorm = qScale / std::sqrt(qSumSq + 1e-6f);
    const float kNorm = 1.0f / std::sqrt(kSumSq + 1e-6f);
    for (size_t d = 0; d < dk;) {
        const size_t vl = __riscv_vsetvl_e32m4(dk - d);
        __riscv_vse32_v_f32m4(q + d, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(q + d, vl), qNorm, vl), vl);
        __riscv_vse32_v_f32m4(k + d, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(k + d, vl), kNorm, vl), vl);
        d += vl;
    }
}

// Sweep one head's Q/K/V tile (up to 512 tokens) through 256 KB TCM so DRAM reads
// are contiguous 8 KB streams, then transpose (+ optional L2 norm) into L2-hot headBuf.
static void sweepHeadTileTcm(float* dstHead, const float* src, const float* history, const float* weight,
                             int length, int tileStart, int tileLen, bool doNorm, float normScale, bool low,
                             float* tcmPlane, float* wBuf, float* rowTmp) {
    constexpr int dim = kRvvGatedDeltaHeadDim;
    for (int c = 0; c < dim; c += 4) {
        MNNConv4SiLuC4Sweep_RVV(tcmPlane + size_t(c) * tileLen,
                                src + size_t(c) * length + size_t(tileStart) * 4,
                                history + c * 3, weight + c * 4, tileLen, tileStart == 0, low, wBuf);
    }
    const size_t vl0 = __riscv_vsetvl_e32m4(dim);
    auto vZero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (int t = 0; t < tileLen; ++t) {
        uint64_t* rd = reinterpret_cast<uint64_t*>(rowTmp);
        for (int c = 0; c < dim; c += 4) {
            const uint64_t* ts = reinterpret_cast<const uint64_t*>(tcmPlane + size_t(c) * tileLen + size_t(t) * 4);
            rd[c / 2 + 0] = ts[0];
            rd[c / 2 + 1] = ts[1];
        }
        float* outRow = dstHead + size_t(t) * dim;
        if (doNorm) {
            if (vl0 == dim) {
                auto v = __riscv_vle32_v_f32m4(rowTmp, dim);
                const float ss = __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(v, v, dim), vZero, dim));
                __riscv_vse32_v_f32m4(outRow, __riscv_vfmul_vf_f32m4(v, normScale / std::sqrt(ss + 1e-6f), dim), dim);
            } else {
                float ss = 0.0f;
                for (int d = 0; d < dim;) {
                    const size_t vl = __riscv_vsetvl_e32m4(dim - d);
                    auto v = __riscv_vle32_v_f32m4(rowTmp + d, vl);
                    ss += __riscv_vfmv_f_s_f32m1_f32(
                        __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(v, v, vl), vZero, vl));
                    d += vl;
                }
                const float scale = normScale / std::sqrt(ss + 1e-6f);
                for (int d = 0; d < dim;) {
                    const size_t vl = __riscv_vsetvl_e32m4(dim - d);
                    __riscv_vse32_v_f32m4(outRow + d,
                                          __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(rowTmp + d, vl), scale, vl), vl);
                    d += vl;
                }
            }
        } else {
            for (int d = 0; d < dim;) {
                const size_t vl = __riscv_vsetvl_e32m4(dim - d);
                __riscv_vse32_v_f32m4(outRow + d, __riscv_vle32_v_f32m4(rowTmp + d, vl), vl);
                d += vl;
            }
        }
    }
}

static inline void updateHeadConvHistory(float* history, const float* src, int dim, int length) {
    for (int c = 0; c < dim; ++c) {
        const float* s = src + size_t(c / 4) * length * 4 + (c % 4);
        float* h = history + c * 3;
        h[0] = s[(length - 3) * 4];
        h[1] = s[(length - 2) * 4];
        h[2] = s[(length - 1) * 4];
    }
}

void RvvLinearAttention::runHeads(float* conv, bool convIsC4, const Tensor* fusedQkvTensor, const float* convWeight,
                                  float* convState, const float* beta, const float* decay, float* output,
                                  float* panels, Tensor* outTensor) {
    const float* fusedQkv = fusedQkvTensor ? fusedQkvTensor->host<float>() : nullptr;
    const int batch = mBatch, channels = mChannels, length = mLength, heads = mParam->num_v_heads;
    const int dk = mParam->head_k_dim, dv = mParam->head_v_dim;
    const auto cpu = static_cast<CPUBackend*>(backend());
    const auto core = cpu->functions();
    const int threads = cpu->threadNumber();
    const int workers = std::min(batch * heads, threads);
    const bool low = cpu->precisionMode() == BackendConfig::Precision_Low;
    constexpr int chunk = kRvvGatedDeltaChunk;
    const bool chunked = useChunks();
    const int panelLength = chunked ? chunk : 1;
    const size_t panelFloats = panelLength * (2 * dk + 2 * dv);
    const int stride = heads * dv;
    const float qScale = 1.0f / std::sqrt(float(dk));
    RvvHandoffContext* handoff = mAllowHandoff ? MNNRvvGetHandoffContext() : nullptr;
    const bool wantHeadMajor = handoff && handoff->backend == backend() && batch == 1 && isC4(outTensor) &&
                               handoff->linearAttnToGatedRms.count(outTensor) != 0;
    auto runHeads = [&](int worker, void* scratch, size_t bytes) {
        const bool headMajorOut = (fusedQkv != nullptr) && wantHeadMajor;
        const int headStride = headMajorOut ? dv : stride;
        auto headOutAt = [&](int h, size_t tokenBase, int tok) -> float* {
            return headMajorOut ? outTensor->host<float>() + (size_t(h) * length + tok) * dv
                                : output + (tokenBase + tok) * stride + h * dv;
        };
        const size_t stateBytes = size_t(dk) * dv * sizeof(float);
        const size_t usedBytes = stateBytes + panelFloats * sizeof(float);
        const bool staged = scratch && bytes >= usedBytes;
#ifdef MNN_USE_SPACEMIT_IME2
        constexpr size_t kChunk64 = 64;
        constexpr size_t kChunk64PanelBytes = kChunk64 * 4 * kRvvGatedDeltaHeadDim * sizeof(float);
        constexpr size_t kChunk64ScratchBytes = 162 * 1024;
        constexpr int kSweepMaxTile = 512;
        const size_t kSweepTcmBytes = stateBytes + size_t(kRvvGatedDeltaHeadDim) * kSweepMaxTile * sizeof(float) +
                                      (256 + kRvvGatedDeltaHeadDim) * sizeof(float);
        const bool canChunk64 = staged && chunked && bytes >= stateBytes + kChunk64PanelBytes + kChunk64ScratchBytes;
        const bool canSweepTcm = fusedQkv != nullptr && canChunk64 && bytes >= kSweepTcmBytes;
#endif
        auto q = staged ? static_cast<float*>(scratch) + dk * dv : panels + worker * panelFloats;
        auto k = q + panelLength * dk;
        auto v = k + panelLength * dk;
        auto projection = v + panelLength * dv;
        for (int idx = worker; idx < batch * heads; idx += workers) {
            const int h = idx % heads, b = idx / heads;
            const int keyHead = h / (heads / mParam->num_k_heads);
            const size_t tokenBase = size_t(b) * length;
            auto srcBase = (fusedQkv ? fusedQkv : conv) + tokenBase * channels;
            const size_t qCh = size_t(keyHead) * dk;
            const size_t kCh = size_t(mParam->num_k_heads + keyHead) * dk;
            const size_t vCh = size_t(2 * mParam->num_k_heads) * dk + size_t(h) * dv;
            const float* qSrc = srcBase + qCh * length;
            const float* kSrc = srcBase + kCh * length;
            const float* vSrc = srcBase + vCh * length;
            bool localConvIsC4 = convIsC4;
            auto persistent = mState->mRecurrentState->host<float>() + size_t(idx) * dk * dv;
            auto recurrent = staged ? static_cast<float*>(scratch) : persistent;
            if (staged) {
                ::memcpy(recurrent, persistent, stateBytes);
            }
#ifdef MNN_USE_SPACEMIT_IME2
            // Disjoint per-worker slice of the unused conv workspace (3 * length * dk floats <= channels * length).
            float* workerBuf = fusedQkv ? conv + size_t(worker) * 3 * size_t(length) * dk : nullptr;
            if (canSweepTcm) {
                const int tileMax = std::min(length, kSweepMaxTile);
                float* qHead = workerBuf;
                float* kHead = qHead + size_t(tileMax) * dk;
                float* vHead = kHead + size_t(tileMax) * dk;
                float* tcmPlane = static_cast<float*>(scratch) + dk * dv;
                float* wBuf = tcmPlane + size_t(dk) * tileMax;
                float* rowTmp = wBuf + 256;
                float* proj64 = static_cast<float*>(scratch) + dk * dv;
                void* scratch64 = static_cast<uint8_t*>(scratch) + stateBytes + kChunk64 * dv * sizeof(float);
                const size_t scratch64Bytes = bytes - stateBytes - kChunk64 * dv * sizeof(float);
                for (int tileStart = 0; tileStart < length; tileStart += kSweepMaxTile) {
                    const int tileLen = std::min(kSweepMaxTile, length - tileStart);
                    sweepHeadTileTcm(qHead, qSrc, convState + qCh * 3, convWeight + qCh * 4, length, tileStart,
                                     tileLen, true, qScale, low, tcmPlane, wBuf, rowTmp);
                    sweepHeadTileTcm(kHead, kSrc, convState + kCh * 3, convWeight + kCh * 4, length, tileStart,
                                     tileLen, true, 1.0f, low, tcmPlane, wBuf, rowTmp);
                    sweepHeadTileTcm(vHead, vSrc, convState + vCh * 3, convWeight + vCh * 4, length, tileStart,
                                     tileLen, false, 1.0f, low, tcmPlane, wBuf, rowTmp);
                    int t = 0;
                    for (; t + int(kChunk64) <= tileLen; t += int(kChunk64)) {
                        float decays[kChunk64], betas[kChunk64];
                        for (size_t i = 0; i < kChunk64; ++i) {
                            decays[i] = decay[(tokenBase + tileStart + t + i) * heads + h];
                            betas[i] = beta[(tokenBase + tileStart + t + i) * heads + h];
                        }
                        float* qt = qHead + size_t(t) * dk;
                        float* kt = kHead + size_t(t) * dk;
                        float* vt = vHead + size_t(t) * dv;
                        auto out = headOutAt(h, tokenBase, tileStart + t);
                        if (MNNIme2GatedPrefill64(recurrent, qt, kt, vt, proj64, out, decays, betas, headStride,
                                                  scratch64, scratch64Bytes)) {
                            continue;
                        }
                        for (size_t s = 0; s < kChunk64; s += chunk) {
                            MNNGatedDeltaChunk_RVV(recurrent, qt + s * dk, kt + s * dk, vt + s * dv, proj64,
                                                   out + s * headStride, decays + s, betas + s, headStride, false);
                        }
                    }
                    const int rem = tileLen - t;
                    if (rem >= chunk && bytes >= stateBytes + kChunk64PanelBytes + kChunk64ScratchBytes) {
                        // Pad final <64 tail with identity tokens (q=k=v=0, decay=1, beta=0) in TCM.
                        auto qTail = static_cast<float*>(scratch) + dk * dv;
                        auto kTail = qTail + kChunk64 * dk;
                        auto vTail = kTail + kChunk64 * dk;
                        auto projTail = vTail + kChunk64 * dv;
                        void* scTail = static_cast<uint8_t*>(scratch) + stateBytes + kChunk64PanelBytes;
                        const size_t scTailBytes = bytes - stateBytes - kChunk64PanelBytes;
                        const size_t remFloats = size_t(rem) * dk;
                        const size_t padFloats = size_t(int(kChunk64) - rem) * dk;
                        ::memcpy(qTail, qHead + size_t(t) * dk, remFloats * sizeof(float));
                        ::memset(qTail + remFloats, 0, padFloats * sizeof(float));
                        ::memcpy(kTail, kHead + size_t(t) * dk, remFloats * sizeof(float));
                        ::memset(kTail + remFloats, 0, padFloats * sizeof(float));
                        ::memcpy(vTail, vHead + size_t(t) * dv, remFloats * sizeof(float));
                        ::memset(vTail + remFloats, 0, padFloats * sizeof(float));
                        float decays[kChunk64], betas[kChunk64];
                        for (int i = 0; i < rem; ++i) {
                            decays[i] = decay[(tokenBase + tileStart + t + i) * heads + h];
                            betas[i] = beta[(tokenBase + tileStart + t + i) * heads + h];
                        }
                        for (int i = rem; i < int(kChunk64); ++i) {
                            decays[i] = 1.0f;
                            betas[i] = 0.0f;
                        }
                        auto out = headOutAt(h, tokenBase, tileStart + t);
                        if (MNNIme2GatedPrefill64(recurrent, qTail, kTail, vTail, projTail, out, decays, betas,
                                                  headStride, scTail, scTailBytes, rem)) {
                            t = tileLen;
                        }
                    }
                    for (; t + chunk <= tileLen; t += chunk) {
                        float decays[chunk], betas[chunk];
                        for (int i = 0; i < chunk; ++i) {
                            decays[i] = decay[(tokenBase + tileStart + t + i) * heads + h];
                            betas[i] = beta[(tokenBase + tileStart + t + i) * heads + h];
                        }
                        float* qt = qHead + size_t(t) * dk;
                        float* kt = kHead + size_t(t) * dk;
                        float* vt = vHead + size_t(t) * dv;
                        auto out = headOutAt(h, tokenBase, tileStart + t);
                        if (MNNIme2GatedPrefill(recurrent, qt, kt, vt, proj64, out, decays, betas, headStride,
                                                scratch64, scratch64Bytes)) {
                            continue;
                        }
                        MNNGatedDeltaChunk_RVV(recurrent, qt, kt, vt, proj64, out, decays, betas, headStride, false);
                    }
                    for (; t < tileLen; ++t) {
                        const float* qt = qHead + size_t(t) * dk;
                        const float* kt = kHead + size_t(t) * dk;
                        const float* vt = vHead + size_t(t) * dv;
                        float kq = 0.0f;
                        for (int d = 0; d < dk; ++d) {
                            kq += qt[d] * kt[d];
                        }
                        auto out = headOutAt(h, tokenBase, tileStart + t);
                        const float g = decay[(tokenBase + tileStart + t) * heads + h];
                        const float bVal = beta[(tokenBase + tileStart + t) * heads + h];
                        core->MNNFusedGatedDelta(recurrent, kt, qt, vt, out, g, bVal, kq, dk, dv);
                    }
                }
                updateHeadConvHistory(convState + qCh * 3, qSrc, dk, length);
                updateHeadConvHistory(convState + kCh * 3, kSrc, dk, length);
                updateHeadConvHistory(convState + vCh * 3, vSrc, dv, length);
                ::memcpy(persistent, recurrent, stateBytes);
                continue;
            }
            if (fusedQkv != nullptr) {
                // Worker lacks full sweep TCM: run per-head conv+SiLU locally so each head advances state once.
                float* qLocal = workerBuf;
                float* kLocal = qLocal + size_t(length) * dk;
                float* vLocal = kLocal + size_t(length) * dk;
                auto convHead = [&](float* dst, const float* src, float* hist, const float* w, int dim) {
                    for (int c = 0; c < dim; c += 4) {
                        MNNConv4SiLuC4KeepC4_RVV(dst + size_t(c) * length, src + size_t(c) * length, hist + c * 3,
                                                 w + c * 4, length, low);
                    }
                };
                convHead(qLocal, qSrc, convState + qCh * 3, convWeight + qCh * 4, dk);
                convHead(kLocal, kSrc, convState + kCh * 3, convWeight + kCh * 4, dk);
                convHead(vLocal, vSrc, convState + vCh * 3, convWeight + vCh * 4, dv);
                qSrc = qLocal;
                kSrc = kLocal;
                vSrc = vLocal;
                localConvIsC4 = true;
            }
#endif
            int t = 0;
#ifdef MNN_USE_SPACEMIT_IME2
            if (canChunk64) {
                auto q64 = static_cast<float*>(scratch) + dk * dv;
                auto k64 = q64 + kChunk64 * dk;
                auto v64 = k64 + kChunk64 * dk;
                auto proj64 = v64 + kChunk64 * dv;
                void* scratch64 = static_cast<uint8_t*>(scratch) + stateBytes + kChunk64PanelBytes;
                const size_t scratch64Bytes = bytes - stateBytes - kChunk64PanelBytes;
                for (; t + int(kChunk64) <= length; t += int(kChunk64)) {
                    float decays[kChunk64], betas[kChunk64];
                    for (size_t i = 0; i < kChunk64; ++i) {
                        decays[i] = decay[(tokenBase + t + i) * heads + h];
                        betas[i] = beta[(tokenBase + t + i) * heads + h];
                    }
                    if (localConvIsC4) {
                        packPanelC4<kChunk64>(q64, qSrc, length, t);
                        packPanelC4<kChunk64>(k64, kSrc, length, t);
                        packPanelC4<kChunk64>(v64, vSrc, length, t);
                    } else {
                        packPanelN<kChunk64>(q64, qSrc + t, length);
                        packPanelN<kChunk64>(k64, kSrc + t, length);
                        packPanelN<kChunk64>(v64, vSrc + t, length);
                    }
                    for (size_t i = 0; i < kChunk64; ++i) {
                        normalizeQKRow_RVV(q64 + i * dk, k64 + i * dk, qScale, dk);
                    }
                    auto out = headOutAt(h, tokenBase, t);
                    if (MNNIme2GatedPrefill64(recurrent, q64, k64, v64, proj64, out, decays, betas, headStride,
                                              scratch64, scratch64Bytes)) {
                        continue;
                    }
                    for (size_t s = 0; s < kChunk64; s += chunk) {
                        MNNGatedDeltaChunk_RVV(recurrent, q64 + s * dk, k64 + s * dk, v64 + s * dv, proj64,
                                               out + s * headStride, decays + s, betas + s, headStride, false);
                    }
                }
            }
#endif
            for (; chunked && t + chunk <= length; t += chunk) {
                float decays[chunk], betas[chunk];
                for (int i = 0; i < chunk; ++i) {
                    decays[i] = decay[(tokenBase + t + i) * heads + h];
                    betas[i] = beta[(tokenBase + t + i) * heads + h];
                }
                if (localConvIsC4) {
                    packPanelC4<chunk>(q, qSrc, length, t);
                    packPanelC4<chunk>(k, kSrc, length, t);
                    packPanelC4<chunk>(v, vSrc, length, t);
                } else {
                    packPanelN<chunk>(q, qSrc + t, length);
                    packPanelN<chunk>(k, kSrc + t, length);
                    packPanelN<chunk>(v, vSrc + t, length);
                }
                for (int i = 0; i < chunk; ++i) {
                    normalizeQKRow_RVV(q + i * dk, k + i * dk, qScale, dk);
                }
                auto out = headOutAt(h, tokenBase, t);
#ifdef MNN_USE_SPACEMIT_IME2
                if (staged && MNNIme2GatedPrefill(recurrent, q, k, v, projection, out, decays, betas, headStride,
                                                  static_cast<uint8_t*>(scratch) + usedBytes, bytes - usedBytes)) {
                    continue;
                }
#endif
                MNNGatedDeltaChunk_RVV(recurrent, q, k, v, projection, out, decays, betas, headStride, false);
            }
            for (; t < length; ++t) {
                if (localConvIsC4) {
                    for (int c = 0; c < dk; c += 4) {
                        ::memcpy(q + c, qSrc + size_t(c) * length + size_t(t) * 4, 16);
                        ::memcpy(k + c, kSrc + size_t(c) * length + size_t(t) * 4, 16);
                    }
                    for (int c = 0; c < dv; c += 4) {
                        ::memcpy(v + c, vSrc + size_t(c) * length + size_t(t) * 4, 16);
                    }
                } else {
                    for (int c = 0; c < dk; ++c) {
                        q[c] = qSrc[c * length + t];
                        k[c] = kSrc[c * length + t];
                    }
                    for (int c = 0; c < dv; ++c) {
                        v[c] = vSrc[c * length + t];
                    }
                }
                const auto kq = core->MNNNormalizeQKAndDot(q, k, qScale, mParam->use_qk_l2norm, dk);
                auto out = headOutAt(h, tokenBase, t);
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
    if (chunked && length >= kRvvGatedDeltaChunk && cpu->precisionMode() == BackendConfig::Precision_Low) {
        auto task = [](size_t id, void* scratch, size_t bytes, void* ctx) {
            (*static_cast<decltype(runHeads)*>(ctx))(static_cast<int>(id), scratch, bytes);
        };
        dispatched = MNNSpacemitIme2RunTcmTasks(workers, task, &runHeads) == size_t(workers);
    }
#endif
    if (!dispatched) {
        if (fusedQkvTensor) {
            // TCM dispatch unavailable: materialize conv buffer before CPU fallback.
            convIsC4 = convolve(fusedQkvTensor, convWeight, conv, panels);
            fusedQkv = nullptr;
        }
        MNN_CONCURRENCY_BEGIN(worker, workers) {
            runHeads(worker, nullptr, 0);
        }
        MNN_CONCURRENCY_END();
    }
    const bool deliveredHeadMajor = dispatched && fusedQkv != nullptr && wantHeadMajor;
    if (deliveredHeadMajor) {
        auto& info = handoff->activeLinearAttn[outTensor];
        info.ptr = outTensor->host<float>();
        info.batch = batch;
        info.length = length;
        info.heads = heads;
        info.dv = dv;
        return;
    }
    if (handoff && handoff->backend == backend()) {
        handoff->activeLinearAttn.erase(outTensor);
    }
    if (TensorUtils::getDescribe(outTensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        const int tokens = batch * length * heads;
        int offsets[2] = {dv, tokens};
        bool packedOnAi = false;
#ifdef MNN_USE_SPACEMIT_IME2
        if (dispatched && threads <= 8) {
            struct PackCtx {
                const CoreFunctions* core;
                float* dst;
                const float* src;
                int tokens;
                int dv;
                int threads;
            } pctx{core, outTensor->host<float>(), output, tokens, dv, threads};
            auto task = [](size_t w, void*, size_t, void* data) {
                auto& p = *static_cast<PackCtx*>(data);
                const int begin = p.tokens * static_cast<int>(w) / p.threads;
                const int count = p.tokens * static_cast<int>(w + 1) / p.threads - begin;
                if (count > 0) {
                    int off[2] = {p.dv, p.tokens};
                    p.core->MNNPackCUnitTranspose(p.dst + size_t(begin) * p.core->pack, p.src + size_t(begin) * p.dv,
                                                  count, p.dv, off);
                }
            };
            packedOnAi = MNNSpacemitIme2RunTcmTasks(threads, task, &pctx) == size_t(threads);
        }
#endif
        if (!packedOnAi) {
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
