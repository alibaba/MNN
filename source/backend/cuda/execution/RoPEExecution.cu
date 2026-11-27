#include "backend/cuda/core/CUDABackend.hpp"
#include "core/TensorUtils.hpp"
#include "MNNCUDADefine.hpp"

#include <cuda_fp16.h>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

template <typename T>
__global__ void ropeC4Kernel(const T* q, const T* k, const T* cos, const T* sin, T* qOut, T* kOut, const float* qGamma,
                             const float* kGamma, int seqLen, int numHead, int kvNumHead, int headDim, int ropeHalfDim,
                             int qHiddenPack, int kHiddenPack, float qEps, float kEps, bool qNorm, bool kNorm,
                             bool interleaved) {
    const int fullHead = numHead + kvNumHead;
    const int lane = threadIdx.x & 31;
    const int tokenHead = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (tokenHead >= seqLen * fullHead) return;
    const int token = tokenHead / fullHead;
    const int combinedHead = tokenHead - token * fullHead;
    const bool isQ = combinedHead < numHead;
    const int head = isQ ? combinedHead : combinedHead - numHead;
    const int hiddenPack = isQ ? qHiddenPack : kHiddenPack;
    const T* input = isQ ? q : k;
    T* output = isQ ? qOut : kOut;
    const float* gamma = isQ ? qGamma : kGamma;
    const bool useNorm = isQ ? qNorm : kNorm;
    const float eps = isQ ? qEps : kEps;
    const bool hasGamma = gamma != nullptr;
    const int headCount = isQ ? numHead : kvNumHead;
    const int inputBase = token * hiddenPack + head * headDim;
    const int outputBase = (token * headCount + head) * headDim;

    float scale = 1.0f;
    if (useNorm) {
        float localSum = 0.0f;
        for (int d = lane; d < headDim; d += 32) {
            float value = static_cast<float>(input[inputBase + d]);
            localSum += value * value;
        }
        for (int delta = 16; delta > 0; delta >>= 1) {
            localSum += __shfl_down_sync(0xffffffff, localSum, delta);
        }
        localSum = __shfl_sync(0xffffffff, localSum, 0);
        scale = rsqrtf(localSum / static_cast<float>(headDim) + eps);
    }
    const int ropeDim = ropeHalfDim * 2;
    const int trigBase = token * ropeDim;

    if (interleaved) {
        for (int i = lane; i < ropeHalfDim; i += 32) {
            const int ch = 2 * i;
            float even = static_cast<float>(input[inputBase + ch]);
            float odd = static_cast<float>(input[inputBase + ch + 1]);
            if (useNorm) {
                even *= scale;
                odd *= scale;
                if (hasGamma) {
                    even *= gamma[ch];
                    odd *= gamma[ch + 1];
                }
            }
            const float c = static_cast<float>(cos[trigBase + ch]);
            const float s = static_cast<float>(sin[trigBase + ch]);
            output[outputBase + ch] = static_cast<T>(even * c - odd * s);
            output[outputBase + ch + 1] = static_cast<T>(even * s + odd * c);
        }
    } else {
        for (int d = lane; d < ropeHalfDim; d += 32) {
            float even = static_cast<float>(input[inputBase + d]);
            float odd = static_cast<float>(input[inputBase + d + ropeHalfDim]);
            if (useNorm) {
                even *= scale;
                odd *= scale;
                if (hasGamma) {
                    even *= gamma[d];
                    odd *= gamma[d + ropeHalfDim];
                }
            }
            const float cEven = static_cast<float>(cos[trigBase + d]);
            const float cOdd = static_cast<float>(cos[trigBase + d + ropeHalfDim]);
            const float sEven = static_cast<float>(sin[trigBase + d]);
            const float sOdd = static_cast<float>(sin[trigBase + d + ropeHalfDim]);
            output[outputBase + d] = static_cast<T>(even * cEven - odd * sEven);
            output[outputBase + d + ropeHalfDim] = static_cast<T>(odd * cOdd + even * sOdd);
        }
    }
    for (int d = ropeDim + lane; d < headDim; d += 32) {
        float value = static_cast<float>(input[inputBase + d]);
        if (useNorm) {
            value *= scale;
            if (hasGamma) {
                value *= gamma[d];
            }
        }
        output[outputBase + d] = static_cast<T>(value);
    }
}

class RoPEExecution : public Execution {
public:
    RoPEExecution(const RoPEParam* param, Backend* backend) : Execution(backend) {
        mRopeCutHeadDim = param->rope_cut_head_dim();
        mNumHead = param->num_head();
        mKvNumHead = param->kv_num_head();
        mHeadDim = param->head_dim();
        mInterleaved = param->interleaved();
        mQNorm = param->q_norm() != nullptr;
        mKNorm = param->k_norm() != nullptr;
        if (!prepareGamma(param->q_norm(), mQGamma, mQEps)) {
            mValid = false;
            return;
        }
        if (!prepareGamma(param->k_norm(), mKGamma, mKEps)) {
            if (mQGamma) {
                backend->onReleaseBuffer(mQGamma.get(), Backend::STATIC);
                mQGamma.reset();
            }
            mValid = false;
            return;
        }
    }

    RoPEExecution(Backend* backend, const RoPEExecution* source)
        : Execution(backend),
          mRopeCutHeadDim(source->mRopeCutHeadDim),
          mNumHead(source->mNumHead),
          mKvNumHead(source->mKvNumHead),
          mHeadDim(source->mHeadDim),
          mQEps(source->mQEps),
          mKEps(source->mKEps),
          mQNorm(source->mQNorm),
          mKNorm(source->mKNorm),
          mInterleaved(source->mInterleaved),
          mQGamma(source->mQGamma),
          mKGamma(source->mKGamma) {
        mValid = source->valid();
    }

    bool onClone(Backend* backend, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        *dst = new RoPEExecution(backend, this);
        return true;
    }

    // The kernel applies an optional RMSNorm with a headDim gamma before the rotation, and
    // reads no other norm state: a beta term, a non-RMS norm or a gamma of another length is
    // something the CPU kernel can execute but this one cannot express. Those cases have to be
    // rejected while the execution is created, otherwise resize() reports NOT_SUPPORT on a
    // pipeline stage that no longer falls back.
    static bool normSupported(const LayerNorm* norm, int headDim) {
        if (norm == nullptr) {
            return true;
        }
        if (!norm->useRMSNorm()) {
            return false;
        }
        if (norm->gamma() != nullptr && norm->gamma()->size() != 0 && norm->gamma()->size() != headDim) {
            return false;
        }
        if (norm->beta() != nullptr) {
            for (size_t i = 0; i < norm->beta()->size(); ++i) {
                if (norm->beta()->data()[i] != 0.0f) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool paramSupported(const RoPEParam* param) {
        if (param == nullptr || param->norm_after()) {
            return false;
        }
        if (param->num_head() <= 0 || param->kv_num_head() <= 0 || param->head_dim() <= 0) {
            return false;
        }
        return normSupported(param->q_norm(), param->head_dim()) && normSupported(param->k_norm(), param->head_dim());
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 4 || outputs.size() != 2) {
            return NOT_SUPPORT;
        }
        auto q = inputs[0];
        auto k = inputs[1];
        const bool valid = q != nullptr && k != nullptr && inputs[2] != nullptr && inputs[3] != nullptr &&
                           outputs[0] != nullptr && outputs[1] != nullptr && q->dimensions() == 4 &&
                           k->dimensions() == 4 && q->length(0) == k->length(0) &&
                           q->length(1) == mNumHead * mHeadDim && k->length(1) == mKvNumHead * mHeadDim &&
                           q->length(2) == 1 && q->length(3) == 1 && k->length(2) == 1 && k->length(3) == 1 &&
                           (TensorUtils::getDescribe(q)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ||
                            TensorUtils::getDescribe(q)->dimensionFormat == MNN_DATA_FORMAT_NCHW) &&
                           (TensorUtils::getDescribe(k)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ||
                            TensorUtils::getDescribe(k)->dimensionFormat == MNN_DATA_FORMAT_NCHW) &&
                           TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC &&
                           TensorUtils::getDescribe(outputs[1])->dimensionFormat == MNN_DATA_FORMAT_NHWC;
        if (!valid) {
            MNN_ERROR("CUDA RoPE: invalid q/k layout or head configuration.\n");
            return NOT_SUPPORT;
        }
        mSeqLen = q->length(0);
        // With unit spatial dimensions, both layouts store each token contiguously.
        mQHiddenPack = TensorUtils::getDescribe(q)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4
                           ? UP_DIV(q->length(1), PACK_NUMBER) * PACK_NUMBER
                           : q->length(1);
        mKHiddenPack = TensorUtils::getDescribe(k)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4
                           ? UP_DIV(k->length(1), PACK_NUMBER) * PACK_NUMBER
                           : k->length(1);
        int ropeDim = mRopeCutHeadDim;
        if (ropeDim <= 0 || ropeDim > mHeadDim) {
            ropeDim = mHeadDim;
        }
        ropeDim = ropeDim / 2 * 2;
        if (ropeDim <= 0 || inputs[2]->elementSize() < mSeqLen * ropeDim ||
            inputs[3]->elementSize() < mSeqLen * ropeDim) {
            MNN_ERROR("CUDA RoPE: invalid rotary table shape.\n");
            return NOT_SUPPORT;
        }
        mRopeHalfDim = ropeDim / 2;
        mUseFP16 = static_cast<CUDABackend*>(backend())->useFp16();
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        constexpr int threads = 128;
        const int blocks = UP_DIV(mSeqLen * (mNumHead + mKvNumHead), threads / 32);
        if (mUseFP16) {
            ropeC4Kernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(inputs[0]->deviceId()),
                reinterpret_cast<const half*>(inputs[1]->deviceId()),
                reinterpret_cast<const half*>(inputs[2]->deviceId()),
                reinterpret_cast<const half*>(inputs[3]->deviceId()), reinterpret_cast<half*>(outputs[0]->deviceId()),
                reinterpret_cast<half*>(outputs[1]->deviceId()), gammaPtr(mQGamma), gammaPtr(mKGamma), mSeqLen,
                mNumHead, mKvNumHead, mHeadDim, mRopeHalfDim, mQHiddenPack, mKHiddenPack, mQEps, mKEps, mQNorm, mKNorm,
                mInterleaved);
        } else {
            ropeC4Kernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(inputs[0]->deviceId()),
                reinterpret_cast<const float*>(inputs[1]->deviceId()),
                reinterpret_cast<const float*>(inputs[2]->deviceId()),
                reinterpret_cast<const float*>(inputs[3]->deviceId()), reinterpret_cast<float*>(outputs[0]->deviceId()),
                reinterpret_cast<float*>(outputs[1]->deviceId()), gammaPtr(mQGamma), gammaPtr(mKGamma), mSeqLen,
                mNumHead, mKvNumHead, mHeadDim, mRopeHalfDim, mQHiddenPack, mKHiddenPack, mQEps, mKEps, mQNorm, mKNorm,
                mInterleaved);
        }
        checkKernelErrors;
        return NO_ERROR;
    }

private:
    bool prepareGamma(const LayerNorm* norm, std::shared_ptr<Tensor>& gamma, float& eps) {
        if (norm == nullptr) {
            return true;
        }
        eps = norm->epsilon();
        // normSupported() has already bound the norm kind and the gamma length to headDim.
        if (norm->gamma() == nullptr || norm->gamma()->size() == 0) {
            return true;
        }
        gamma.reset(Tensor::createDevice<int32_t>({mHeadDim}));
        if (!backend()->onAcquireBuffer(gamma.get(), Backend::STATIC)) {
            gamma.reset();
            return false;
        }
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        runtime->memcpy(reinterpret_cast<void*>(gamma->deviceId()), norm->gamma()->data(), mHeadDim * sizeof(float),
                        MNNMemcpyHostToDevice, true);
        return true;
    }

    static const float* gammaPtr(const std::shared_ptr<Tensor>& gamma) {
        return gamma ? reinterpret_cast<const float*>(gamma->deviceId()) : nullptr;
    }

    int mRopeCutHeadDim = 0;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;
    int mSeqLen = 0;
    int mRopeHalfDim = 0;
    int mQHiddenPack = 0;
    int mKHiddenPack = 0;
    float mQEps = 0.0f;
    float mKEps = 0.0f;
    bool mQNorm = false;
    bool mKNorm = false;
    bool mInterleaved = false;
    bool mUseFP16 = false;
    std::shared_ptr<Tensor> mQGamma;
    std::shared_ptr<Tensor> mKGamma;
};

class RoPECreator : public CUDABackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op,
                        Backend* backend) const override {
        if (inputs.size() != 4 || outputs.size() != 2 ||
            (TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 &&
             TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NCHW) ||
            (TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 &&
             TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NCHW)) {
            return nullptr;
        }
        auto param = op->main_as_RoPEParam();
        if (!RoPEExecution::paramSupported(param)) {
            return nullptr;
        }
        return new RoPEExecution(param, backend);
    }
};

static CUDACreatorRegister<RoPECreator> __init_rope(OpType_RoPE);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
