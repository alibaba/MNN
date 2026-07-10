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
                             int qHiddenPack, int kHiddenPack, float qEps, float kEps, bool qNorm, bool kNorm) {
    const int fullHead = numHead + kvNumHead;
    const int tokenHead = blockIdx.x;
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
    const int base = token * hiddenPack + head * headDim;

    __shared__ float squareSum[128];
    float localSum = 0.0f;
    if (useNorm) {
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            float value = static_cast<float>(input[base + d]);
            localSum += value * value;
        }
    }
    squareSum[threadIdx.x] = localSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            squareSum[threadIdx.x] += squareSum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = useNorm ? rsqrtf(squareSum[0] / static_cast<float>(headDim) + eps) : 1.0f;
    const int ropeDim = ropeHalfDim * 2;
    const int trigBase = token * ropeDim;

    for (int d = threadIdx.x; d < ropeHalfDim; d += blockDim.x) {
        float even = static_cast<float>(input[base + d]);
        float odd = static_cast<float>(input[base + d + ropeHalfDim]);
        if (useNorm) {
            even *= scale * gamma[d];
            odd *= scale * gamma[d + ropeHalfDim];
        }
        const float cEven = static_cast<float>(cos[trigBase + d]);
        const float cOdd = static_cast<float>(cos[trigBase + d + ropeHalfDim]);
        const float sEven = static_cast<float>(sin[trigBase + d]);
        const float sOdd = static_cast<float>(sin[trigBase + d + ropeHalfDim]);
        output[base + d] = static_cast<T>(even * cEven - odd * sEven);
        output[base + d + ropeHalfDim] = static_cast<T>(odd * cOdd + even * sOdd);
    }
    for (int d = ropeDim + threadIdx.x; d < headDim; d += blockDim.x) {
        float value = static_cast<float>(input[base + d]);
        if (useNorm) {
            value *= scale * gamma[d];
        }
        output[base + d] = static_cast<T>(value);
    }
}

class RoPEExecution : public Execution {
public:
    RoPEExecution(const Op* op, Backend* backend) : Execution(backend) {
        auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
        if (param == nullptr) {
            return;
        }
        mRopeCutHeadDim = param->rope_cut_head_dim();
        mNumHead = param->num_head();
        mKvNumHead = param->kv_num_head();
        mHeadDim = param->head_dim();
        mQNorm = prepareGamma(param->q_norm(), mQGamma, mQEps);
        mKNorm = prepareGamma(param->k_norm(), mKGamma, mKEps);
        mValid = mNumHead > 0 && mKvNumHead > 0 && mHeadDim > 0 && (param->q_norm() == nullptr || mQNorm) &&
                 (param->k_norm() == nullptr || mKNorm);
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
          mValid(source->mValid),
          mQGamma(source->mQGamma),
          mKGamma(source->mKGamma) {}

    bool onClone(Backend* backend, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        *dst = new RoPEExecution(backend, this);
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 4 || outputs.size() != 2) {
            return NOT_SUPPORT;
        }
        auto q = inputs[0];
        auto k = inputs[1];
        const bool valid = q != nullptr && k != nullptr && inputs[2] != nullptr && inputs[3] != nullptr &&
                           outputs[0] != nullptr && outputs[1] != nullptr && q->dimensions() == 4 &&
                           k->dimensions() == 4 && q->length(0) == k->length(0) &&
                           q->length(1) == mNumHead * mHeadDim && k->length(1) == mKvNumHead * mHeadDim &&
                           q->length(2) == 1 && q->length(3) == 1 && k->length(2) == 1 && k->length(3) == 1 &&
                           TensorUtils::getDescribe(q)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                           TensorUtils::getDescribe(k)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                           TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                           TensorUtils::getDescribe(outputs[1])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        if (!valid) {
            MNN_ERROR("CUDA RoPE: invalid C4 q/k layout or head configuration.\n");
            return NOT_SUPPORT;
        }
        mSeqLen = q->length(0);
        mQHiddenPack = UP_DIV(q->length(1), PACK_NUMBER) * PACK_NUMBER;
        mKHiddenPack = UP_DIV(k->length(1), PACK_NUMBER) * PACK_NUMBER;
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
        const int blocks = mSeqLen * (mNumHead + mKvNumHead);
        if (mUseFP16) {
            ropeC4Kernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(inputs[0]->deviceId()),
                reinterpret_cast<const half*>(inputs[1]->deviceId()),
                reinterpret_cast<const half*>(inputs[2]->deviceId()),
                reinterpret_cast<const half*>(inputs[3]->deviceId()), reinterpret_cast<half*>(outputs[0]->deviceId()),
                reinterpret_cast<half*>(outputs[1]->deviceId()), gammaPtr(mQGamma), gammaPtr(mKGamma), mSeqLen,
                mNumHead, mKvNumHead, mHeadDim, mRopeHalfDim, mQHiddenPack, mKHiddenPack, mQEps, mKEps, mQNorm, mKNorm);
        } else {
            ropeC4Kernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(inputs[0]->deviceId()),
                reinterpret_cast<const float*>(inputs[1]->deviceId()),
                reinterpret_cast<const float*>(inputs[2]->deviceId()),
                reinterpret_cast<const float*>(inputs[3]->deviceId()), reinterpret_cast<float*>(outputs[0]->deviceId()),
                reinterpret_cast<float*>(outputs[1]->deviceId()), gammaPtr(mQGamma), gammaPtr(mKGamma), mSeqLen,
                mNumHead, mKvNumHead, mHeadDim, mRopeHalfDim, mQHiddenPack, mKHiddenPack, mQEps, mKEps, mQNorm, mKNorm);
        }
        checkKernelErrors;
        return NO_ERROR;
    }

private:
    bool prepareGamma(const LayerNorm* norm, std::shared_ptr<Tensor>& gamma, float& eps) {
        if (norm == nullptr) {
            return false;
        }
        eps = norm->epsilon();
        if (norm->gamma() == nullptr || norm->gamma()->size() != mHeadDim || !norm->useRMSNorm()) {
            MNN_ERROR("CUDA RoPE: q/k norm must be RMSNorm with headDim gamma.\n");
            return false;
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
    bool mValid = false;
    bool mUseFP16 = false;
    std::shared_ptr<Tensor> mQGamma;
    std::shared_ptr<Tensor> mKGamma;
};

class RoPECreator : public CUDABackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op,
                        Backend* backend) const override {
        if (inputs.size() != 4 || outputs.size() != 2 ||
            TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        return new RoPEExecution(op, backend);
    }
};

static CUDACreatorRegister<RoPECreator> __init_rope(OpType_RoPE);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
