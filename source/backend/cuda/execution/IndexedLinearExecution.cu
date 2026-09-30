//
//  IndexedLinearExecution.cu
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/IndexedLinearCommon.hpp"
#include "core/TensorUtils.hpp"
#include <cuda_fp16.h>
#include <mma.h>
#include <climits>

namespace MNN {
namespace CUDA {
namespace {

struct IndexedLinearParams {
    int inputChannels, outputChannels, experts, topK, routes, inputMode, fuseSwiGLU;
    int blockSize, blockCount, weightStride;
};

struct IndexedLinearResource {
    std::shared_ptr<Tensor> weight, scale, bias;
    IndexedLinearParams params = {};
    int bits = 0;
    int device = 0;
    BufferAllocator* staticAllocator = nullptr;
    bool fp16 = false;
};

static std::shared_ptr<Tensor> allocIndexedBytes(Backend* backend, size_t bytes, Backend::StorageType storage) {
    if (bytes == 0 || bytes > INT_MAX)
        return nullptr;
    std::shared_ptr<Tensor> tensor(Tensor::createDevice<uint8_t>({static_cast<int>(bytes)}));
    if (!backend->onAcquireBuffer(tensor.get(), storage))
        return nullptr;
    return tensor;
}

static std::shared_ptr<IndexedLinearResource> loadIndexedResource(const Op* op, CUDABackend* backend) {
    auto res = std::make_shared<IndexedLinearResource>();
    auto& p = res->params;
    if (!IndexedLinearCommon::dimensions(op, p.experts, p.inputChannels, p.outputChannels))
        return nullptr;
    res->fp16 = backend->useFp16();
    auto runtime = backend->getCUDARuntime();
    res->device = runtime->device_id();
    res->staticAllocator = backend->getStaticBufferPool();
    p.inputMode = op->main_as_IndexedLinearParam()->inputMode();
    p.fuseSwiGLU = (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU);
    const size_t elements = static_cast<size_t>(p.inputChannels) * p.outputChannels;
    if (elements > INT_MAX)
        return nullptr;
    std::vector<float> bias, scales;
    std::vector<half> halfWeight;
    for (int e = 0; e < p.experts; ++e) {
        auto weight = IndexedLinearCommon::loadLinear(op, e, backend);
        if (!weight || !IndexedLinearCommon::loadBias(op, e, bias))
            return nullptr;
        if (weight->canUseInt2 || weight->canUseInt3)
            return nullptr;
        const bool quantized = weight->weight.get() != nullptr;
        const int bits = quantized ? (weight->canUseInt4 ? 4 : 8) : 0;
        int blockCount = 1;
        const float* alpha = nullptr;
        if (quantized) {
            alpha = weight->getAlphaFloat();
            const int divisor = p.outputChannels * (weight->asymmetric ? 2 : 1);
            if (!alpha || weight->alphaSize <= 0 || weight->alphaSize % divisor)
                return nullptr;
            blockCount = weight->alphaSize / divisor;
            if (p.inputChannels % blockCount)
                return nullptr;
        } else if (!weight->weightFloat.get() || weight->weightFloat.size() < elements) {
            return nullptr;
        }
        const size_t stride = bits ? ((elements * bits + 7) / 8) : elements * (res->fp16 ? 2 : 4);
        if (stride > INT_MAX || (quantized && weight->weight.size() < stride))
            return nullptr;
        if (e == 0) {
            res->bits = bits;
            p.blockCount = blockCount;
            p.blockSize = p.inputChannels / blockCount;
            p.weightStride = static_cast<int>(stride);
            res->weight = allocIndexedBytes(backend, stride * p.experts, Backend::STATIC);
            res->scale = allocIndexedBytes(
                backend, static_cast<size_t>(p.experts) * p.outputChannels * blockCount * sizeof(float2),
                Backend::STATIC);
            res->bias = allocIndexedBytes(backend, static_cast<size_t>(p.experts) * p.outputChannels * sizeof(float),
                                          Backend::STATIC);
            if (!res->weight || !res->scale || !res->bias)
                return nullptr;
            scales.resize(static_cast<size_t>(p.outputChannels) * blockCount * 2);
        } else if (bits != res->bits || blockCount != p.blockCount || stride != p.weightStride) {
            return nullptr;
        }
        const void* source = weight->weight.get();
        if (!quantized) {
            source = weight->weightFloat.get();
            if (res->fp16) {
                halfWeight.resize(elements);
                for (size_t i = 0; i < elements; ++i)
                    halfWeight[i] = __float2half(weight->weightFloat.get()[i]);
                source = halfWeight.data();
            }
        }
        // The staging vectors and decoded source live only for this iteration.
        // These uploads happen once at creation, never in the inference stream.
        runtime->memcpy(reinterpret_cast<uint8_t*>(res->weight->deviceId()) + static_cast<size_t>(e) * stride, source,
                        stride, MNNMemcpyHostToDevice, true);
        for (int i = 0; i < p.outputChannels * blockCount; ++i) {
            scales[2 * i] = quantized ? alpha[weight->asymmetric ? 2 * i + 1 : i] : 1.0f;
            scales[2 * i + 1] = quantized && weight->asymmetric ? alpha[2 * i] : 0.0f;
        }
        runtime->memcpy(reinterpret_cast<float*>(res->scale->deviceId()) + static_cast<size_t>(e) * scales.size(),
                        scales.data(), scales.size() * sizeof(float), MNNMemcpyHostToDevice, true);
        runtime->memcpy(reinterpret_cast<float*>(res->bias->deviceId()) + static_cast<size_t>(e) * p.outputChannels,
                        bias.data(), p.outputChannels * sizeof(float), MNNMemcpyHostToDevice, true);
    }
    return res;
}

template <typename T, int Bits>
__device__ __forceinline__ float indexedWeight(const uint8_t* weights, const float2* scales,
                                               const IndexedLinearParams& p, int expert, int output, int input) {
    const size_t index = static_cast<size_t>(output) * p.inputChannels + input;
    const uint8_t* bank = weights + static_cast<size_t>(expert) * p.weightStride;
    if (Bits == 0)
        return static_cast<float>(reinterpret_cast<const T*>(bank)[index]);
    int q;
    if (Bits == 4) {
        const uint8_t packed = bank[index / 2];
        q = ((index & 1) ? (packed & 15) : (packed >> 4)) - 8;
    } else {
        q = reinterpret_cast<const int8_t*>(bank)[index];
    }
    const float2 scale =
        scales[(static_cast<size_t>(expert) * p.outputChannels + output) * p.blockCount + input / p.blockSize];
    return static_cast<float>(q) * scale.x + scale.y;
}

template <typename T>
__device__ T indexedSwiGLU(float gate, float up) {
    const float g = static_cast<float>(static_cast<T>(gate));
    // Match Raster's Unary SILU, including its finite-range guard and FP32 cast.
    const float silu = g > 87. ? g : (g < -87. ? 0.0f : g / (1. + exp(-g)));
    const T activated = static_cast<T>(silu);
    return static_cast<T>(static_cast<float>(activated) * static_cast<float>(static_cast<T>(up)));
}

template <typename T, int Bits, bool Fused>
__global__ void indexedGemv(const T* input, const int* ids, T* output, IndexedLinearParams p, const uint8_t* weights,
                            const float2* scales, const float* bias, int outputTiles) {
    const int route = blockIdx.x / outputTiles;
    const int lane = threadIdx.x & 31;
    const int firstOutput = ((blockIdx.x % outputTiles) * 4 + threadIdx.x / 32) * 4;
    const int resultChannels = p.outputChannels / (Fused ? 2 : 1);
    if (firstOutput >= resultChannels)
        return; // Warp-uniform.
    const int expert = ids[route];
    const bool valid = expert >= 0 && expert < p.experts;
    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
    const T* row = input + static_cast<size_t>(sourceRow) * p.inputChannels;
    float sum[4] = {}, upSum[4] = {};
    if (valid) {
        for (int i = lane * 4; i < p.inputChannels; i += 128) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (i + j >= p.inputChannels)
                    break;
                const float x = static_cast<float>(row[i + j]);
#pragma unroll
                for (int o = 0; o < 4; ++o) {
                    if (firstOutput + o < resultChannels) {
                        const int channel = (firstOutput + o) * (Fused ? 2 : 1);
                        sum[o] += x * indexedWeight<T, Bits>(weights, scales, p, expert, channel, i + j);
                        if (Fused)
                            upSum[o] += x * indexedWeight<T, Bits>(weights, scales, p, expert, channel + 1, i + j);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int o = 0; o < 4; ++o) {
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            sum[o] += __shfl_down_sync(0xffffffff, sum[o], delta);
            if (Fused)
                upSum[o] += __shfl_down_sync(0xffffffff, upSum[o], delta);
        }
        if (lane == 0 && firstOutput + o < resultChannels) {
            const size_t channel = static_cast<size_t>(expert) * p.outputChannels + (firstOutput + o) * (Fused ? 2 : 1);
            const float b = valid ? bias[channel] : 0.0f;
            output[static_cast<size_t>(route) * resultChannels + firstOutput + o] =
                Fused ? indexedSwiGLU<T>(sum[o] + b, upSum[o] + (valid ? bias[channel + 1] : 0.0f))
                      : static_cast<T>(sum[o] + b);
        }
    }
}

__global__ void indexedClear(int* counts, int experts) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < experts)
        counts[e] = 0;
}

__global__ void indexedHistogram(const int* ids, int* counts, int* ranks, int experts, int routes) {
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes)
        return;
    const int e = ids[route];
    ranks[route] = e >= 0 && e < experts ? atomicAdd(counts + e, 1) : -1;
}

__global__ void indexedPrefix(const int* counts, int* offsets, int* tileOffsets, int experts) {
    int routes = 0, tiles = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e] = routes;
        tileOffsets[e] = tiles;
        routes += counts[e];
        tiles += (counts[e] + 31) / 32;
    }
    offsets[experts] = routes;
    tileOffsets[experts] = tiles;
}

template <typename T>
__global__ void indexedScatter(const int* ids, const int* ranks, const int* offsets, int* order, T* output, int routes,
                               int channels) {
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes)
        return;
    if (ranks[route] < 0) {
        for (int o = 0; o < channels; ++o)
            output[static_cast<size_t>(route) * channels + o] = static_cast<T>(0.0f);
        return;
    }
    order[offsets[ids[route]] + ranks[route]] = route;
}

// 32x32x32 tile. B is staged column-major so consecutive loader lanes read
// consecutive compressed input channels of one expert/output channel.
template <typename T, int Bits, bool Matrix, bool Fused>
__global__ void indexedGrouped(const T* input, T* output, IndexedLinearParams p, const uint8_t* weights,
                               const float2* scales, const float* bias, const int* offsets, const int* tileOffsets,
                               const int* order, int outputTiles) {
    static_assert(!Matrix || sizeof(T) == sizeof(half), "WMMA inputs must use FP16 storage");
    const int tile = blockIdx.x / outputTiles;
    if (tile >= tileOffsets[p.experts])
        return; // Block-uniform, before barriers.
    int low = 0, high = p.experts;
    while (low + 1 < high) {
        const int mid = (low + high) / 2;
        if (tileOffsets[mid] <= tile)
            low = mid;
        else
            high = mid;
    }
    const int expert = low;
    const int firstRow = offsets[expert] + (tile - tileOffsets[expert]) * 32;
    const int endRow = offsets[expert + 1];
    const int firstOutput = (blockIdx.x % outputTiles) * 32;
    const int tid = threadIdx.x;
    __shared__ __align__(32) T aTile[32 * 32];
    __shared__ __align__(32) T bTile[32 * 32];
    __shared__ __align__(32) float cTile[32 * 32];
    float sums[8] = {};
#if __CUDA_ARCH__ >= 700
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accum;
    if (Matrix)
        nvcuda::wmma::fill_fragment(accum, 0.0f);
#endif
    for (int firstInput = 0; firstInput < p.inputChannels; firstInput += 32) {
        for (int i = tid; i < 1024; i += 128) {
            const int rowIndex = firstRow + i / 32;
            const int k = firstInput + i % 32;
            T a = static_cast<T>(0.0f);
            if (rowIndex < endRow && k < p.inputChannels) {
                const int route = order[rowIndex];
                const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
                a = input[static_cast<size_t>(sourceRow) * p.inputChannels + k];
            }
            aTile[i] = a;
            const int o = firstOutput + i / 32;
            bTile[i] = k < p.inputChannels && o < p.outputChannels
                           ? static_cast<T>(indexedWeight<T, Bits>(weights, scales, p, expert, o, k))
                           : static_cast<T>(0.0f);
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 700
        if (Matrix) {
            const int warp = tid / 32;
            for (int k = 0; k < 32; k += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b;
                nvcuda::wmma::load_matrix_sync(a, reinterpret_cast<const half*>(aTile) + (warp / 2) * 16 * 32 + k, 32);
                nvcuda::wmma::load_matrix_sync(b, reinterpret_cast<const half*>(bTile) + (warp % 2) * 16 * 32 + k, 32);
                nvcuda::wmma::mma_sync(accum, a, b, accum);
            }
        } else
#endif
        {
            const int col = tid % 32;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float b = static_cast<float>(bTile[col * 32 + k]);
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    sums[r] += static_cast<float>(aTile[(tid / 32 + r * 4) * 32 + k]) * b;
                }
            }
        }
        __syncthreads();
    }
#if __CUDA_ARCH__ >= 700
    if (Matrix) {
        const int warp = tid / 32;
        nvcuda::wmma::store_matrix_sync(cTile + (warp / 2) * 16 * 32 + (warp % 2) * 16, accum, 32,
                                        nvcuda::wmma::mem_row_major);
        __syncthreads();
    }
#endif
    if (Fused) {
#if __CUDA_ARCH__ >= 700
        if (!Matrix) {
#endif
            for (int r = 0; r < 8; ++r)
                cTile[tid + r * 128] = sums[r];
            __syncthreads();
#if __CUDA_ARCH__ >= 700
        }
#endif
        for (int r = 0; r < 4; ++r) {
            const int i = tid + r * 128;
            const int rowIndex = firstRow + i / 16;
            const int channel = firstOutput + i % 16 * 2;
            if (rowIndex < endRow && channel < p.outputChannels) {
                const int index = i / 16 * 32 + i % 16 * 2;
                const size_t b = static_cast<size_t>(expert) * p.outputChannels + channel;
                output[static_cast<size_t>(order[rowIndex]) * (p.outputChannels / 2) + channel / 2] =
                    indexedSwiGLU<T>(cTile[index] + bias[b], cTile[index + 1] + bias[b + 1]);
            }
        }
        return;
    }
    for (int r = 0; r < 8; ++r) {
        const int i = tid + r * 128;
        const int rowIndex = firstRow + i / 32;
        const int o = firstOutput + i % 32;
        if (rowIndex < endRow && o < p.outputChannels) {
            float value = sums[r];
#if __CUDA_ARCH__ >= 700
            if (Matrix)
                value = cTile[i];
#endif
            output[static_cast<size_t>(order[rowIndex]) * p.outputChannels + o] =
                static_cast<T>(value + bias[static_cast<size_t>(expert) * p.outputChannels + o]);
        }
    }
}

} // namespace

class IndexedLinearExecution : public Execution {
public:
    IndexedLinearExecution(Backend* backend, std::shared_ptr<IndexedLinearResource> resource)
        : Execution(backend), mResource(std::move(resource)), mParams(mResource->params) {}

    bool onClone(Backend* backend, const Op* op, Execution** dst) override {
        if (!dst)
            return true;
        if (!backend || backend->type() != MNN_FORWARD_CUDA)
            return false;
        int experts, inputChannels, outputChannels;
        if (!IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels) ||
            experts != mResource->params.experts || inputChannels != mResource->params.inputChannels ||
            outputChannels != mResource->params.outputChannels ||
            op->main_as_IndexedLinearParam()->inputMode() != mResource->params.inputMode ||
            (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) !=
                bool(mResource->params.fuseSwiGLU))
            return false;
        auto cuda = static_cast<CUDABackend*>(backend);
        if (cuda->getPrecision() == 3 || cuda->useFp16() != mResource->fp16 ||
            cuda->getCUDARuntime()->device_id() != mResource->device ||
            cuda->getStaticBufferPool() != mResource->staticAllocator)
            return false;
        *dst = new IndexedLinearExecution(backend, mResource);
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 2 || outputs.size() != 1 || inputs[1]->dimensions() != 2 ||
            inputs[1]->getType() != halide_type_of<int32_t>() || inputs[0]->getType() != halide_type_of<float>() ||
            outputs[0]->getType() != halide_type_of<float>())
            return NOT_SUPPORT;
        for (auto t : {inputs[0], inputs[1], outputs[0]}) {
            if (TensorUtils::getDescribe(t)->dimensionFormat != MNN_DATA_FORMAT_NCHW)
                return NOT_SUPPORT;
            int64_t stride = 1;
            for (int d = t->dimensions() - 1; d >= 0; --d) {
                if (t->length(d) < 0 || t->stride(d) != stride)
                    return INPUT_DATA_ERROR;
                stride *= t->length(d);
                if (stride > INT_MAX)
                    return INPUT_DATA_ERROR;
            }
        }
        auto cuda = static_cast<CUDABackend*>(backend());
        const int bytes = mResource->fp16 ? 2 : 4;
        if (cuda->getBytes(inputs[0]) != bytes || cuda->getBytes(outputs[0]) != bytes ||
            cuda->getBytes(inputs[1]) != sizeof(int32_t))
            return NOT_SUPPORT;
        const int tokens = inputs[1]->length(0), topK = inputs[1]->length(1);
        if (tokens <= 0 || topK <= 0 || static_cast<int64_t>(tokens) * topK > INT_MAX)
            return INPUT_DATA_ERROR;
        mParams.topK = topK;
        mParams.routes = tokens * topK;
        if (inputs[0]->dimensions() != (mParams.inputMode == 0 ? 2 : 3) || inputs[0]->length(0) != tokens ||
            (mParams.inputMode == 1 && inputs[0]->length(1) != topK) ||
            inputs[0]->length(inputs[0]->dimensions() - 1) != mParams.inputChannels || outputs[0]->dimensions() != 3 ||
            outputs[0]->length(0) != tokens || outputs[0]->length(1) != topK ||
            outputs[0]->length(2) != mParams.outputChannels / (mParams.fuseSwiGLU ? 2 : 1))
            return INPUT_DATA_ERROR;
        mGrouped = tokens > 1;
        const int tile = mGrouped ? 32 : 16;
        const int gridChannels = mParams.outputChannels / (!mGrouped && mParams.fuseSwiGLU ? 2 : 1);
        mOutputTiles = (static_cast<int64_t>(gridChannels) + tile - 1) / tile;
        const int64_t routeTiles =
            mGrouped ? (static_cast<int64_t>(mParams.routes) + 31) / 32 + mParams.experts : mParams.routes;
        const int64_t blocks = routeTiles * mOutputTiles;
        if (blocks > INT_MAX || blocks > cuda->getCUDARuntime()->prop().maxGridSize[0])
            return NOT_SUPPORT;
        mBlocks = static_cast<int>(blocks);
        if (mGrouped) {
            const size_t sizes[] = {static_cast<size_t>(mParams.experts), static_cast<size_t>(mParams.routes),
                                    static_cast<size_t>(mParams.experts) + 1, static_cast<size_t>(mParams.experts) + 1,
                                    static_cast<size_t>(mParams.routes)};
            for (int i = 0; i < 5; ++i) {
                mWork[i] = allocIndexedBytes(backend(), sizes[i] * sizeof(int), Backend::DYNAMIC);
                if (!mWork[i])
                    return OUT_OF_MEMORY;
            }
            for (auto& work : mWork)
                backend()->onReleaseBuffer(work.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (mResource->fp16)
            dispatch<half>(inputs, outputs);
        else
            dispatch<float>(inputs, outputs);
        checkKernelErrors;
        return NO_ERROR;
    }

private:
    template <typename T>
    void dispatch(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        if (mParams.fuseSwiGLU)
            dispatchMode<T, true>(inputs, outputs);
        else
            dispatchMode<T, false>(inputs, outputs);
    }

    template <typename T, bool Fused>
    void dispatchMode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        if (mResource->bits == 4)
            run<T, 4, Fused>(inputs, outputs);
        else if (mResource->bits == 8)
            run<T, 8, Fused>(inputs, outputs);
        else
            run<T, 0, Fused>(inputs, outputs);
    }

    template <typename T, int Bits, bool Fused>
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        const T* input = reinterpret_cast<const T*>(inputs[0]->deviceId());
        const int* ids = reinterpret_cast<const int*>(inputs[1]->deviceId());
        T* output = reinterpret_cast<T*>(outputs[0]->deviceId());
        const uint8_t* weights = reinterpret_cast<const uint8_t*>(mResource->weight->deviceId());
        const float2* scales = reinterpret_cast<const float2*>(mResource->scale->deviceId());
        const float* bias = reinterpret_cast<const float*>(mResource->bias->deviceId());
        if (!mGrouped) {
            indexedGemv<T, Bits, Fused>
                <<<mBlocks, 128>>>(input, ids, output, mParams, weights, scales, bias, mOutputTiles);
            return;
        }
        int* work[5];
        for (int i = 0; i < 5; ++i)
            work[i] = reinterpret_cast<int*>(mWork[i]->deviceId());
        indexedClear<<<UP_DIV(mParams.experts, 128), 128>>>(work[0], mParams.experts);
        indexedHistogram<<<UP_DIV(mParams.routes, 128), 128>>>(ids, work[0], work[1], mParams.experts, mParams.routes);
        indexedPrefix<<<1, 1>>>(work[0], work[2], work[3], mParams.experts);
        indexedScatter<T><<<UP_DIV(mParams.routes, 128), 128>>>(ids, work[1], work[2], work[4], output, mParams.routes,
                                                                mParams.outputChannels / (mParams.fuseSwiGLU ? 2 : 1));
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        if (mResource->fp16 && runtime->major_sm() >= 7) {
            indexedGrouped<half, Bits, true, Fused>
                <<<mBlocks, 128>>>(reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output), mParams,
                                   weights, scales, bias, work[2], work[3], work[4], mOutputTiles);
        } else {
            indexedGrouped<T, Bits, false, Fused><<<mBlocks, 128>>>(input, output, mParams, weights, scales, bias,
                                                                    work[2], work[3], work[4], mOutputTiles);
        }
    }

    std::shared_ptr<IndexedLinearResource> mResource;
    IndexedLinearParams mParams;
    std::shared_ptr<Tensor> mWork[5];
    int mBlocks = 0;
    int mOutputTiles = 0;
    bool mGrouped = false;
};

class IndexedLinearCreator : public CUDABackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op,
                        Backend* backend) const override {
        auto cuda = static_cast<CUDABackend*>(backend);
        if (cuda->getPrecision() == 3)
            return nullptr; // BF16 needs its own typed kernel.
        auto resource = loadIndexedResource(op, cuda);
        if (!resource)
            return nullptr;
        return new IndexedLinearExecution(backend, resource);
    }
};

// Called by backend initialization. Avoid a namespace-scope registering object.
void registerCUDAIndexedLinear() {
    static const bool registered = CUDABackend::addCreator(OpType_IndexedLinear, new IndexedLinearCreator);
    (void)registered;
}

} // namespace CUDA
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
