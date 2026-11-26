//
//  MetalIndexedLinear.mm
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalExecution.hpp"
#import "backend/metal/MetalIndexedLinearShader.hpp"
#import "backend/metal/MNNMetalContext.h"
#include "core/IndexedLinearCommon.hpp"
#include "core/TensorUtils.hpp"
#include <climits>
#include <cstring>

#if MNN_METAL_ENABLED
#if defined(MNN_METAL_TENSOR) && ((defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000) || \
                                (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 260000))
#define MNN_METAL_INDEXED_TENSOR_AVAILABLE
#endif

namespace MNN {
namespace {

struct IndexedLinearResource {
    std::shared_ptr<Tensor> weight, scale, bias;
    int experts = 0;
    int inputChannels = 0;
    int outputChannels = 0;
    int bits = 0;
    int blockSize = 0;
    int blockCount = 0;
    int weightStride = 0; // Bytes per expert, including a possible final packed nibble.
    bool fp16 = false;
    bool fuseSwiGLU = false;
};

static std::shared_ptr<Tensor> allocIndexedBytes(Backend* backend, size_t bytes, Backend::StorageType storage) {
    if (bytes == 0 || bytes > INT_MAX)
        return nullptr;
    std::shared_ptr<Tensor> tensor(Tensor::createDevice<uint8_t>({static_cast<int>(bytes)}));
    if (!backend->onAcquireBuffer(tensor.get(), storage))
        return nullptr;
    return tensor;
}

static uint8_t* indexedContents(const std::shared_ptr<Tensor>& tensor) {
    auto buffer = MetalBackend::getBuffer(tensor.get());
    return reinterpret_cast<uint8_t*>(buffer.first.contents) + buffer.second;
}

static void packIndexedWeights(uint8_t* dst, const uint8_t* src, int inputChannels, int outputChannels, int bits,
                               bool fuseSwiGLU) {
    const int resultChannels = fuseSwiGLU ? outputChannels / 2 : outputChannels;
    const int streams = fuseSwiGLU ? 2 : 1;
    const int resultQuads = UP_DIV(resultChannels, 4);
    const int inputQuads = UP_DIV(inputChannels, 4);
    const int tileBytes = bits == 4 ? 8 : 16;
    for (int stream = 0; stream < streams; ++stream) {
        auto streamDst = dst + stream * resultQuads * inputQuads * tileBytes;
        for (int quad = 0; quad < resultQuads; ++quad) {
            for (int inputQuad = 0; inputQuad < inputQuads; ++inputQuad) {
                auto tile = streamDst + (quad * inputQuads + inputQuad) * tileBytes;
                ::memset(tile, 0, tileBytes);
                for (int row = 0; row < 4; ++row) {
                    const int resultChannel = quad * 4 + row;
                    if (resultChannel >= resultChannels) {
                        continue;
                    }
                    const int weightChannel = fuseSwiGLU ? resultChannel * 2 + stream : resultChannel;
                    for (int inner = 0; inner < 4; ++inner) {
                        const int inputChannel = inputQuad * 4 + inner;
                        if (inputChannel >= inputChannels) {
                            continue;
                        }
                        const int index = weightChannel * inputChannels + inputChannel;
                        if (bits == 4) {
                            const uint8_t packed = src[index / 2];
                            const uint8_t nibble = index & 1 ? (packed & 15) : (packed >> 4);
                            tile[row * 2 + inner / 2] |= nibble << (4 * (1 - inner % 2));
                        } else {
                            tile[row * 4 + inner] = src[index];
                        }
                    }
                }
            }
        }
    }
}

static std::shared_ptr<IndexedLinearResource> loadIndexedResource(const Op* op, MetalBackend* backend) {
    auto res = std::make_shared<IndexedLinearResource>();
    if (!IndexedLinearCommon::dimensions(op, res->experts, res->inputChannels, res->outputChannels))
        return nullptr;
    res->fuseSwiGLU = op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU;
    const int resultChannels = res->fuseSwiGLU ? res->outputChannels / 2 : res->outputChannels;
    res->fp16 = backend->useFp16InsteadFp32();
    const size_t elements = static_cast<size_t>(res->inputChannels) * res->outputChannels;
    if (elements > INT_MAX)
        return nullptr;
    std::vector<float> bias;
    for (int e = 0; e < res->experts; ++e) {
        auto weight = IndexedLinearCommon::loadLinear(op, e, backend);
        if (!weight || !IndexedLinearCommon::loadBias(op, e, bias))
            return nullptr;
        // W2/W3 use a different source packing contract and are not supported.
        if (weight->canUseInt2 || weight->canUseInt3)
            return nullptr;
        const bool quantized = weight->weight.get() != nullptr;
        const int bits = quantized ? (weight->canUseInt4 ? 4 : 8) : 0;
        int blockCount = 1;
        const float* alpha = nullptr;
        if (quantized) {
            alpha = weight->getAlphaFloat();
            const int divisor = res->outputChannels * (weight->asymmetric ? 2 : 1);
            if (!alpha || weight->alphaSize <= 0 || weight->alphaSize % divisor)
                return nullptr;
            blockCount = weight->alphaSize / divisor;
            if (res->inputChannels % blockCount)
                return nullptr;
        } else if (!weight->weightFloat.get() || weight->weightFloat.size() < elements) {
            return nullptr;
        }
        const size_t stride =
            bits ? static_cast<size_t>((bits == 4 ? 8 : 16)) * (static_cast<size_t>(resultChannels + 3) / 4) *
                       ((static_cast<size_t>(res->inputChannels) + 3) / 4) * (res->fuseSwiGLU ? 2 : 1)
                 : elements * (res->fp16 ? 2 : 4);
        if (stride > INT_MAX || (quantized && weight->weight.size() < (elements * bits + 7) / 8))
            return nullptr;
        if (e == 0) {
            res->bits = bits;
            res->blockCount = blockCount;
            res->blockSize = res->inputChannels / blockCount;
            if (quantized && res->blockSize % 4 != 0)
                return nullptr;
            res->weightStride = static_cast<int>(stride);
            res->weight = allocIndexedBytes(backend, stride * res->experts, Backend::STATIC);
            // FP32 metadata is allocated in bytes so fp16 tensor storage never halves it.
            res->scale = allocIndexedBytes(
                backend, static_cast<size_t>(res->experts) * res->outputChannels * blockCount * 2 * sizeof(float),
                Backend::STATIC);
            res->bias = allocIndexedBytes(
                backend, static_cast<size_t>(res->experts) * res->outputChannels * sizeof(float), Backend::STATIC);
            if (!res->weight || !res->scale || !res->bias)
                return nullptr;
        } else if (bits != res->bits || blockCount != res->blockCount || stride != res->weightStride) {
            return nullptr;
        }
        auto dst = indexedContents(res->weight) + static_cast<size_t>(e) * stride;
        if (quantized) {
            packIndexedWeights(dst, reinterpret_cast<const uint8_t*>(weight->weight.get()), res->inputChannels,
                               res->outputChannels, bits, res->fuseSwiGLU);
        } else if (res->fp16) {
            auto half = reinterpret_cast<__fp16*>(dst);
            for (size_t i = 0; i < elements; ++i)
                half[i] = weight->weightFloat.get()[i];
        } else {
            ::memcpy(dst, weight->weightFloat.get(), stride);
        }
        auto scales = reinterpret_cast<float*>(indexedContents(res->scale)) +
                      static_cast<size_t>(e) * res->outputChannels * blockCount * 2;
        for (int i = 0; i < res->outputChannels * blockCount; ++i) {
            scales[2 * i] = quantized ? alpha[weight->asymmetric ? 2 * i + 1 : i] : 1.0f;
            scales[2 * i + 1] = quantized && weight->asymmetric ? alpha[2 * i] : 0.0f;
        }
        ::memcpy(indexedContents(res->bias) + static_cast<size_t>(e) * res->outputChannels * sizeof(float), bias.data(),
                 res->outputChannels * sizeof(float));
    }
    return res;
}
} // namespace

class MetalIndexedLinear : public MetalExecution {
    enum Kernel {
        Gemv,
        PackedGemv,
        Clear,
        Histogram,
        Prefix,
        Scatter,
        GroupedSimd,
        GroupedTensor,
        WeightedGemv,
        WeightedReduce,
        Route,
        ZeroY,
        PermuteA,
        KernelCount
    };

public:
    MetalIndexedLinear(Backend* backend, std::shared_ptr<IndexedLinearResource> resource, int inputMode,
                       bool fuseSwiGLU, int routeWeight, float routeEps, float routeScale)
        : MetalExecution(backend), mResource(std::move(resource)), mInputMode(inputMode), mFuseSwiGLU(fuseSwiGLU),
          mRouteWeight(routeWeight), mRouteEps(routeEps), mRouteScale(routeScale) {}

    bool onClone(Backend* backend, const Op* op, Execution** dst) override {
        if (!dst)
            return true;
        if (!backend || backend->type() != MNN_FORWARD_METAL ||
            static_cast<MetalBackend*>(backend)->useFp16InsteadFp32() != mResource->fp16)
            return false;
        int experts, inputChannels, outputChannels;
        if (!IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels) ||
            experts != mResource->experts || inputChannels != mResource->inputChannels ||
            outputChannels != mResource->outputChannels ||
            op->main_as_IndexedLinearParam()->inputMode() != mInputMode ||
            (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) != mFuseSwiGLU)
            return false;
        if (op->main_as_IndexedLinearParam()->routeWeight() != mRouteWeight ||
            op->main_as_IndexedLinearParam()->routeEps() != mRouteEps ||
            op->main_as_IndexedLinearParam()->routeScale() != mRouteScale)
            return false;
        auto context = (__bridge MNNMetalContext*)static_cast<MetalBackend*>(backend)->context();
        if (MetalBackend::getBuffer(mResource->weight.get()).first.device != context.device)
            return false;
        *dst = new MetalIndexedLinear(backend, mResource, mInputMode, mFuseSwiGLU, mRouteWeight, mRouteEps,
                                      mRouteScale);
        MNN_METAL_PROFILE_REGISTER_CLONE(backend, op, *dst);
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (mRouteWeight) {
            return onResizeRouteWeight(inputs, outputs);
        }
        if (inputs.size() != 2 || outputs.size() != 1 || inputs[1]->dimensions() != 2 ||
            inputs[1]->getType() != halide_type_of<int32_t>() || inputs[0]->getType() != halide_type_of<float>() ||
            outputs[0]->getType() != halide_type_of<float>())
            return NOT_SUPPORT;
        auto x    = inputs[0];
        auto xDes = TensorUtils::getDescribe(x);
        int xChannels = 0;
        if (xDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            if (mInputMode != 0 || x->dimensions() != 4 || x->length(2) != 1 || x->length(3) != 1 ||
                x->length(1) % 4 != 0)
                return NOT_SUPPORT;
            xChannels = x->length(1);
        } else {
            if (xDes->dimensionFormat != MNN_DATA_FORMAT_NCHW || x->dimensions() != (mInputMode == 0 ? 2 : 3))
                return NOT_SUPPORT;
            xChannels = x->length(x->dimensions() - 1);
        }
        for (auto t : {inputs[1], outputs[0]}) {
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
        const int tokens = inputs[1]->length(0), topK = inputs[1]->length(1);
        if (tokens <= 0 || topK <= 0 || static_cast<int64_t>(tokens) * topK > INT_MAX)
            return INPUT_DATA_ERROR;
        mRoutes = tokens * topK;
        if (inputs[0]->length(0) != tokens || (mInputMode == 1 && inputs[0]->length(1) != topK) ||
            xChannels != mResource->inputChannels ||
            outputs[0]->dimensions() != 3 || outputs[0]->length(0) != tokens || outputs[0]->length(1) != topK ||
            outputs[0]->length(2) != mResource->outputChannels / (mFuseSwiGLU ? 2 : 1))
            return INPUT_DATA_ERROR;
        auto backend = static_cast<MetalBackend*>(this->backend());
        mGrouped = tokens > 1;
        auto code = preparePipelines();
        if (code != NO_ERROR) return code;
        if (nil == mParams) {
            mParams = backend->getConstBuffer(13 * sizeof(int));
            if (nil == mParams) {
                return OUT_OF_MEMORY;
            }
        }
        const int params[] = {
            mResource->inputChannels, mResource->outputChannels, mResource->experts,     topK, mRoutes, mInputMode,
            mResource->blockSize,     mResource->blockCount,     mResource->weightStride, tokens,
            xDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0};
        ::memcpy(mParams.contents, params, sizeof(params));
        ::memset(static_cast<char*>(mParams.contents) + sizeof(params), 0, 2 * sizeof(float));
        if (mGrouped) {
            const size_t sizes[] = {static_cast<size_t>(mResource->experts), static_cast<size_t>(mRoutes),
                                    static_cast<size_t>(mResource->experts) + 1,
                                    static_cast<size_t>(mResource->experts) + 1, static_cast<size_t>(mRoutes)};
            std::shared_ptr<Tensor> work[5];
            std::shared_ptr<Tensor> aPerm;
            std::vector<std::shared_ptr<Tensor>> staged;
            auto fail = [&]() {
                for (auto& t : staged) {
                    backend->onReleaseBuffer(t.get(), Backend::DYNAMIC);
                }
                return OUT_OF_MEMORY;
            };
            for (int i = 0; i < 5; ++i) {
                work[i] = allocIndexedBytes(backend, sizes[i] * sizeof(int), Backend::DYNAMIC);
                if (!work[i]) return fail();
                staged.push_back(work[i]);
            }
            if (mAdev) {
                aPerm = allocIndexedBytes(backend, (int64_t)mRoutes * mResource->inputChannels *
                                                        (mResource->fp16 ? 2 : 4),
                                           Backend::DYNAMIC);
                if (!aPerm) return fail();
                staged.push_back(aPerm);
            }
            for (auto& t : staged) {
                backend->onReleaseBuffer(t.get(), Backend::DYNAMIC);
            }
            for (int i = 0; i < 5; ++i) {
                mWork[i] = std::move(work[i]);
            }
            mAPerm = std::move(aPerm);
        } else {
            for (auto& work : mWork) {
                work.reset();
            }
            mAPerm.reset();
        }
        // Resizing changes the parameter buffer and workspace captured by encode replay.
        mReplayEvents.clear();
        mReplayState = 0;
        mReplayStable = 0;
        return NO_ERROR;
    }

    ErrorCode onResizeRouteWeight(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        if (inputs.size() != 3 || outputs.size() != 1 || mInputMode != 1 || mFuseSwiGLU) {
            return NOT_SUPPORT;
        }
        auto x = inputs[0];
        auto ids = inputs[1];
        auto w = inputs[2];
        auto y = outputs[0];
        if (ids->dimensions() != 2 || ids->getType() != halide_type_of<int32_t>() ||
            x->getType() != halide_type_of<float>() || w->getType() != halide_type_of<float>() ||
            y->getType() != halide_type_of<float>()) {
            return NOT_SUPPORT;
        }
        for (auto t : {ids, w}) {
            if (TensorUtils::getDescribe(t)->dimensionFormat != MNN_DATA_FORMAT_NCHW) {
                return NOT_SUPPORT;
            }
            int64_t stride = 1;
            for (int d = t->dimensions() - 1; d >= 0; --d) {
                if (t->length(d) < 0 || t->stride(d) != stride) {
                    return INPUT_DATA_ERROR;
                }
                stride *= t->length(d);
            }
        }
        if (TensorUtils::getDescribe(x)->dimensionFormat != MNN_DATA_FORMAT_NCHW || x->dimensions() != 3) {
            return NOT_SUPPORT;
        }
        auto yDes = TensorUtils::getDescribe(y);
        if (yDes->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 || y->dimensions() != 4 || y->length(2) != 1 ||
            y->length(3) != 1 || y->length(1) != mResource->outputChannels || y->length(1) % 4 != 0) {
            return NOT_SUPPORT;
        }
        const int tokens = ids->length(0), topK = ids->length(1);
        const bool wOk = (mRouteWeight == 2 || mRouteWeight == 3)
                             ? (w->elementSize() == static_cast<int64_t>(tokens) * mResource->experts &&
                                w->length(w->dimensions() - 1) == mResource->experts)
                             : (w->length(0) == tokens && w->length(1) == topK);
        if (tokens <= 0 || topK <= 0 || static_cast<int64_t>(tokens) * topK > INT_MAX ||
            x->length(0) != tokens || x->length(1) != topK || x->length(2) != mResource->inputChannels ||
            !wOk || y->length(0) != tokens) {
            return INPUT_DATA_ERROR;
        }
        mRoutes = tokens * topK;
        auto backend = static_cast<MetalBackend*>(this->backend());
        mGrouped = tokens > 1;
        auto code = preparePipelines();
        if (code != NO_ERROR) return code;
        if (nil == mParams) {
            mParams = backend->getConstBuffer(13 * sizeof(int));
            if (nil == mParams) {
                return OUT_OF_MEMORY;
            }
        }
        const int params[] = {mResource->inputChannels, mResource->outputChannels, mResource->experts, topK,
                              mRoutes, mInputMode, mResource->blockSize, mResource->blockCount,
                              mResource->weightStride, tokens, 0};
        ::memcpy(mParams.contents, params, sizeof(params));
        const float rwTail[] = {mRouteEps, mRouteScale};
        ::memcpy(static_cast<char*>(mParams.contents) + sizeof(params), rwTail, sizeof(rwTail));
        if (mGrouped) {
            const size_t sizes[] = {static_cast<size_t>(mResource->experts), static_cast<size_t>(mRoutes),
                                    static_cast<size_t>(mResource->experts) + 1,
                                    static_cast<size_t>(mResource->experts) + 1, static_cast<size_t>(mRoutes)};
            std::shared_ptr<Tensor> work[5];
            std::shared_ptr<Tensor> routeTmp;
            std::shared_ptr<Tensor> aPerm;
            std::vector<std::shared_ptr<Tensor>> staged;
            auto fail = [&]() {
                for (auto& t : staged) {
                    backend->onReleaseBuffer(t.get(), Backend::DYNAMIC);
                }
                return OUT_OF_MEMORY;
            };
            for (int i = 0; i < 5; ++i) {
                work[i] = allocIndexedBytes(backend, sizes[i] * sizeof(int), Backend::DYNAMIC);
                if (!work[i]) return fail();
                staged.push_back(work[i]);
            }
            if (!mAtomicRw) {
                routeTmp = allocIndexedBytes(backend, (int64_t)mRoutes * mResource->outputChannels *
                                                           (mResource->fp16 ? 2 : 4),
                                              Backend::DYNAMIC);
                if (!routeTmp) return fail();
                staged.push_back(routeTmp);
            }
            if (mAdev) {
                aPerm = allocIndexedBytes(backend, (int64_t)mRoutes * mResource->inputChannels *
                                                        (mResource->fp16 ? 2 : 4),
                                           Backend::DYNAMIC);
                if (!aPerm) return fail();
                staged.push_back(aPerm);
            }
            for (auto& t : staged) {
                backend->onReleaseBuffer(t.get(), Backend::DYNAMIC);
            }
            for (int i = 0; i < 5; ++i) {
                mWork[i] = std::move(work[i]);
            }
            mRouteTmp = std::move(routeTmp);
            mAPerm = std::move(aPerm);
        } else {
            for (auto& work : mWork) {
                work.reset();
            }
            mRouteTmp.reset();
            mAPerm.reset();
        }
        mReplayEvents.clear();
        mReplayState = 0;
        mReplayStable = 0;
        return NO_ERROR;
    }

    void onEncodeRouteWeight(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             id<MTLComputeCommandEncoder> encoder) {
        const int tokens = inputs[1]->length(0);
        const int resultChannels = mResource->outputChannels;
        if (!mGrouped) {
            [encoder setComputePipelineState:mPipelines[mDecodeKernel]];
            MetalBackend::setTensor(inputs[0], encoder, 0);
            MetalBackend::setTensor(inputs[1], encoder, 1);
            MetalBackend::setTensor(outputs[0], encoder, 2);
            [encoder setBuffer:mParams offset:0 atIndex:3];
            MetalBackend::setTensor(mResource->weight.get(), encoder, 4);
            MetalBackend::setTensor(mResource->scale.get(), encoder, 5);
            MetalBackend::setTensor(mResource->bias.get(), encoder, 6);
            MetalBackend::setTensor(inputs[2], encoder, 7);
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(UP_DIV(resultChannels, 4), 2), tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        if (mAtomicRw) {
            [encoder setComputePipelineState:mPipelines[ZeroY]];
            MetalBackend::setTensor(outputs[0], encoder, 0);
            [encoder setBuffer:mParams offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(tokens * resultChannels / 2, 128), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        encodeRouting(inputs, mAtomicRw ? outputs[0] : mRouteTmp.get(), encoder);
        if (mAdev) {
            encodePermuteA(inputs, encoder);
        }
        [encoder setComputePipelineState:mPipelines[mGroupedKernel]];
        MetalBackend::setTensor(mAdev ? mAPerm.get() : inputs[0], encoder, 0);
        MetalBackend::setTensor(mAtomicRw ? outputs[0] : mRouteTmp.get(), encoder, 1);
        [encoder setBuffer:mParams offset:0 atIndex:2];
        MetalBackend::setTensor(mResource->weight.get(), encoder, 3);
        MetalBackend::setTensor(mResource->scale.get(), encoder, 4);
        MetalBackend::setTensor(mResource->bias.get(), encoder, 5);
        MetalBackend::setTensor(mWork[2].get(), encoder, 6);
        MetalBackend::setTensor(mWork[3].get(), encoder, 7);
        MetalBackend::setTensor(mWork[4].get(), encoder, 8);
        if (mAtomicRw) {
            MetalBackend::setTensor(inputs[2], encoder, 9);
            MetalBackend::setTensor(inputs[1], encoder, 10);
        }
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->outputChannels, mTileN),
                                                  UP_DIV(mRoutes, mTileM) + mResource->experts, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (!mAtomicRw) {
            [encoder setComputePipelineState:mPipelines[WeightedReduce]];
            MetalBackend::setTensor(mRouteTmp.get(), encoder, 0);
            MetalBackend::setTensor(inputs[2], encoder, 1);
            MetalBackend::setTensor(outputs[0], encoder, 2);
            [encoder setBuffer:mParams offset:0 atIndex:3];
            MetalBackend::setTensor(inputs[1], encoder, 4);
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->outputChannels, 128), tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }

    void encodeRouting(const std::vector<Tensor*>& inputs, Tensor* zeroFill,
                       id<MTLComputeCommandEncoder> encoder) {
        if (mFusedRoute) {
            [encoder setComputePipelineState:mPipelines[Route]];
            MetalBackend::setTensor(inputs[1], encoder, 0);
            MetalBackend::setTensor(mWork[1].get(), encoder, 1);
            MetalBackend::setTensor(mWork[2].get(), encoder, 2);
            MetalBackend::setTensor(mWork[3].get(), encoder, 3);
            MetalBackend::setTensor(mWork[4].get(), encoder, 4);
            MetalBackend::setTensor(zeroFill, encoder, 5);
            [encoder setBuffer:mParams offset:0 atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            return;
        }
        [encoder setComputePipelineState:mPipelines[Clear]];
        MetalBackend::setTensor(mWork[0].get(), encoder, 0);
        [encoder setBuffer:mParams offset:0 atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->experts, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder setComputePipelineState:mPipelines[Histogram]];
        MetalBackend::setTensor(inputs[1], encoder, 0);
        MetalBackend::setTensor(mWork[0].get(), encoder, 1);
        MetalBackend::setTensor(mWork[1].get(), encoder, 2);
        [encoder setBuffer:mParams offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mRoutes, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder setComputePipelineState:mPipelines[Prefix]];
        MetalBackend::setTensor(mWork[0].get(), encoder, 0);
        MetalBackend::setTensor(mWork[2].get(), encoder, 1);
        MetalBackend::setTensor(mWork[3].get(), encoder, 2);
        [encoder setBuffer:mParams offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder setComputePipelineState:mPipelines[Scatter]];
        MetalBackend::setTensor(inputs[1], encoder, 0);
        MetalBackend::setTensor(mWork[1].get(), encoder, 1);
        MetalBackend::setTensor(mWork[2].get(), encoder, 2);
        MetalBackend::setTensor(mWork[4].get(), encoder, 3);
        MetalBackend::setTensor(zeroFill, encoder, 4);
        [encoder setBuffer:mParams offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mRoutes, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encodePermuteA(const std::vector<Tensor*>& inputs, id<MTLComputeCommandEncoder> encoder) {
        [encoder setComputePipelineState:mPipelines[PermuteA]];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mWork[4].get(), encoder, 1);
        MetalBackend::setTensor(mAPerm.get(), encoder, 2);
        [encoder setBuffer:mParams offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mRoutes, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                  id<MTLComputeCommandEncoder> encoder) override {
        if (mRouteWeight) {
            onEncodeRouteWeight(inputs, outputs, encoder);
            return;
        }
        if (!mGrouped) {
            [encoder setComputePipelineState:mPipelines[mDecodeKernel]];
            MetalBackend::setTensor(inputs[0], encoder, 0);
            MetalBackend::setTensor(inputs[1], encoder, 1);
            MetalBackend::setTensor(outputs[0], encoder, 2);
            [encoder setBuffer:mParams offset:0 atIndex:3];
            MetalBackend::setTensor(mResource->weight.get(), encoder, 4);
            MetalBackend::setTensor(mResource->scale.get(), encoder, 5);
            MetalBackend::setTensor(mResource->bias.get(), encoder, 6);
            const int resultChannels = mResource->outputChannels / (mFuseSwiGLU ? 2 : 1);
            if (mResource->bits == 4 || mResource->bits == 8) {
                [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(UP_DIV(resultChannels, 4), 2), mRoutes, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else {
                [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(resultChannels, 16), mRoutes, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            return;
        }
        encodeRouting(inputs, outputs[0], encoder);
        if (mAdev) {
            encodePermuteA(inputs, encoder);
        }
        [encoder setComputePipelineState:mPipelines[mGroupedKernel]];
        MetalBackend::setTensor(mAdev ? mAPerm.get() : inputs[0], encoder, 0);
        MetalBackend::setTensor(outputs[0], encoder, 1);
        [encoder setBuffer:mParams offset:0 atIndex:2];
        MetalBackend::setTensor(mResource->weight.get(), encoder, 3);
        MetalBackend::setTensor(mResource->scale.get(), encoder, 4);
        MetalBackend::setTensor(mResource->bias.get(), encoder, 5);
        MetalBackend::setTensor(mWork[2].get(), encoder, 6);
        MetalBackend::setTensor(mWork[3].get(), encoder, 7);
        MetalBackend::setTensor(mWork[4].get(), encoder, 8);
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->outputChannels, mTileN),
                                                  UP_DIV(mRoutes, mTileM) + mResource->experts, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

private:
    Kernel selectGroupedKernel() const {
        if ((mResource->bits != 4 && mResource->bits != 8) ||
            mResource->outputChannels < 64 || mRoutes < 4LL * mResource->experts) {
            return GroupedSimd;
        }
#ifdef MNN_METAL_INDEXED_TENSOR_AVAILABLE
        if (@available(macOS 26.0, iOS 26.0, *)) {
            auto backend = static_cast<MetalBackend*>(this->backend());
            auto runtime = static_cast<MetalRuntime*>(backend->getRuntime());
            const auto& gpu = runtime->gpuInfo();
            if (gpu.gen >= 16) {
                return GroupedTensor;
            }
        }
#endif
        return GroupedSimd;
    }

    bool vecStageEnabled() const {
        return (mResource->inputChannels % 4) == 0 &&
               mResource->blockSize >= 16 &&
               (64 % mResource->blockSize == 0 || mResource->blockSize % 64 == 0);
    }

    void pickTileM() {
        int tileM = 16;
        if (mGroupedKernel == GroupedTensor && vecStageEnabled()) {
            const int avgRows = mRoutes / mResource->experts;
            const int m32Floor = mResource->experts >= 128 ? 8 : 24;
            tileM = avgRows >= 48 ? 64 : (avgRows >= m32Floor ? 32 : 16);
        }
        if (tileM != mTileM) {
            mTileM = tileM;
            mPipelines[Prefix] = nil;
            mPipelines[Route] = nil;
            mPipelines[GroupedTensor] = nil;
            mPipelines[GroupedSimd] = nil;
        }
    }

    bool createPipeline(Kernel kernel) {
        if (nil != mPipelines[kernel]) return true;
        static const char* const names[] = {
            "indexed_linear_gemv", "indexed_linear_gemv_packed",
            "indexed_linear_clear", "indexed_linear_histogram", "indexed_linear_prefix",
            "indexed_linear_scatter", "indexed_linear_grouped_simd", "indexed_linear_grouped_tensor",
            "indexed_linear_gemv_rw", "indexed_linear_rw_reduce",
            "indexed_linear_route", "indexed_linear_zero_y", "indexed_linear_permute_a"
        };
        static_assert(sizeof(names) / sizeof(names[0]) == KernelCount, "Missing IndexedLinear kernel name");
        auto backend = static_cast<MetalBackend*>(this->backend());
        auto runtime = static_cast<MetalRuntime*>(backend->getRuntime());
        auto context = (__bridge MNNMetalContext*)backend->context();
        const bool useSimdGroupMatrix = runtime->supportSimdGroupMatrix();
        const bool rawWeights = (mRouteWeight == 2 || mRouteWeight == 3) &&
                                (kernel == WeightedGemv || kernel == WeightedReduce ||
                                 kernel == Route || kernel == GroupedTensor);
        const bool vecStage = vecStageEnabled();
        std::vector<std::string> keys{names[kernel], mResource->fp16 ? "half" : "float",
                                      std::to_string(mResource->bits), useSimdGroupMatrix ? "matrix" : "scalar",
                                      mFuseSwiGLU ? "swiglu" : "linear", vecStage ? "vec64" : "scalar32",
                                      "m" + std::to_string(mTileM), "n" + std::to_string(mTileN),
                                      "e" + std::to_string(mResource->experts)};
        if (rawWeights) keys.emplace_back("raw");
        if (mRouteWeight == 3) keys.emplace_back("logit");
        if (mAtomicRw) keys.emplace_back("rwat");
        if (mAdev) keys.emplace_back("adev");
        auto pipeline = runtime->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            NSMutableDictionary* macros = [@{
                @"ftype" : mResource->fp16 ? @"half" : @"float",
                @"ftype4" : mResource->fp16 ? @"half4" : @"float4",
                @"WEIGHT_BITS" : @(mResource->bits), @"USE_SIMDGROUP_MATRIX" : @(useSimdGroupMatrix ? 1 : 0),
                @"FUSE_SILU_MUL" : @(mFuseSwiGLU ? 1 : 0), @"RW_RAW" : @(rawWeights ? 1 : 0),
                @"RW_LOGIT" : @(mRouteWeight == 3 && rawWeights ? 1 : 0),
                @"IDX_VEC" : @(vecStage ? 1 : 0), @"IDX_M" : @(mTileM), @"IDX_N" : @(mTileN),
                @"IDX_E" : @(mResource->experts), @"RW_ATOMIC" : @(mAtomicRw ? 1 : 0),
                @"IDX_ADEV" : @(mAdev ? 1 : 0)
            } mutableCopy];
            options.preprocessorMacros = macros;
            NSString* source = @(gIndexedLinear);
            if (kernel == GroupedTensor) {
#ifdef MNN_METAL_INDEXED_TENSOR_AVAILABLE
                if (@available(macOS 26.0, iOS 26.0, *)) {
                    options.languageVersion = MTLLanguageVersion4_0;
                    source = [source stringByAppendingString:@(gIndexedLinearTensor)];
                } else {
                    return false;
                }
#else
                return false;
#endif
            }
            pipeline = [context pipelineWithSourceOption:source name:@(names[kernel]) options:options];
            if (nil == pipeline) return false;
            if (MetalEnv::get().pipelineInfo) {
                MNN_PRINT("[PipelineInfo] %s maxThreads=%lu execWidth=%lu smem=%lu macros=%s\n", names[kernel],
                          (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
                          (unsigned long)pipeline.threadExecutionWidth,
                          (unsigned long)pipeline.staticThreadgroupMemoryLength,
                          options.preprocessorMacros.description.UTF8String);
            }
        }
        const NSUInteger threads = kernel == Prefix ? 1 :
                                   (kernel == Route ? 1024 :
                                    (kernel == PackedGemv || kernel == WeightedGemv ? 64 : 128));
        if (pipeline.maxTotalThreadsPerThreadgroup < threads ||
            pipeline.staticThreadgroupMemoryLength > context.device.maxThreadgroupMemoryLength) {
            return false;
        }
        runtime->insertPipeline(keys, pipeline);
        mPipelines[kernel] = pipeline;
        return true;
    }

    ErrorCode preparePipelines() {
        if (!mGrouped) {
            const bool packed = mResource->bits == 4 || mResource->bits == 8;
            mDecodeKernel = mRouteWeight ? WeightedGemv : (packed ? PackedGemv : Gemv);
            return createPipeline(mDecodeKernel) ? NO_ERROR : NOT_SUPPORT;
        }
        mGroupedKernel = selectGroupedKernel();
        pickTileM();
        mAtomicRw = mRouteWeight != 0 && mGroupedKernel == GroupedTensor &&
                    mResource->fp16 && !mFuseSwiGLU;
        mAdev = mGroupedKernel == GroupedTensor && vecStageEnabled() &&
                (mResource->bits == 4 || mResource->bits == 8) && !mFuseSwiGLU;
        if (!createPipeline(mGroupedKernel)) {
            if (mGroupedKernel != GroupedTensor) return NOT_SUPPORT;
            MNN_PRINT("Metal IndexedLinear Tensor pipeline unavailable; selecting grouped SIMD.\n");
            mGroupedKernel = GroupedSimd;
            mAtomicRw = false;
            mAdev = false;
            pickTileM();
            if (!createPipeline(mGroupedKernel)) return NOT_SUPPORT;
        }
        if (mAdev && !createPipeline(PermuteA)) {
            mAdev = false;
            mPipelines[mGroupedKernel] = nil;
            if (!createPipeline(mGroupedKernel)) return NOT_SUPPORT;
        }
        mFusedRoute = createPipeline(Route);
        if (!mFusedRoute && mAtomicRw) {
            mAtomicRw = false;
            mPipelines[mGroupedKernel] = nil;
            if (!createPipeline(mGroupedKernel)) return NOT_SUPPORT;
        }
        if (mFusedRoute) {
            if (mAtomicRw && !createPipeline(ZeroY)) return NOT_SUPPORT;
        } else {
            for (auto kernel : {Clear, Histogram, Prefix, Scatter}) {
                if (!createPipeline(kernel)) return NOT_SUPPORT;
            }
        }
        if (mRouteWeight && !mAtomicRw && !createPipeline(WeightedReduce)) return NOT_SUPPORT;
        return NO_ERROR;
    }

    std::shared_ptr<IndexedLinearResource> mResource;
    std::shared_ptr<Tensor> mWork[5]; // counts, local ranks, offsets, tile offsets, sorted routes.
    std::shared_ptr<Tensor> mRouteTmp;
    std::shared_ptr<Tensor> mAPerm;
    bool mAdev = false;
    id<MTLComputePipelineState> mPipelines[KernelCount] = {};
    Kernel mDecodeKernel = Gemv;
    Kernel mGroupedKernel = GroupedSimd;
    int mTileM = 16;
    int mTileN = 64;
    bool mFusedRoute = true;
    bool mAtomicRw = false;
    id<MTLBuffer> mParams = nil;
    int mInputMode = 0;
    bool mFuseSwiGLU = false;
    int mRouteWeight = 0;
    float mRouteEps = 0.0f;
    float mRouteScale = 1.0f;
    int mRoutes = 0;
    bool mGrouped = false;
};

class MetalIndexedLinearCreator : public MetalBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const Op* op, Backend* backend,
                        const std::vector<Tensor*>& outputs) const override {
        auto metal = static_cast<MetalBackend*>(backend);
        if (!metal->runtime()->supportSimdGroupReduce())
            return nullptr;
        auto resource = loadIndexedResource(op, metal);
        if (!resource)
            return nullptr;
        return new MetalIndexedLinear(backend, resource, op->main_as_IndexedLinearParam()->inputMode(),
                                      (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU),
                                      op->main_as_IndexedLinearParam()->routeWeight(),
                                      op->main_as_IndexedLinearParam()->routeEps(),
                                      op->main_as_IndexedLinearParam()->routeScale());
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalIndexedLinearCreator, OpType_IndexedLinear);
} // namespace MNN
#endif

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
