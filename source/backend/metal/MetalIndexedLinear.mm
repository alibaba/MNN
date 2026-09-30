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

static std::shared_ptr<IndexedLinearResource> loadIndexedResource(const Op* op, MetalBackend* backend) {
    auto res = std::make_shared<IndexedLinearResource>();
    if (!IndexedLinearCommon::dimensions(op, res->experts, res->inputChannels, res->outputChannels))
        return nullptr;
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
        const size_t stride = bits ? ((elements * bits + 7) / 8) : elements * (res->fp16 ? 2 : 4);
        if (stride > INT_MAX || (quantized && weight->weight.size() < stride))
            return nullptr;
        if (e == 0) {
            res->bits = bits;
            res->blockCount = blockCount;
            res->blockSize = res->inputChannels / blockCount;
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
            ::memcpy(dst, weight->weight.get(), stride);
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
public:
    MetalIndexedLinear(Backend* backend, std::shared_ptr<IndexedLinearResource> resource, int inputMode,
                       bool fuseSwiGLU)
        : MetalExecution(backend), mResource(std::move(resource)), mInputMode(inputMode), mFuseSwiGLU(fuseSwiGLU) {}

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
        auto context = (__bridge MNNMetalContext*)static_cast<MetalBackend*>(backend)->context();
        if (MetalBackend::getBuffer(mResource->weight.get()).first.device != context.device)
            return false;
        *dst = new MetalIndexedLinear(backend, mResource, mInputMode, mFuseSwiGLU);
        MNN_METAL_PROFILE_REGISTER_CLONE(backend, op, *dst);
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
        const int tokens = inputs[1]->length(0), topK = inputs[1]->length(1);
        if (tokens <= 0 || topK <= 0 || static_cast<int64_t>(tokens) * topK > INT_MAX)
            return INPUT_DATA_ERROR;
        mRoutes = tokens * topK;
        if (inputs[0]->dimensions() != (mInputMode == 0 ? 2 : 3) || inputs[0]->length(0) != tokens ||
            (mInputMode == 1 && inputs[0]->length(1) != topK) ||
            inputs[0]->length(inputs[0]->dimensions() - 1) != mResource->inputChannels ||
            outputs[0]->dimensions() != 3 || outputs[0]->length(0) != tokens || outputs[0]->length(1) != topK ||
            outputs[0]->length(2) != mResource->outputChannels / (mFuseSwiGLU ? 2 : 1))
            return INPUT_DATA_ERROR;
        auto backend = static_cast<MetalBackend*>(this->backend());
        auto runtime = const_cast<MetalRuntime*>(backend->runtime());
        mGrouped = tokens > 1;
        const bool matrix = mGrouped && mResource->fp16 && runtime->supportSimdGroupMatrix();
        const char* names[] = {"indexed_linear_gemv",   "indexed_linear_clear",   "indexed_linear_histogram",
                               "indexed_linear_prefix", "indexed_linear_scatter", "indexed_linear_grouped"};
        for (int i = mGrouped ? 1 : 0; i < (mGrouped ? 6 : 1); ++i) {
            std::vector<std::string> keys{names[i], mResource->fp16 ? "half" : "float", std::to_string(mResource->bits),
                                          matrix ? "matrix" : "scalar", mFuseSwiGLU ? "swiglu" : "linear"};
            mPipelines[i] = runtime->findPipeline(keys);
            if (nil == mPipelines[i]) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.preprocessorMacros = @{
                    @"ftype" : mResource->fp16 ? @"half" : @"float",
                    @"WEIGHT_BITS" : @(mResource->bits),
                    @"USE_SIMDGROUP_MATRIX" : @(matrix ? 1 : 0),
                    @"FUSE_SILU_MUL" : @(mFuseSwiGLU ? 1 : 0)
                };
                mPipelines[i] = backend->makeComputePipelineWithSourceOption(gIndexedLinear, names[i], options);
                if (nil == mPipelines[i])
                    return NOT_SUPPORT;
                runtime->insertPipeline(keys, mPipelines[i]);
            }
            if (i != 3 && mPipelines[i].maxTotalThreadsPerThreadgroup < 128)
                return NOT_SUPPORT;
        }
        if (nil == mParams) {
            mParams = backend->getConstBuffer(9 * sizeof(int));
            if (nil == mParams) {
                return OUT_OF_MEMORY;
            }
        }
        const int params[] = {
            mResource->inputChannels, mResource->outputChannels, mResource->experts,     topK, mRoutes, mInputMode,
            mResource->blockSize,     mResource->blockCount,     mResource->weightStride};
        ::memcpy(mParams.contents, params, sizeof(params));
        if (mGrouped) {
            const size_t sizes[] = {static_cast<size_t>(mResource->experts), static_cast<size_t>(mRoutes),
                                    static_cast<size_t>(mResource->experts) + 1,
                                    static_cast<size_t>(mResource->experts) + 1, static_cast<size_t>(mRoutes)};
            for (int i = 0; i < 5; ++i) {
                mWork[i] = allocIndexedBytes(backend, sizes[i] * sizeof(int), Backend::DYNAMIC);
                if (!mWork[i])
                    return OUT_OF_MEMORY;
            }
            for (auto& work : mWork)
                backend->onReleaseBuffer(work.get(), Backend::DYNAMIC);
        }
        // Resizing changes the parameter buffer and workspace captured by encode replay.
        mReplayEvents.clear();
        mReplayState = 0;
        mReplayStable = 0;
        return NO_ERROR;
    }

    void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                  id<MTLComputeCommandEncoder> encoder) override {
        if (!mGrouped) {
            [encoder setComputePipelineState:mPipelines[0]];
            MetalBackend::setTensor(inputs[0], encoder, 0);
            MetalBackend::setTensor(inputs[1], encoder, 1);
            MetalBackend::setTensor(outputs[0], encoder, 2);
            [encoder setBuffer:mParams offset:0 atIndex:3];
            MetalBackend::setTensor(mResource->weight.get(), encoder, 4);
            MetalBackend::setTensor(mResource->scale.get(), encoder, 5);
            MetalBackend::setTensor(mResource->bias.get(), encoder, 6);
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->outputChannels / (mFuseSwiGLU ? 2 : 1), 16),
                                                      mRoutes, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
        [encoder setComputePipelineState:mPipelines[1]];
        MetalBackend::setTensor(mWork[0].get(), encoder, 0);
        [encoder setBuffer:mParams offset:0 atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->experts, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder setComputePipelineState:mPipelines[2]];
        MetalBackend::setTensor(inputs[1], encoder, 0);
        MetalBackend::setTensor(mWork[0].get(), encoder, 1);
        MetalBackend::setTensor(mWork[1].get(), encoder, 2);
        [encoder setBuffer:mParams offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mRoutes, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder setComputePipelineState:mPipelines[3]];
        MetalBackend::setTensor(mWork[0].get(), encoder, 0);
        MetalBackend::setTensor(mWork[2].get(), encoder, 1);
        MetalBackend::setTensor(mWork[3].get(), encoder, 2);
        [encoder setBuffer:mParams offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder setComputePipelineState:mPipelines[4]];
        MetalBackend::setTensor(inputs[1], encoder, 0);
        MetalBackend::setTensor(mWork[1].get(), encoder, 1);
        MetalBackend::setTensor(mWork[2].get(), encoder, 2);
        MetalBackend::setTensor(mWork[4].get(), encoder, 3);
        MetalBackend::setTensor(outputs[0], encoder, 4);
        [encoder setBuffer:mParams offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mRoutes, 128), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder setComputePipelineState:mPipelines[5]];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(outputs[0], encoder, 1);
        [encoder setBuffer:mParams offset:0 atIndex:2];
        MetalBackend::setTensor(mResource->weight.get(), encoder, 3);
        MetalBackend::setTensor(mResource->scale.get(), encoder, 4);
        MetalBackend::setTensor(mResource->bias.get(), encoder, 5);
        MetalBackend::setTensor(mWork[2].get(), encoder, 6);
        MetalBackend::setTensor(mWork[3].get(), encoder, 7);
        MetalBackend::setTensor(mWork[4].get(), encoder, 8);
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mResource->outputChannels, 16),
                                                  UP_DIV(mRoutes, 16) + mResource->experts, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

private:
    std::shared_ptr<IndexedLinearResource> mResource;
    std::shared_ptr<Tensor> mWork[5]; // counts, local ranks, offsets, tile offsets, sorted routes.
    id<MTLComputePipelineState> mPipelines[6] = {nil, nil, nil, nil, nil, nil};
    id<MTLBuffer> mParams = nil;
    int mInputMode = 0;
    bool mFuseSwiGLU = false;
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
                                      (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU));
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalIndexedLinearCreator, OpType_IndexedLinear);
} // namespace MNN
#endif

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
