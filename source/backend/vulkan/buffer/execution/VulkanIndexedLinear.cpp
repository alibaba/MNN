//
//  VulkanIndexedLinear.cpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanIndexedLinear.hpp"
#include "core/IndexedLinearCommon.hpp"
#include "core/TensorUtils.hpp"
#include "half.hpp"
#include <climits>
#include <cstring>

namespace MNN {

VulkanIndexedLinear::VulkanIndexedLinear(const Op* op, Backend* backend)
    : VulkanBasicExecution(backend),
      mInputMode(op->main_as_IndexedLinearParam()->inputMode()),
      mFuseSwiGLU(op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU),
      mRouteWeight(op->main_as_IndexedLinearParam()->routeWeight()),
      mRouteEps(op->main_as_IndexedLinearParam()->routeEps()),
      mRouteScale(op->main_as_IndexedLinearParam()->routeScale()) {
    mValid = initResource(op) && initPipelines();
}

VulkanIndexedLinear::VulkanIndexedLinear(const Op* op, Backend* backend, std::shared_ptr<Resource> resource)
    : VulkanBasicExecution(backend),
      mResource(std::move(resource)),
      mInputMode(op->main_as_IndexedLinearParam()->inputMode()),
      mFuseSwiGLU(op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU),
      mRouteWeight(op->main_as_IndexedLinearParam()->routeWeight()),
      mRouteEps(op->main_as_IndexedLinearParam()->routeEps()),
      mRouteScale(op->main_as_IndexedLinearParam()->routeScale()) {
    mValid = initPipelines();
}

bool VulkanIndexedLinear::initResource(const Op* op) {
    mResource.reset(new Resource);
    auto& r = *mResource;
    if (!IndexedLinearCommon::dimensions(op, r.experts, r.inputChannels, r.outputChannels))
        return false;
    auto backend = static_cast<VulkanBackend*>(this->backend());
    const int64_t elements = (int64_t)r.inputChannels * r.outputChannels;
    if (elements > INT_MAX / 4 || backend->proty().limits.maxPerStageDescriptorStorageBuffers < 7)
        return false;
    auto allocate = [&](std::shared_ptr<VulkanBuffer>& buffer, size_t bytes) {
        if (bytes > backend->proty().limits.maxStorageBufferRange)
            return false;
        buffer.reset(new VulkanBuffer(backend->getMemoryPool(), false, std::max<size_t>(bytes, 4), nullptr,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_SHARING_MODE_EXCLUSIVE, 0));
        return buffer->valid();
    };
    // Decode only one expert at a time and bound staging memory independently
    // of the expert count. One submission uploads all three banks per chunk.
    std::shared_ptr<VulkanBuffer> staging;
    std::shared_ptr<VulkanCommandPool::Buffer> upload(backend->getPool().allocBuffer());
    size_t used = 0;
    bool recording = false;
    auto flush = [&]() {
        if (recording) {
            upload->end();
            backend->getPool().submitAndWait(upload->get());
            recording = false;
            used = 0;
        }
    };
    for (int e = 0; e < r.experts; ++e) {
        auto data = IndexedLinearCommon::loadLinear(op, e, this->backend());
        std::vector<float> bias;
        if (!data || !IndexedLinearCommon::loadBias(op, e, bias))
            return false;
        const bool floating = data->weightFloat.get() != nullptr;
        const int bits = floating ? (data->quan && data->quan->type() == 3 ? 16 : 32) : (data->canUseInt4 ? 4 : 8);
        const int alphaComponents = !floating && data->asymmetric ? 2 : 1;
        const float* sourceAlpha = floating ? nullptr : data->getAlphaFloat();
        const int groups = floating ? 1 : data->alphaSize / (r.outputChannels * alphaComponents);
        if ((!floating && (!sourceAlpha || data->canUseInt2 || data->canUseInt3)) || groups <= 0 ||
            r.inputChannels % groups || (!floating && data->alphaSize != r.outputChannels * groups * alphaComponents)) {
            return false;
        }
        const int bytes = (elements * bits + 7) / 8;
        if (e == 0) {
            r.weightBits = bits;
            r.groups = groups;
            r.groupSize = r.inputChannels / groups;
            r.weightStride = ROUND_UP(bytes, 4);
            if (!allocate(r.weights, (size_t)r.experts * r.weightStride) ||
                !allocate(r.alpha, (size_t)r.experts * r.outputChannels * r.groups * 2 * sizeof(float)) ||
                !allocate(r.bias, (size_t)r.experts * r.outputChannels * sizeof(float)))
                return false;
            const size_t expertBytes = r.weightStride + (size_t)r.outputChannels * (2 * r.groups + 1) * sizeof(float);
            const size_t capacity = std::min(expertBytes * r.experts, std::max<size_t>(16 * 1024 * 1024, expertBytes));
            if (capacity > INT_MAX)
                return false;
            staging = backend->createHostBuffer(capacity);
            if (!staging->valid())
                return false;
        } else if (r.weightBits != bits || r.groups != groups) {
            return false;
        }
        std::vector<float> alpha(r.outputChannels * groups * 2, 0.0f);
        for (int index = 0; index < r.outputChannels * groups; ++index) {
            alpha[index * 2] = floating || !data->asymmetric ? 0.0f : sourceAlpha[index * 2];
            alpha[index * 2 + 1] = floating ? 1.0f : sourceAlpha[index * alphaComponents + alphaComponents - 1];
        }
        // VkBuffer copies require four-byte-aligned lengths and offsets.
        const size_t alphaBytes = alpha.size() * sizeof(float), biasBytes = bias.size() * sizeof(float);
        const size_t expertBytes = r.weightStride + alphaBytes + biasBytes;
        if (used + expertBytes > staging->size())
            flush();
        auto mapped = static_cast<uint8_t*>(staging->map());
        if (!mapped)
            return false;
        auto weight = mapped + used;
        ::memset(weight, 0, r.weightStride);
        if (bits == 16) {
            // Construct half objects in typed storage; byte staging has no half
            // object lifetime in C++11.
            std::vector<half_float::half> halfWeight(elements);
            for (int i = 0; i < elements; ++i)
                halfWeight[i] = half_float::half(data->weightFloat.get()[i]);
            ::memcpy(weight, halfWeight.data(), bytes);
        } else {
            const void* src = floating ? (const void*)data->weightFloat.get() : (const void*)data->weight.get();
            if (!src || (!floating && data->weight.size() < bytes)) {
                staging->unmap();
                return false;
            }
            ::memcpy(weight, src, bytes);
        }
        ::memcpy(weight + r.weightStride, alpha.data(), alphaBytes);
        ::memcpy(weight + r.weightStride + alphaBytes, bias.data(), biasBytes);
        staging->unmap();
        if (!recording) {
            upload->begin(0);
            recording = true;
        }
        VkBufferCopy weightCopy{used, (size_t)e * r.weightStride, (size_t)r.weightStride};
        VkBufferCopy alphaCopy{used + r.weightStride, (size_t)e * alphaBytes, alphaBytes};
        VkBufferCopy biasCopy{used + r.weightStride + alphaBytes, (size_t)e * biasBytes, biasBytes};
        vkCmdCopyBuffer(upload->get(), staging->buffer(), r.weights->buffer(), 1, &weightCopy);
        vkCmdCopyBuffer(upload->get(), staging->buffer(), r.alpha->buffer(), 1, &alphaCopy);
        vkCmdCopyBuffer(upload->get(), staging->buffer(), r.bias->buffer(), 1, &biasCopy);
        used += expertBytes;
    }
    flush();
    return true;
}

bool VulkanIndexedLinear::initPipelines() {
    auto backend = static_cast<VulkanBackend*>(this->backend());
    const std::string suffix = backend->useFP16() ? "_FP16_comp" : "_comp";
    std::vector<VkDescriptorType> types(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    const std::vector<uint32_t> constants{(uint32_t)mResource->weightBits, (uint32_t)mFuseSwiGLU,
                                          (uint32_t)mRouteWeight};
    mGemv = backend->getPipeline("glsl_indexed_linear_gemv" + suffix, types, {}, constants);
    if (!mGemv)
        return false;
    mGemvSet.reset(mGemv->createSet());
    if (!mGemvSet)
        return false;
    const auto& r = *mResource;
    for (int tile = 0; tile < 2; ++tile) {
        const uint32_t tileRows = tile == 0 ? 8 : 32;
        const uint32_t tileOutputs = tile == 0 ? 32 : 64;
        const std::vector<uint32_t> gemmConstants{(uint32_t)r.weightBits,
                                                  (uint32_t)r.inputChannels,
                                                  (uint32_t)r.outputChannels,
                                                  (uint32_t)r.groups,
                                                  (uint32_t)r.groupSize,
                                                  (uint32_t)r.weightStride,
                                                  (uint32_t)mInputMode,
                                                  tileRows,
                                                  tileOutputs,
                                                  (uint32_t)mFuseSwiGLU};
        mGemm[tile] =
            backend->getPipeline("glsl_indexed_linear_gemm" + suffix, types, {tileOutputs / 4, 8, 1}, gemmConstants);
        if (!mGemm[tile])
            return false;
        mGemmSet[tile].reset(mGemm[tile]->createSet());
        if (!mGemmSet[tile])
            return false;
    }
    if (mRouteWeight) {
        mReduce = backend->getPipeline("glsl_indexed_linear_rw_reduce" + suffix,
                                       std::vector<VkDescriptorType>(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), {},
                                       {(uint32_t)mRouteWeight});
        if (!mReduce)
            return false;
        mReduceSet.reset(mReduce->createSet());
        if (!mReduceSet)
            return false;
    }
    types.resize(2);
    for (int phase = 0; phase < 4; ++phase) {
        mRoute[phase] = backend->getPipeline("glsl_indexed_linear_route_comp", types, {}, {(uint32_t)phase});
        if (!mRoute[phase])
            return false;
        mRouteSet[phase].reset(mRoute[phase]->createSet());
        if (!mRouteSet[phase])
            return false;
    }
    mLargePrefix = backend->getPipeline("glsl_indexed_linear_route_comp", types, {}, {2, 32});
    if (!mLargePrefix)
        return false;
    mLargePrefixSet.reset(mLargePrefix->createSet());
    if (!mLargePrefixSet)
        return false;
    return true;
}

bool VulkanIndexedLinear::onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) {
    if (!dst)
        return mValid;
    int experts, channels, outputs;
    if (!IndexedLinearCommon::dimensions(op, experts, channels, outputs) || experts != mResource->experts ||
        channels != mResource->inputChannels || outputs != mResource->outputChannels ||
        op->main_as_IndexedLinearParam()->inputMode() != mInputMode ||
        (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) != mFuseSwiGLU ||
        op->main_as_IndexedLinearParam()->routeWeight() != mRouteWeight ||
        op->main_as_IndexedLinearParam()->routeEps() != mRouteEps ||
        op->main_as_IndexedLinearParam()->routeScale() != mRouteScale)
        return false;
    if (!backend || backend->type() != MNN_FORWARD_VULKAN ||
        &static_cast<VulkanBackend*>(backend)->getDevice() !=
            &static_cast<VulkanBackend*>(this->backend())->getDevice()) {
        return false;
    }
    auto clone = new VulkanIndexedLinear(op, backend, mResource);
    if (!clone->valid()) {
        delete clone;
        return false;
    }
    *dst = clone;
    return true;
}

ErrorCode VulkanIndexedLinear::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const VulkanCommandPool::Buffer* command) {
    if (!mValid || mRouteWeight < 0 || mRouteWeight > 2 || inputs.size() != (mRouteWeight ? 3 : 2) ||
        outputs.size() != 1 || inputs[1]->dimensions() != 2 || inputs[0]->getType() != halide_type_of<float>() ||
        outputs[0]->getType() != halide_type_of<float>() || inputs[1]->getType() != halide_type_of<int32_t>())
        return NOT_SUPPORT;
    const auto& r = *mResource;
    const int tokens = inputs[1]->length(0), topk = inputs[1]->length(1);
    if (tokens <= 0 || topk <= 0 || (int64_t)tokens * topk > INT_MAX)
        return NOT_SUPPORT;
    const auto x = inputs[0];
    const bool xC4 = TensorUtils::getDescribe(x)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    for (auto tensor : inputs) {
        if (tensor == x && xC4)
            continue;
        if (TensorUtils::getDescribe(tensor)->dimensionFormat != MNN_DATA_FORMAT_NCHW)
            return NOT_SUPPORT;
        int64_t stride = 1;
        for (int dimension = tensor->dimensions() - 1; dimension >= 0; --dimension) {
            if (tensor->length(dimension) < 0 || tensor->stride(dimension) != stride)
                return INPUT_DATA_ERROR;
            stride *= tensor->length(dimension);
            if (stride > INT_MAX)
                return INPUT_DATA_ERROR;
        }
    }
    const int rows = tokens * topk;
    const bool xOk = xC4 ? (mInputMode == 0 && x->dimensions() == 4 && x->length(2) == 1 && x->length(3) == 1 &&
                            x->length(1) == r.inputChannels)
                         : (x->dimensions() == (mInputMode == 0 ? 2 : 3) && (mInputMode == 0 || x->length(1) == topk) &&
                            x->length(x->dimensions() - 1) == r.inputChannels);
    if (!xOk || x->length(0) != tokens)
        return INPUT_DATA_ERROR;
    auto y = outputs[0];
    if (mRouteWeight) {
        auto w = inputs[2];
        const bool wOk = mRouteWeight == 1 ? (w->dimensions() == 2 && w->length(0) == tokens && w->length(1) == topk)
                                           : ((w->dimensions() == 2 || (w->dimensions() == 3 && w->length(0) == 1)) &&
                                              w->length(w->dimensions() - 1) == r.experts &&
                                              w->elementSize() == (int64_t)tokens * r.experts);
        if (mInputMode != 1 || mFuseSwiGLU || !wOk || w->getType() != halide_type_of<float>() ||
            TensorUtils::getDescribe(y)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 || y->dimensions() != 4 ||
            y->length(0) != tokens || y->length(1) != r.outputChannels || y->length(2) != 1 || y->length(3) != 1)
            return INPUT_DATA_ERROR;
    } else if (TensorUtils::getDescribe(y)->dimensionFormat != MNN_DATA_FORMAT_NCHW || y->dimensions() != 3 ||
               y->length(0) != tokens || y->length(1) != topk ||
               y->length(2) != r.outputChannels / (mFuseSwiGLU ? 2 : 1)) {
        return INPUT_DATA_ERROR;
    }
    auto backend = static_cast<VulkanBackend*>(this->backend());
    // Small expert batches avoid the padding cost of the larger register tile.
    const int tile = (int64_t)rows >= 8LL * r.experts ? 1 : 0;
    const int tileRows = tile == 0 ? 8 : 32;
    const int tileOutputs = tile == 0 ? 32 : 64;
    const int64_t maxTiles =
        std::min<int64_t>(rows, UP_DIV((int64_t)rows, tileRows) + std::min<int64_t>(rows, r.experts));
    const auto& limits = backend->proty().limits;
    const int gemvGroups = UP_DIV((int64_t)r.outputChannels / (mFuseSwiGLU ? 2 : 1), 4);
    const int gemmGroups = UP_DIV((int64_t)r.outputChannels, tileOutputs);
    const int expertGroups = UP_DIV((int64_t)r.experts, 64);
    const int rowGroups = UP_DIV((int64_t)rows, 64);
    Parameters parameters{r.inputChannels, r.outputChannels, r.experts,      topk,       mInputMode,
                          r.groups,        r.groupSize,      r.weightStride, rows,       static_cast<int>(maxTiles),
                          tokens,          xC4 ? 1 : 0,      mRouteEps,      mRouteScale};
    auto bindLinear = [&](VulkanLayout::DescriptorSet* descriptors) {
        descriptors->writeBuffer(backend->getBuffer(inputs[0]), 0);
        descriptors->writeBuffer(backend->getBuffer(inputs[1]), 1);
        descriptors->writeBuffer(r.weights->buffer(), 2, r.weights->size());
        descriptors->writeBuffer(r.alpha->buffer(), 3, r.alpha->size());
        descriptors->writeBuffer(r.bias->buffer(), 4, r.bias->size());
        if (mRouteWeight && tokens > 4) {
            descriptors->writeBuffer(mRouteTmp->buffer(), 5, mRouteTmp->size());
        } else {
            descriptors->writeBuffer(backend->getBuffer(outputs[0]), 5);
        }
    };
    auto dispatch = [&](const VulkanPipeline* pipeline, VulkanLayout::DescriptorSet* descriptors, uint32_t x,
                        uint32_t y) {
        pipeline->bind(command->get(), descriptors->get());
        vkCmdPushConstants(command->get(), pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Parameters),
                           &parameters);
        vkCmdDispatch(command->get(), x, y, 1);
    };
    if (tokens <= 4) {
        if ((uint32_t)rows > limits.maxComputeWorkGroupCount[1] ||
            (uint32_t)gemvGroups > limits.maxComputeWorkGroupCount[0])
            return NOT_SUPPORT;
        bindLinear(mGemvSet.get());
        mGemvSet->writeBuffer(backend->getBuffer(inputs[mRouteWeight ? 2 : 0]), 6);
        dispatch(mGemv, mGemvSet.get(), gemvGroups, mRouteWeight ? tokens : rows);
        return NO_ERROR;
    }
    const int64_t workspaceElements = 2LL * r.experts + 1 + rows + 2LL * parameters.maxTiles;
    if (workspaceElements > INT_MAX ||
        workspaceElements * sizeof(int) > backend->proty().limits.maxStorageBufferRange ||
        (uint32_t)parameters.maxTiles > limits.maxComputeWorkGroupCount[1] ||
        (uint32_t)gemmGroups > limits.maxComputeWorkGroupCount[0] ||
        (uint32_t)expertGroups > limits.maxComputeWorkGroupCount[0] ||
        (uint32_t)rowGroups > limits.maxComputeWorkGroupCount[0])
        return NOT_SUPPORT;
    // A smaller row count may select the smaller tile and require more routing
    // entries. Track the actual buffer capacity rather than only the row count.
    if (!mRouting || !mRouting->valid() || mRouting->size() < workspaceElements * sizeof(int)) {
        mRouting.reset(new VulkanBuffer(backend->getMemoryPool(), false, workspaceElements * sizeof(int), nullptr,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
        if (!mRouting->valid())
            return OUT_OF_MEMORY;
    }
    if (mRouteWeight) {
        const int64_t bytes = (int64_t)rows * r.outputChannels * (backend->useFP16() ? 2 : 4);
        if (bytes > limits.maxStorageBufferRange)
            return NOT_SUPPORT;
        if (!mRouteTmp || mRouteTmp->size() < bytes) {
            mRouteTmp.reset(new VulkanBuffer(backend->getMemoryPool(), false, bytes, nullptr,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
        }
        if (!mRouteTmp->valid())
            return OUT_OF_MEMORY;
    }
    for (int phase = 0; phase < 4; ++phase) {
        const auto pipeline = phase == 2 && tile == 1 ? mLargePrefix : mRoute[phase];
        const auto descriptors = phase == 2 && tile == 1 ? mLargePrefixSet.get() : mRouteSet[phase].get();
        descriptors->writeBuffer(backend->getBuffer(inputs[1]), 0);
        descriptors->writeBuffer(mRouting->buffer(), 1, mRouting->size());
        const int groups = phase == 0 ? expertGroups : (phase == 2 ? 1 : rowGroups);
        dispatch(pipeline, descriptors, groups, 1);
        command->barrierSource(mRouting->buffer(), 0, mRouting->size());
    }
    bindLinear(mGemmSet[tile].get());
    mGemmSet[tile]->writeBuffer(mRouting->buffer(), 6, mRouting->size());
    dispatch(mGemm[tile], mGemmSet[tile].get(), gemmGroups, parameters.maxTiles);
    if (mRouteWeight) {
        command->barrierSource(mRouteTmp->buffer(), 0, mRouteTmp->size());
        mReduceSet->writeBuffer(mRouteTmp->buffer(), 0, mRouteTmp->size());
        mReduceSet->writeBuffer(backend->getBuffer(inputs[2]), 1);
        mReduceSet->writeBuffer(backend->getBuffer(inputs[1]), 2);
        mReduceSet->writeBuffer(backend->getBuffer(outputs[0]), 3);
        dispatch(mReduce, mReduceSet.get(), UP_DIV(r.outputChannels, 64), tokens);
    }
    return NO_ERROR;
}

class VulkanIndexedLinearCreator : public VulkanBackend::Creator {
public:
    VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const Op* op, Backend* backend) const override {
        if (op->main_as_IndexedLinearParam()->routeWeight() == 3) {
            return nullptr;
        }
        // StaticModule creates the weight resource before input shape/type and
        // layout inference. Dynamic input validation belongs to onEncode.
        auto execution = new VulkanIndexedLinear(op, backend);
        if (!execution->valid()) {
            delete execution;
            return nullptr;
        }
        return execution;
    }
};

void registerVulkanIndexedLinearCreator() {
    static const bool registered = []() {
        static VulkanIndexedLinearCreator creator;
        VulkanBackend::addCreator(OpType_IndexedLinear, &creator);
        return true;
    }();
    (void)registered;
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
