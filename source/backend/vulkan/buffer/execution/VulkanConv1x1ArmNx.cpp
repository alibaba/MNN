#ifdef MNN_VULKAN_ARM_NX
#include "VulkanConv1x1ArmNx.hpp"
#include "VulkanBackend.hpp"
#include "VulkanConv1x1General.hpp"
#include <cstring>

namespace MNN {
namespace {
struct Params {
    uint32_t rows, ci, co, blocks, blockSize, activation;
};
std::shared_ptr<VulkanBuffer> buffer(VulkanBackend* backend, size_t bytes, const void* data = nullptr) {
    auto result = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, (bytes + 3) & ~size_t(3), nullptr,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (data)
        backend->copyToGPUBuffer(data, result->buffer(), bytes, 0);
    return result;
}
bool trace() {
    const char* value = getenv("MNN_VULKAN_ARM_NX_TRACE");
    return value && !strcmp(value, "1");
}
} // namespace

VulkanBasicExecution* VulkanConv1x1ArmNx::create(VulkanBackend* backend, const Convolution2DCommon* common,
                                                 const float* bias, int ci, int co,
                                                 std::shared_ptr<ConvolutionCommon::Int8Common> quant) {
    if (!backend->getDevice().supportsArmNx() || !backend->getDevice().getInt8Support() || !quant ||
        !quant->weight.get() || ci <= 0 || co <= 0 || co > INT32_MAX / 2 || quant->canUseInt2 || quant->canUseInt3)
        return nullptr;
    const int soSize = quant->asymmetric ? 2 : 1;
    const int alphaSize = quant->alphaSize;
    if (alphaSize <= 0 || alphaSize % (co * soSize) != 0)
        return nullptr;
    const int blocks = alphaSize / (co * soSize);
    if (ci % blocks || (ci / blocks) % 4 || ci / blocks > INT32_MAX / (127 * 128))
        return nullptr;
    const size_t count = static_cast<size_t>(ci) * co;
    if (count > UINT32_MAX / 4 || quant->weight.size() < (quant->canUseInt4 ? (count + 1) / 2 : count))
        return nullptr;
    auto resource = std::make_shared<Resource>();
    resource->device = backend->getDevice().get();
    resource->ci = ci;
    resource->co = co;
    resource->blocks = blocks;
    resource->blockSize = ci / blocks;
    resource->activation = common->relu6() ? 2 : (common->relu() ? 1 : 0);
    resource->common = common;
    resource->quant = quant;
    std::vector<int8_t> weights((count + 3) & ~size_t(3), 0);
    const auto* src = reinterpret_cast<const uint8_t*>(quant->weight.get());
    for (int n = 0; n < co; ++n) {
        for (int k = 0; k < ci; ++k) {
            const size_t index = static_cast<size_t>(n) * ci + k;
            const int value =
                quant->canUseInt4 ? int((src[index / 2] >> (index % 2 ? 0 : 4)) & 15) - 8 : quant->weight.get()[index];
            const size_t dst = static_cast<size_t>(k) * co + n;
            weights[dst] = value;
        }
    }
    const auto* alpha = quant->getAlphaFloat();
    std::vector<float> meta(static_cast<size_t>(co) * blocks * 2);
    for (int i = 0; i < co * blocks; ++i) {
        meta[2 * i] = alpha[soSize * i + soSize - 1];
        meta[2 * i + 1] = soSize == 2 ? alpha[2 * i] : 0.0f;
    }
    resource->biasHost.resize((co + 3) & ~3, 0.0f);
    if (bias)
        memcpy(resource->biasHost.data(), bias, co * sizeof(float));
    resource->weight = buffer(backend, weights.size(), weights.data());
    resource->meta = buffer(backend, meta.size() * sizeof(float), meta.data());
    resource->bias = buffer(backend, resource->biasHost.size() * sizeof(float), resource->biasHost.data());
    return new VulkanConv1x1ArmNx(backend, std::move(resource));
}

VulkanConv1x1ArmNx::VulkanConv1x1ArmNx(VulkanBackend* backend, std::shared_ptr<Resource> resource)
    : VulkanBasicExecution(backend), mResource(std::move(resource)) {
    const auto suffix = backend->useFP16() ? "_FP16_comp" : "_comp";
    mQuantize = backend->getPipeline(std::string("glsl_arm_nx_quantize") + suffix,
                                     std::vector<VkDescriptorType>(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    mDequantize = backend->getPipeline(std::string("glsl_arm_nx_dequantize") + suffix,
                                       std::vector<VkDescriptorType>(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
    if (mQuantize)
        mQuantizeSet.reset(mQuantize->createSet());
    if (mDequantize)
        mDequantizeSet.reset(mDequantize->createSet());
}

bool VulkanConv1x1ArmNx::onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) {
    // StaticModule probes clone support with both backend and destination null.
    if (!dst)
        return true;
    if (!backend || static_cast<VulkanBackend*>(backend)->getDevice().get() != mResource->device)
        return false;
    *dst = new VulkanConv1x1ArmNx(static_cast<VulkanBackend*>(backend), mResource);
    return true;
}

ErrorCode VulkanConv1x1ArmNx::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                       const VulkanCommandPool::Buffer* cmd) {
    auto* vk = static_cast<VulkanBackend*>(backend());
    const auto& r = *mResource;
    const uint32_t rows = inputs[0]->batch() * inputs[0]->height() * inputs[0]->width();
    if (mRows != rows) {
        mArmNx = VulkanArmNx::createMatMul(vk->getDevice(), rows, r.blockSize, r.co, true, r.blocks);
        mRows = rows;
        if (mArmNx) {
            mActivation = buffer(vk, static_cast<size_t>(rows) * r.ci);
            mActivationMeta = buffer(vk, static_cast<size_t>(rows) * r.blocks * 2 * sizeof(float));
            mAccumulator = buffer(vk, static_cast<size_t>(rows) * r.blocks * r.co * sizeof(int32_t));
        }
    }
    if (!mArmNx || !mQuantizeSet || !mDequantizeSet) {
        if (!mFallback) {
            MNN_ERROR("ARM NX Linear fallback K=%u N=%u M=%u\n", r.ci, r.co, rows);
            mFallback.reset(new VulkanConv1x1General(vk, r.common, r.biasHost.data(), r.ci, r.co, r.quant));
        }
        return mFallback->onEncode(inputs, outputs, cmd);
    }
    if (trace())
        MNN_PRINT("ARM NX Linear INT8: M=%u K=%u N=%u groups=%u\n", rows, r.ci, r.co, r.blocks);
    const auto input = vk->getTensorBuffer(inputs[0]);
    const auto output = vk->getTensorBuffer(outputs[0]);
    mQuantizeSet->writeBuffer(input.first->buffer(), 0, vk->getTensorSize(inputs[0]), input.second);
    mQuantizeSet->writeBuffer(mActivation->buffer(), 1, mActivation->size());
    mQuantizeSet->writeBuffer(mActivationMeta->buffer(), 2, mActivationMeta->size());
    Params params{rows, r.ci, r.co, r.blocks, r.blockSize, r.activation};
    cmd->barrierSource(input.first->buffer(), input.second, vk->getTensorSize(inputs[0]));
    mQuantize->bind(cmd->get(), mQuantizeSet->get());
    vkCmdPushConstants(cmd->get(), mQuantize->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    vkCmdDispatch(cmd->get(), UP_DIV(rows * r.blocks, 64), 1, 1);
    mArmNx->encode(cmd->get(), mActivation->buffer(), 0, r.weight->buffer(), 0, mAccumulator->buffer(), 0);
    mDequantizeSet->writeBuffer(mAccumulator->buffer(), 0, mAccumulator->size());
    mDequantizeSet->writeBuffer(mActivationMeta->buffer(), 1, mActivationMeta->size());
    mDequantizeSet->writeBuffer(r.meta->buffer(), 2, r.meta->size());
    mDequantizeSet->writeBuffer(r.bias->buffer(), 3, r.bias->size());
    mDequantizeSet->writeBuffer(output.first->buffer(), 4, vk->getTensorSize(outputs[0]), output.second);
    mDequantize->bind(cmd->get(), mDequantizeSet->get());
    vkCmdPushConstants(cmd->get(), mDequantize->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    vkCmdDispatch(cmd->get(), UP_DIV(rows * UP_DIV(r.co, 4), 64), 1, 1);
    return NO_ERROR;
}
} // namespace MNN
#endif
