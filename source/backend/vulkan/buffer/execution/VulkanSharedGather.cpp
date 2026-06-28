#include "VulkanSharedGather.hpp"
#include "core/Macro.h"

#ifdef ENABLE_VULKAN_TIME_PROFILE
#include "backend/vulkan/component/VulkanTimeProfiler.hpp"
#endif

namespace MNN {

struct SharedGatherParam {
    uint32_t selectSize;
    uint32_t ic;
    uint32_t oc;
    uint32_t blockSize;
    uint32_t blockStride;
    uint32_t weightStride;
    uint32_t quantBits;
    uint32_t pad;
};

VulkanSharedGather::VulkanSharedGather(VulkanBackend* backend, int ci, int co, int quantBits, uint32_t padN,
                                       uint32_t blockSize, uint32_t blockStride, uint32_t weightStride, bool offsetZero,
                                       std::shared_ptr<VulkanBuffer> weight, std::shared_ptr<VulkanBuffer> meta)
    : VulkanBasicExecution(backend),
      mCi(ci),
      mCo(co),
      mQuantBits(quantBits),
      mPadN(padN),
      mBlockSize(blockSize),
      mBlockStride(blockStride),
      mWeightStride(weightStride),
      mOffsetZero(offsetZero ? 1u : 0u),
      mWeight(std::move(weight)),
      mMeta(std::move(meta)) {
    auto vkBn = static_cast<VulkanBackend*>(backend);
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    std::string key = "glsl_shared_gather_";
    if (vkBn->useFP16()) {
        key += "FP16_";
    }
    key += "comp";
    std::vector<uint32_t> spec{
        static_cast<uint32_t>(mQuantBits),
        mBlockSize,
        mBlockStride,
        mOffsetZero,
    };
    mPipeline = vkBn->getPipeline(key, types, {mLocalSize, 1u, 1u}, spec);
    if (nullptr != mPipeline) {
        mDescriptorSet.reset(mPipeline->createSet());
    }
}

VulkanSharedGather::~VulkanSharedGather() {}

bool VulkanSharedGather::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return nullptr != mPipeline && nullptr != mDescriptorSet && nullptr != mWeight && nullptr != mMeta;
    }
    *dst = new VulkanSharedGather(static_cast<VulkanBackend*>(bn), mCi, mCo, mQuantBits, mPadN, mBlockSize,
                                  mBlockStride, mWeightStride, mOffsetZero != 0u, mWeight, mMeta);
    return true;
}

ErrorCode VulkanSharedGather::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                       const VulkanCommandPool::Buffer* cmdBuffer) {
    if (inputs.empty() || outputs.empty() || nullptr == mPipeline || nullptr == mDescriptorSet || nullptr == mWeight ||
        nullptr == mMeta) {
        return NOT_SUPPORT;
    }
    if (mCi <= 0 || mCo <= 0 || mBlockSize == 0 || mBlockStride == 0 || mWeightStride == 0) {
        return NOT_SUPPORT;
    }
    if (mQuantBits != 4 && mQuantBits != 8) {
        return NOT_SUPPORT;
    }

    auto vkBn = static_cast<VulkanBackend*>(backend());
    const uint32_t selectSize = static_cast<uint32_t>(inputs[0]->elementSize());
    SharedGatherParam param;
    param.selectSize = selectSize;
    param.ic = static_cast<uint32_t>(mCi);
    param.oc = static_cast<uint32_t>(mCo);
    param.blockSize = mBlockSize;
    param.blockStride = mBlockStride;
    param.weightStride = mWeightStride;
    param.quantBits = static_cast<uint32_t>(mQuantBits);
    param.pad = mPadN;

    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputs[0]), 0);
    mDescriptorSet->writeBuffer(mWeight->buffer(), 1, mWeight->size());
    mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[0]), 2);
    mDescriptorSet->writeBuffer(mMeta->buffer(), 3, mMeta->size());

#ifdef ENABLE_VULKAN_TIME_PROFILE
    auto* profiler = vkBn->timeProfiler();
    if (nullptr != profiler) {
        VulkanTimeProfileScope scope(profiler, cmdBuffer->get(),
                                     vkBn->useFP16() ? "glsl_shared_gather_FP16_comp" : "glsl_shared_gather_comp",
                                     VulkanTimeProfiler::Kind::Shader);
        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdPushConstants(cmdBuffer->get(), mPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(param),
                           &param);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(UP_DIV(static_cast<uint32_t>(mCi), 4u), mLocalSize), selectSize, 1);
        return NO_ERROR;
    }
#endif
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdPushConstants(cmdBuffer->get(), mPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(param), &param);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(UP_DIV(static_cast<uint32_t>(mCi), 4u), mLocalSize), selectSize, 1);
    return NO_ERROR;
}

} // namespace MNN
