//
//  VulkanPermute.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanPermute.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
struct GpuParam {
    ivec4 dims;
    ivec4 inImSize;
    ivec4 outImSize;
};

VulkanPermute::VulkanPermute(const Op* op, Backend* bn) : VulkanBasicExecution(bn), mTempSource(4), mTempDest(4) {
    auto newDim = op->main_as_Permute()->dims();
    for (int i = 0; i < newDim->size(); ++i) {
        mDims.push_back(newDim->data()[i]);
    }

    std::vector<VkDescriptorType> VulkanPermuteTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto extra = static_cast<VulkanBackend*>(bn);
    mVulkanPermutePipeline =
        extra->getPipeline("glsl_permute_comp", /*glsl_permute_comp, glsl_permute_comp_len,*/ VulkanPermuteTypes);
    mParamBuffer.reset(
        new VulkanBuffer(extra->getMemoryPool(), false, sizeof(GpuParam), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    mSourceTransform.reset(new VulkanImageConverter(extra));
    mDestTransform.reset(new VulkanImageConverter(extra));
}
VulkanPermute::~VulkanPermute() {
}
ErrorCode VulkanPermute::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(output->buffer().dimensions == 4);
    // acquire permute mid buffer
    TensorUtils::copyShape(input, &mTempSource);
    mTempSource.buffer().dim[1].flags                       = 0;
    TensorUtils::getDescribe(&mTempSource)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    TensorUtils::copyShape(output, &mTempDest);
    mTempDest.buffer().dim[1].flags                       = 0;
    TensorUtils::getDescribe(&mTempDest)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    backend()->onAcquireBuffer(&mTempDest, Backend::DYNAMIC);
    backend()->onAcquireBuffer(&mTempSource, Backend::DYNAMIC);

    // nc4hw4 -> nchw
    mSourceTransform->encodeTensorToBuffer(input, reinterpret_cast<VkBuffer>(mTempSource.deviceId()), input->size(), 0,
                                           TensorUtils::getDescribe(&mTempSource)->dimensionFormat, cmdBuffer);

    // set gpu permute parameter
    auto VulkanPermuteParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanPermuteParam, 0, sizeof(GpuParam));
    VulkanPermuteParam->dims[0]      = mDims[0];
    VulkanPermuteParam->dims[1]      = mDims[1];
    VulkanPermuteParam->dims[2]      = mDims[2];
    VulkanPermuteParam->dims[3]      = mDims[3];
    VulkanPermuteParam->inImSize[0]  = input->width();
    VulkanPermuteParam->inImSize[1]  = input->height();
    VulkanPermuteParam->inImSize[2]  = input->channel();
    VulkanPermuteParam->inImSize[3]  = input->batch();
    VulkanPermuteParam->outImSize[0] = output->width();
    VulkanPermuteParam->outImSize[1] = output->height();
    VulkanPermuteParam->outImSize[2] = output->channel();
    VulkanPermuteParam->outImSize[3] = output->batch();
    mParamBuffer->flush(true, 0, sizeof(GpuParam));
    mParamBuffer->unmap();

    // do permute
    mDescriptorSet.reset(mVulkanPermutePipeline->createSet());
    auto tempSourceSize = input->size();
    auto tempDestSize   = output->size();
    mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(mTempSource.deviceId()), 0, tempSourceSize);
    mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(mTempDest.deviceId()), 1, tempDestSize);
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 2, mParamBuffer->size());

    mVulkanPermutePipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(mTempSource.deviceId()), 0, tempSourceSize);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(mTempDest.width(), 8), UP_DIV(mTempDest.height(), 8),
                  UP_DIV(mTempDest.channel(), 1));

    cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(mTempDest.deviceId()), 0, tempDestSize);
    // nchw -> nc4hw4
    mDestTransform->encodeBufferToTensor(reinterpret_cast<VkBuffer>(mTempDest.deviceId()), output, output->size(), 0,
                                         TensorUtils::getDescribe(&mTempDest)->dimensionFormat, cmdBuffer);

    backend()->onReleaseBuffer(&mTempDest, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempSource, Backend::DYNAMIC);
    return NO_ERROR;
}

class VulkanPermuteCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanPermute(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Permute, new VulkanPermuteCreator);
    return true;
}();

} // namespace MNN
