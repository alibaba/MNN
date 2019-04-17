//
//  VulkanLRN.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanLRN.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
struct GpuParam {
    ivec4 imgSize;
    float alpha;
    float beta;
    int localSize;
};

VulkanLRN::VulkanLRN(const Op* op, Backend* bn) : VulkanReshape(bn), mTempTensor(4) {
    const auto lrnParam = op->main_as_LRN();
    mAlpha              = lrnParam->alpha();
    mBeta               = lrnParam->beta();
    mLocalSize          = lrnParam->localSize();
    MNN_ASSERT(lrnParam->regionType() == 0);
    std::vector<VkDescriptorType> VulkanLRNTypes{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto extra         = static_cast<VulkanBackend*>(bn);
    mVulkanLRNPipeline = extra->getPipeline(
        "glsl_lrnAcrossChannel_comp", /*glsl_lrnAcrossChannel_comp, glsl_lrnAcrossChannel_comp_len,*/ VulkanLRNTypes);
    mParamBuffer.reset(
        new VulkanBuffer(extra->getMemoryPool(), false, sizeof(GpuParam), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}
VulkanLRN::~VulkanLRN() {
}
ErrorCode VulkanLRN::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                              const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];

    this->setLayout(input, output);
    // nc4hw4 -> nchw
    mTensorConvert0->encodeTensorToBuffer(input, reinterpret_cast<VkBuffer>(mWrapTensorForInput.deviceId()),
                                          mWrapTensorForInput.size(), 0,
                                          TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat, cmdBuffer);
    // acquire lrn mid buffer

    mTempTensor.buffer().type = input->buffer().type;
    TensorUtils::copyShape(output, &mTempTensor);
    TensorUtils::getDescribe(&mTempTensor)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    mTempTensor.buffer().dim[1].flags                       = 0;
    backend()->onAcquireBuffer(&mTempTensor, Backend::DYNAMIC);

    // set gpu config
    auto VulkanLRNParam = reinterpret_cast<GpuParam*>(mParamBuffer->map());
    ::memset(VulkanLRNParam, 0, sizeof(GpuParam));
    VulkanLRNParam->imgSize[0] = input->width();
    VulkanLRNParam->imgSize[1] = input->height();
    VulkanLRNParam->imgSize[2] = input->channel();
    VulkanLRNParam->imgSize[3] = 0;
    VulkanLRNParam->alpha      = mAlpha / mLocalSize;
    VulkanLRNParam->beta       = mBeta;
    VulkanLRNParam->localSize  = mLocalSize;
    mParamBuffer->flush(true, 0, sizeof(GpuParam));
    mParamBuffer->unmap();

    // do lrn
    mDescriptorSet.reset(mVulkanLRNPipeline->createSet());
    mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(mTempTensor.deviceId()), 0, mTempTensor.size());
    mDescriptorSet->writeBuffer(reinterpret_cast<VkBuffer>(mWrapTensorForInput.deviceId()), 1,
                                mWrapTensorForInput.size());
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 2, mParamBuffer->size());
    mVulkanLRNPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(mWrapTensorForInput.deviceId()), 0, mWrapTensorForInput.size());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(mTempTensor.width(), 16), UP_DIV(mTempTensor.height(), 16),
                  mTempTensor.channel() * mTempTensor.batch());

    // nchw -> nc4hw4
    mTensorConvert1->encodeBufferToTensor(reinterpret_cast<VkBuffer>(mTempTensor.deviceId()), output,
                                          mTempTensor.size(), 0,
                                          TensorUtils::getDescribe(&mTempTensor)->dimensionFormat, cmdBuffer);

    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempTensor, Backend::DYNAMIC);
    return NO_ERROR;
}

class VulkanLRNCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanLRN(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LRN, new VulkanLRNCreator);
    return true;
}();

} // namespace MNN
