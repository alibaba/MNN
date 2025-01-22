//
//  VulkanConvolutionImpl.hpp
//  MNN
//
//  Created by MNN on 2025/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolution1x1.hpp"
#include "VulkanConvolution.hpp"
#include "VulkanMatMul.hpp"

namespace MNN{

static void writeParameters(VulkanMatMul::Reorder::nchwBuffer& parameters, int co, int ci, int kh, int kw) {
    parameters.size[0] = co;
    parameters.size[1] = ci;
    parameters.size[2] = kh;
    parameters.size[3] = kw;
    parameters.stride[0] = ci * kh * kw;
    parameters.stride[1] = kh * kw;
    parameters.stride[2] = kw;
    parameters.stride[3] = 1;
}

struct VulkanImageConv1x1Param {
    ivec4 inputSize;
    ivec4 outputSize;
};

VulkanConvolution1x1::VulkanConvolution1x1(VulkanBackend* vkBn, const Convolution2DCommon* convCommon, const float* weightPtr, const float* biasPtr, const int ic, const int oc) : VulkanBasicExecution(vkBn) {
    mConvCommon = convCommon;

    mConv1x1Param = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(VulkanImageConv1x1Param), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    auto macroRelu = VulkanConvolutionCommon::getPostTreatMacro(mConvCommon);
    bool useFP16 = (vkBn->gpuType() == VulkanRuntime::ADRENO || vkBn->gpuType() == VulkanRuntime::MALI) && vkBn->getMemoryPool().permitFp16();
    std::string macroPrecision = useFP16 ? "FP16_" : "FP32_";
    std::string macro = macroRelu + macroPrecision;

    mCands.push_back(vkBn->getPrivatePipeline("glsl_convolution1x1_" + macro + "comp", types));
    mCands.push_back(vkBn->getPrivatePipeline("glsl_convolution1x1_w4_" + macro + "comp", types));
    mCands.push_back(vkBn->getPrivatePipeline("glsl_convolution1x1_c8w4_" + macro + "comp", types));

    mDescriptorSet.reset(mCands[0]->createSet());

    // write mBias
    mBias.reset(new VulkanImage(vkBn->getMemoryPool(), false, {UP_DIV(oc, 4), 1}));
    auto biasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(float) * ALIGN_UP4(oc));
    auto bias = biasBuffer->map();
    ::memset(bias, 0, sizeof(float) * ALIGN_UP4(oc));
    if (biasPtr != nullptr) {
        ::memcpy(bias, biasPtr, sizeof(float) * ALIGN_UP4(oc));
    }
    biasBuffer->unmap();
    vkBn->copyBufferToImage(biasBuffer.get(), mBias.get(), VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);

{
    if (nullptr != weightPtr) {
        size_t weightSize = sizeof(float) * ALIGN_UP4(ic) * ALIGN_UP4(oc);
        std::shared_ptr<VulkanBuffer> kernelStageBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, weightSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto kernelPtr = (float*)kernelStageBuffer->map();
        ::memset(kernelPtr, 0, weightSize);
        for (int indexOc = 0; indexOc < oc; indexOc++) {
            int indexOcOut = indexOc / 4;
            int indexOcIn = indexOc % 4;
            for (int indexIc = 0; indexIc < ic; indexIc++) {
                int dstOffset = indexOcIn + indexIc * 4 + 4 * ALIGN_UP4(ic) * indexOcOut;
                kernelPtr[dstOffset] = weightPtr[indexIc + indexOc * ic];
            }
        }
        kernelStageBuffer->unmap();
        mKernel = std::make_shared<VulkanImage>(vkBn->getMemoryPool(), false, ALIGN_UP4(ic), UP_DIV(oc, 4));
        vkBn->copyBufferToImage(kernelStageBuffer.get(), mKernel.get());
    }
}

}


VulkanConvolution1x1::~VulkanConvolution1x1() {
    // Do nothing
}


ErrorCode VulkanConvolution1x1::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = (VulkanBackend*)backend();

    auto input = inputs[0];
    auto output = outputs[0];

    int icDiv4 = UP_DIV(input->channel(), 4);
    int ocDiv4 = UP_DIV(output->channel(), 4);

    {
        auto conv1x1Param = reinterpret_cast<VulkanImageConv1x1Param*>(mConv1x1Param->map());
        conv1x1Param->inputSize[0] = input->width();
        conv1x1Param->inputSize[1] = input->height();
        conv1x1Param->inputSize[2] = icDiv4;
        conv1x1Param->inputSize[3] = input->batch();
        conv1x1Param->outputSize[0] = output->width();
        conv1x1Param->outputSize[1] = output->height();
        conv1x1Param->outputSize[2] = ocDiv4;
        conv1x1Param->outputSize[3] = output->batch();
        mConv1x1Param->unmap();
    }

    mDescriptorSet->writeImage(((VulkanTensor*)output->deviceId())->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(((VulkanTensor*)input->deviceId())->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage(mKernel->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mDescriptorSet->writeImage(mBias->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3);
    mDescriptorSet->writeBuffer(mConv1x1Param->buffer(), 4, mConv1x1Param->size());

    mKernel->barrierRead(cmdBuffer->get());
    mBias->barrierRead(cmdBuffer->get());
    ((VulkanTensor*)input->deviceId())->image()->barrierRead(cmdBuffer->get());
    ((VulkanTensor*)output->deviceId())->image()->barrierWrite(cmdBuffer->get());

    std::vector<uint32_t> gws, lws;

    mCandidataGws.push_back({output->width() * output->height() * output->batch(), ocDiv4, 1});
    mCandidataGws.push_back({UP_DIV(output->width(), 4) * output->height() * output->batch(), ocDiv4, 1});
    mCandidataGws.push_back({UP_DIV(output->width(), 4) * output->height() * output->batch(), UP_DIV(ocDiv4, 2), 1});

    float costMin = -1.0f;
    float costCurr;
    int optimalIndex = -1;
    for (int i = 0; i < mCands.size(); i++) {
        auto lwsCurr = vkBn->autoTunePipeline(mCands[i], mDescriptorSet, {(uint32_t)mCandidataGws[i][0], (uint32_t)mCandidataGws[i][1], (uint32_t)mCandidataGws[i][2]}, 2, {8, 8, 1}, &costCurr);
        if (costCurr < costMin || costMin < 0) {
            optimalIndex = i;
            costMin = costCurr;
            lws = lwsCurr;
        }
    }
    mPipeline = mCands[optimalIndex];
    gws = {(uint32_t)mCandidataGws[optimalIndex][0], (uint32_t)mCandidataGws[optimalIndex][1], (uint32_t)mCandidataGws[optimalIndex][2]};


    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(gws[0], lws[0]), UP_DIV(gws[1], lws[1]), UP_DIV(gws[2], lws[2]));

    return NO_ERROR;
}

} // end namespace MNN
