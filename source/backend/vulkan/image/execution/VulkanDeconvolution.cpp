//
//  VulkanDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanDeconvolution.hpp"
#include "core/Macro.h"
namespace MNN {
static void writeReorderBuffer(VulkanMatMul::Reorder::nchwBuffer& buffer, int co, int ci, int kh, int kw) {
    buffer.size[0] = co;
    buffer.size[1] = ci;
    buffer.size[2] = kh;
    buffer.size[3] = kw;
    buffer.stride[0] = kh * kw;
    buffer.stride[1] = kh * kw * co;
    buffer.stride[2] = kw;
    buffer.stride[3] = 1;
}

VulkanDeconvolution::VulkanDeconvolution(Backend* bn, const std::vector<Tensor*>& inputs, const Convolution2D* conv) : VulkanBasicExecution(bn) {
    mConvCommonOption = conv->common();
    auto vkBn         = (VulkanBackend*)bn;
    mConvParam = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false,
                                                sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    int kh     = mConvCommonOption->kernelY();
    int kw     = mConvCommonOption->kernelX();
    int co     = mConvCommonOption->outputCount();
    int ci     = inputs[0]->channel();

    const float* filterDataPtr = nullptr;
    int tempWeightSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, bn, conv, &filterDataPtr, &tempWeightSize);

    if (nullptr != filterDataPtr) {
        MNN_ASSERT(inputs.size() == 1);
        std::shared_ptr<VulkanBuffer> origin(new VulkanBuffer(vkBn->getMemoryPool(), false, ci * kh * kw * co * sizeof(float), filterDataPtr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        std::shared_ptr<VulkanBuffer> midBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, co * kh * kw * ALIGN_UP4(ci) * sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto kernel = VulkanMatrixMultier4x4::createKernel(vkBn, nullptr, ci,  ALIGN_UP4(co) * kh * kw, 1);
        VulkanMatMul::Reorder::nchwBuffer parameters;
        writeReorderBuffer(parameters, co, ci, kh, kw);
        VulkanMatMul::Reorder reorder(vkBn, true, false);
        std::shared_ptr<VulkanCommandPool::Buffer> cmdBuffer(vkBn->getPool().allocBuffer());
        cmdBuffer->begin(0);
        reorder.encode(origin->buffer(), origin->size(), midBuffer->buffer(), midBuffer->size(), kernel.get(), cmdBuffer.get(), parameters);
        cmdBuffer->end();
        vkBn->getPool().submitAndWait(cmdBuffer->get());

        mMultiler.reset(new VulkanMatrixMultier4x4(vkBn, nullptr, ALIGN_UP4(ci), ALIGN_UP4(co) * kh * kw, 1, kernel));
    }
    if (inputs.size() < 3) {
        int outputC4      = UP_DIV(mConvCommonOption->outputCount(), 4);
        mBias             = std::make_shared<VulkanImage>(vkBn->getMemoryPool(), false, std::vector<int>{outputC4, 1});
        auto biasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, outputC4 * 4 * sizeof(float));
        auto biasPtr    = biasBuffer->map();
        ::memset(biasPtr, 0, outputC4 * 4 * sizeof(float));
        if (nullptr != conv->bias()) {
            ::memcpy(biasPtr, conv->bias()->data(), conv->bias()->size() * sizeof(float));
        }
        biasBuffer->unmap();
        vkBn->copyBufferToImage(biasBuffer.get(), mBias.get());
    }
    {
        std::vector<VkDescriptorType> im2ColTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        auto macro = VulkanConvolutionCommon::getPostTreatMacro(mConvCommonOption);
        mIm2Col    = vkBn->getPipeline("glsl_deconvIm2Col_" + macro + "comp", im2ColTypes);
        mIm2ColSet.reset(mIm2Col->createSet());
    }
    {
        std::vector<VkDescriptorType> col2ImTypes{
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mCol2Im = vkBn->getPipeline("glsl_deconvCol2Im_comp", col2ImTypes);
        mCol2ImSet.reset(mCol2Im->createSet());
    }
    mSampler = vkBn->getCommonSampler();
}

void VulkanDeconvolution::writeConvolutionConst(VulkanConvolutionCommon::ConvolutionParameter* convCons,
                                                const Convolution2DCommon* common, const Tensor* src,
                                                const Tensor* dst) {
    const int icDiv4 = UP_DIV(src->channel(), 4);
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    auto pad = ConvolutionCommon::convolutionTransposePad(src, dst, common);
    int padX         = pad.first;
    int padY         = pad.second;
    convCons->dilate[0]     = common->dilateX();
    convCons->dilate[1]     = common->dilateY();
    convCons->stride[0]     = common->strideX();
    convCons->stride[1]     = common->strideY();
    convCons->pad[0]        = padX;
    convCons->pad[1]        = padY;
    convCons->kernelSize[0] = common->kernelX();
    convCons->kernelSize[1] = common->kernelY();

    convCons->inputSize[0] = src->width();
    convCons->inputSize[1] = src->height();
    convCons->inputSize[2] = icDiv4;
    convCons->inputSize[3] = src->batch();

    convCons->outputSize[0] = dst->width();
    convCons->outputSize[1] = dst->height();
    convCons->outputSize[2] = ocDiv4;
    convCons->outputSize[3] = dst->batch();
}

ErrorCode VulkanDeconvolution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const VulkanCommandPool::Buffer* cmdBuffer) {
    auto src         = inputs[0];
    auto dst         = outputs[0];
    const int icDiv4 = UP_DIV(src->channel(), 4);
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    auto vkBn = (VulkanBackend*)backend();
    {
        auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParam->map());
        writeConvolutionConst(convCons, mConvCommonOption, src, dst);
        convCons->outputSize[3] = src->batch();
        mConvParam->unmap();
    }

    mMultiler->prepare(static_cast<VulkanBackend*>(backend())->getInitCommandBuffer(), src->width() * src->height() * src->batch());
    if (true) {
        auto totalInputSize = src->width() * src->height() * icDiv4 * src->batch();
        auto dstImage = mMultiler->source();
        mCol2ImSet->writeImage((reinterpret_cast<VulkanTensor*>(src->deviceId()))->image()->view(), mSampler->get(),
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
        mCol2ImSet->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 1);

        mCol2ImSet->writeBuffer(mConvParam->buffer(), 2, mConvParam->size());
        mCol2Im->bind(cmdBuffer->get(), mCol2ImSet->get());

        dstImage->barrierWrite(cmdBuffer->get());
        (reinterpret_cast<VulkanTensor*>(src->deviceId()))->image()->barrierRead(cmdBuffer->get());
        
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalInputSize, VulkanConvolutionCommon::gImage2ColLocal), 1, 1);
    }

    mMultiler->compute(cmdBuffer);
    if (inputs.size() > 1) {
        mKernel->release();
    }

    if (true) {
        auto dstImage = mMultiler->dest();
        auto totalSize = dst->width() * dst->height() * ocDiv4 * src->batch();

        mIm2ColSet->writeImage((reinterpret_cast<VulkanTensor*>(dst->deviceId()))->image()->view(), mSampler->get(),
                               VK_IMAGE_LAYOUT_GENERAL, 0);
        mIm2ColSet->writeImage(dstImage->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        mIm2ColSet->writeImage(mBias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
        mIm2ColSet->writeBuffer(mConvParam->buffer(), 3, mConvParam->size());
        mIm2Col->bind(cmdBuffer->get(), mIm2ColSet->get());

        dstImage->barrierRead(cmdBuffer->get());
        mBias->barrierRead(cmdBuffer->get());
        reinterpret_cast<VulkanTensor*>(dst->deviceId())->image()->barrierWrite(cmdBuffer->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, VulkanConvolutionCommon::gImage2ColLocal), 1, 1);
    }
    if (inputs.size() > 2) {
        mBias->release();
    }
    return NO_ERROR;
}
class VulkanDeconvolutionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new VulkanDeconvolution(backend, inputs, op->main_as_Convolution2D());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Deconvolution, new VulkanDeconvolutionCreator);
    return true;
}();
} // namespace MNN
