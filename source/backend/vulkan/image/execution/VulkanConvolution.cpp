//
//  VulkanConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolution.hpp"
#include "core/Macro.h"
#include "VulkanConvolutionImpl.hpp"
#include "core/ConvolutionCommon.hpp"
namespace MNN {
int VulkanConvolutionCommon::gImage2ColLocal = 256;
std::string VulkanConvolutionCommon::getPostTreatMacro(const Convolution2DCommon* common) {
    if (common->relu()) {
        return "RELU_";
    } else if (common->relu6()) {
        return "RELU6_";
    }
    return "";
}

static std::shared_ptr<VulkanBuffer> _createBufferForConvDepthwise(VulkanBackend* extra,
                                                                   const Convolution2DCommon* mCommon,
                                                                   const float* weightSource, size_t weightSize) {
    auto outputCount     = mCommon->outputCount();
    auto totalWeightSize = ALIGN_UP4(mCommon->outputCount()) * (mCommon->kernelY() * mCommon->kernelX());
    auto kernelBuffer    = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * totalWeightSize, nullptr,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto layer        = mCommon;

    auto weight     = (float*)kernelBuffer->map();
    int kw          = layer->kernelX();
    int kh          = layer->kernelY();
    int planeStride = kw * kh * 4;

    int cur = 0;
    for (int c = 0; c < outputCount; ++c) {
        int plane  = c / 4;
        int offset = c % 4;
        for (int y = 0; y < kh; ++y) {
            for (int x = 0; x < kw; ++x) {
                float* dst = weight + offset + (x + y * kw) * 4 + planeStride * plane;
                *dst       = weightSource[cur++];
            }
        }
    }
    kernelBuffer->unmap();
    return kernelBuffer;
}

void VulkanConvolutionCommon::writeParameter(ConvolutionParameter* convCons, const Convolution2DCommon* common,
                                             const Tensor* input, const Tensor* output) {
    int icDiv4 = UP_DIV(input->channel(), 4);
    int ocDiv4 = UP_DIV(output->channel(), 4);
    auto pad = ConvolutionCommon::convolutionPad(input, output, common);
    int padX   = pad.first;
    int padY   = pad.second;
    {
        convCons->dilate[0]     = common->dilateX();
        convCons->dilate[1]     = common->dilateY();
        convCons->stride[0]     = common->strideX();
        convCons->stride[1]     = common->strideY();
        convCons->pad[0]        = padX;
        convCons->pad[1]        = padY;
        convCons->kernelSize[0] = common->kernelX();
        convCons->kernelSize[1] = common->kernelY();

        convCons->inputSize[0] = input->width();
        convCons->inputSize[1] = input->height();
        convCons->inputSize[2] = icDiv4;
        convCons->inputSize[3] = input->batch();

        convCons->outputSize[0] = output->width();
        convCons->outputSize[1] = output->height();
        convCons->outputSize[2] = ocDiv4;
        convCons->outputSize[3] = output->batch();
        convCons->offset[0]     = 0;
        convCons->offset[1]     = 0;
        convCons->offset[2]     = output->height();
    }
}

VulkanConvolutionCommon::VulkanConvolutionCommon(const Op* convOp, Backend* bn) : VulkanBasicExecution(bn) {
    auto extra    = static_cast<VulkanBackend*>(bn);
    mCommon       = convOp->main_as_Convolution2D()->common();
    mConvCons = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(ConvolutionParameter), nullptr,
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanConvolutionCommon::~VulkanConvolutionCommon() {
}

ErrorCode VulkanConvolutionCommon::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    {
        auto convCons = (ConvolutionParameter*)mConvCons->map();
        writeParameter(convCons, mCommon, input, output);
        mConvCons->unmap();
    }

    auto code = this->onEncodeConvolution(mCommon, inputs, outputs, cmdBuffer, mConvCons.get());
    if (NO_ERROR != code) {
        return code;
    }
    return NO_ERROR;
}
bool VulkanConvolutionDepthwise::_init(const float* weightData, size_t weightSize, const Op* convOp, Backend* bn) {
    auto extra      = static_cast<VulkanBackend*>(bn);
    auto common    = convOp->main_as_Convolution2D()->common();
    mSampler        = extra->getCommonSampler();
    // Create Pipeline
    std::vector<VkDescriptorType> convTypes{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    MNN_ASSERT(OpType_ConvolutionDepthwise == convOp->type());
    auto macro = getPostTreatMacro(common);
    if (extra->gpuType() == VulkanRuntime::ADRENO) {
        mConvPipeline = extra->getPipeline("glsl_convolutionDepthwise_" + macro + "comp", convTypes);
        mLocalX       = 16;
        mLocalY       = 16;
    } else {
        mConvPipeline = extra->getPipeline("glsl_convolutionDepthwiseMali_" + macro + "comp", convTypes);
        mLocalX       = 8;
        mLocalY       = 8;
    }
    auto c4 = UP_DIV(common->outputCount(), 4);
    mKernel = std::make_shared<VulkanImage>(extra->getMemoryPool(), false, common->kernelX() * common->kernelY(), c4);
    if (nullptr != weightData){
        auto tempBuffer = _createBufferForConvDepthwise(extra, common, weightData, weightSize);
        extra->copyBufferToImage(tempBuffer.get(), mKernel.get());
    }
    auto convReal = convOp->main_as_Convolution2D();
    mBias.reset(new VulkanImage(extra->getMemoryPool(), false, {c4, 1}));
    auto biasBuffer = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false,
                                                     sizeof(float) * ALIGN_UP4(common->outputCount()));

    auto bias = biasBuffer->map();
    ::memset(bias, 0, ALIGN_UP4(common->outputCount()) * sizeof(float));
    if (nullptr != convReal->bias()) {
        // Create Buffer
        ::memcpy(bias, convReal->bias()->data(), common->outputCount() * sizeof(float));
    }
    biasBuffer->unmap();
    extra->copyBufferToImage(biasBuffer.get(), mBias.get());
    return true;
}


VulkanConvolutionDepthwise::VulkanConvolutionDepthwise(const float* weightData, size_t weightSize, const Op* convOp, Backend* bn)
    : VulkanConvolutionCommon(convOp, bn) {
    _init(weightData, weightSize, convOp, bn);
}

VulkanConvolutionDepthwise::~VulkanConvolutionDepthwise() {
}

ErrorCode VulkanConvolutionDepthwise::onEncodeConvolution(const Convolution2DCommon* common,
                                                          const std::vector<Tensor*>& inputs,
                                                          const std::vector<Tensor*>& outputs,
                                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                                          const VulkanBuffer* convCons) {
    auto input  = inputs[0];
    auto output = outputs[0];
    /*Set Const Parameters*/
    int ocDiv4 = UP_DIV(output->channel(), 4);
    int ow     = output->width();
    int oh     = output->height();
    auto extra = static_cast<VulkanBackend*>(backend());
    mExtraSets.clear();
    mExtraBuffers.clear();
    if (inputs.size() >= 2) {
        auto weight = reinterpret_cast<VulkanTensor*>(inputs[1]->deviceId())->image();
        auto pipeline = extra->getPipeline("glsl_dwweightcopy_comp", {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        });
        std::shared_ptr<VulkanLayout::DescriptorSet> des(pipeline->createSet());
        des->writeImage(weight->view(), extra->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        des->writeImage(mKernel->view(), extra->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        weight->barrierRead(cmdBuffer->get());
        mKernel->barrierWrite(cmdBuffer->get());
        int dim[4] = {
            weight->width(),
            weight->height(),
            inputs[1]->height(),
            weight->depth() * weight->height() * weight->width()
        };
        std::shared_ptr<VulkanBuffer> uniforms(new VulkanBuffer(extra->getMemoryPool(), false, sizeof(dim), &dim, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniforms->buffer(), 2, uniforms->size());
        pipeline->bind(cmdBuffer->get(), des->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(dim[3], 256), 1, 1);
        mExtraBuffers.emplace_back(uniforms);
        mExtraSets.emplace_back(des);
    }
    const VulkanImage* bias;
    if (inputs.size() >= 3) {
        bias = reinterpret_cast<VulkanTensor*>(inputs[2]->deviceId())->image();
    } else {
        bias = mBias.get();
    }
    if (nullptr == bias) {
        mBias.reset(new VulkanImage(extra->getMemoryPool(), false, {1, 1}));
        // Create Buffer
        auto biasBuffer = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false,
                                                         sizeof(float) * 4);
        auto biasPtr = biasBuffer->map();
        ::memset(biasPtr, 0, 4 * sizeof(float));
        biasBuffer->unmap();
        extra->copyBufferToImage(biasBuffer.get(), mBias.get());
        bias = mBias.get();
    }
    /*Write Command Buffer*/
    mConvSet.reset(mConvPipeline->createSet());
    mConvSet->writeImage(((VulkanTensor*)output->deviceId())->image()->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mConvSet->writeImage(((VulkanTensor*)input->deviceId())->image()->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         1);
    mConvSet->writeImage(mKernel->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);
    mConvSet->writeImage(bias->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3);
    mConvSet->writeBuffer(convCons->buffer(), 4, convCons->size());
    mConvPipeline->bind(cmdBuffer->get(), mConvSet->get());
    mKernel->barrierRead(cmdBuffer->get());
    mBias->barrierRead(cmdBuffer->get());
    ((VulkanTensor*)input->deviceId())->image()->barrierRead(cmdBuffer->get());
    ((VulkanTensor*)output->deviceId())->image()->barrierWrite(cmdBuffer->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(ow, mLocalX), UP_DIV(oh, mLocalY), ocDiv4 * input->batch());
    return NO_ERROR;
}

class VulkanConvolutionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto extra          = static_cast<VulkanBackend *>(backend);
        auto convReal       = op->main_as_Convolution2D();
        auto common         = convReal->common();
        auto outputCount    = common->outputCount();
        const int fh        = common->kernelY();
        const int fw        = common->kernelX();
        int srcCount        = 0;
        const float* source = nullptr;
        const float* biasPtr = nullptr;
        int weightSize = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
            quanWeight = ConvolutionCommon::load(op->main_as_Convolution2D(), backend, true);
            srcCount = quanWeight->weightFloat.size() / (outputCount * fh * fw);
            source   = quanWeight->weightFloat.get();
            weightSize = quanWeight->weightFloat.size();
        } else {
            if (nullptr != convReal->weight()) {
                srcCount = convReal->weight()->size() / (outputCount * fh * fw);
                source   = convReal->weight()->data();
                weightSize = convReal->weight()->size();
            } else {
                srcCount = convReal->common()->inputCount();
            }
        }
        if (nullptr != convReal->bias()) {
            biasPtr = convReal->bias()->data();
        }
        if (op->type() == OpType_Convolution) {
            if (inputs.size() > 1) {
                return nullptr;
            }
            auto convCommonParam = op->main_as_Convolution2D()->common();
            const int group      = convCommonParam->group();
            if (1 == group) {
                return VulkanConvolutionImpl::create(extra, common, inputs, outputs[0], source,
                                                     biasPtr, srcCount, outputCount);

            } else {
                return nullptr;
            }
        }
        return new VulkanConvolutionDepthwise(source, weightSize, op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Convolution, new VulkanConvolutionCreator);
    VulkanBackend::addCreator(OpType_ConvolutionDepthwise, new VulkanConvolutionCreator);
    return true;
}();

} // namespace MNN
