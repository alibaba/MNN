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

void VulkanConvolutionCommon::writeDeconvolution(VulkanConvolutionCommon::ConvolutionParameter* convCons,
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

VulkanConvolutionCommon::VulkanConvolutionCommon(const Convolution2DCommon* common, Backend* bn) : VulkanBasicExecution(bn) {
    auto extra    = static_cast<VulkanBackend*>(bn);
    mCommon       = common;
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
bool VulkanConvolutionDepthwise::_init(const float* weightData, size_t weightSize, const Op* convOp, Backend* bn, bool initweights) {
    auto extra      = static_cast<VulkanBackend*>(bn);
    auto common    = convOp->main_as_Convolution2D()->common();
    // Create Pipeline
    std::vector<VkDescriptorType> convTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    MNN_ASSERT(OpType_ConvolutionDepthwise == convOp->type());
    auto macro = getPostTreatMacro(common);
    mConvPipeline = extra->getPipeline("glsl_convolutionDepthwise_" + macro + "comp", convTypes);
    mLocalX       = 16;
    mLocalY       = 16;

    mConvSet.reset(mConvPipeline->createSet());
    if (!initweights) {
        return true;
    }
    auto bytes = sizeof(float);
    auto c4 = UP_DIV(common->outputCount(), 4);
    if (nullptr != weightData){
        mKernel = _createBufferForConvDepthwise(extra, common, weightData, weightSize);
    } else {
        mKernel.reset(new VulkanBuffer(extra->getMemoryPool(), false, common->kernelX() * common->kernelY() * c4 * 4 * sizeof(float), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto weight     = (float*)mKernel->map();
        ::memset(weight, 0, mKernel->size());
        mKernel->unmap();
    }
    auto convReal = convOp->main_as_Convolution2D();
    auto biasBuffer = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false,
                                                     sizeof(float) * ALIGN_UP4(common->outputCount()));

    auto bias = biasBuffer->map();
    ::memset(bias, 0, ALIGN_UP4(common->outputCount()) * sizeof(float));
    if (nullptr != convReal->bias()) {
        // Create Buffer
        ::memcpy(bias, convReal->bias()->data(), common->outputCount() * sizeof(float));
    }
    biasBuffer->unmap();
    mBias = biasBuffer;
    return true;
}

bool VulkanConvolutionDepthwise::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanConvolutionDepthwise(op, bn);
    res->mBias = mBias;
    res->mKernel = mKernel;
    *dst = res;
    return true;
}

VulkanConvolutionDepthwise::VulkanConvolutionDepthwise(const float* weightData, size_t weightSize, const Op* convOp, Backend* bn)
    : VulkanConvolutionCommon(convOp->main_as_Convolution2D()->common(), bn) {
    _init(weightData, weightSize, convOp, bn, true);
}
VulkanConvolutionDepthwise::VulkanConvolutionDepthwise(const Op* op, Backend* bn) : VulkanConvolutionCommon(op->main_as_Convolution2D()->common(), bn) {
    _init(nullptr, 0, op, bn, false);
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
    if (inputs.size() >= 2) {
        auto weight =  extra->getTensorBuffer(inputs[1]);
        auto weightSize = extra->getTensorSize(inputs[1]);
        auto pipeline = extra->getPipeline("glsl_dwweightcopy_comp", {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        });
        std::shared_ptr<VulkanLayout::DescriptorSet> des(pipeline->createSet());
        des->writeBuffer(weight.first->buffer(), 1, weightSize, weight.second);
        des->writeBuffer(mKernel->buffer(), 0, mKernel->size());
        int dim[4] = {
            common->kernelX(),
            common->kernelY(),
            output->channel(),
            output->channel() * common->kernelX() * common->kernelY()
        };
        std::shared_ptr<VulkanBuffer> uniforms(new VulkanBuffer(extra->getMemoryPool(), false, sizeof(dim), &dim, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        des->writeBuffer(uniforms->buffer(), 2, uniforms->size());
        pipeline->bind(cmdBuffer->get(), des->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(dim[3], 256), 1, 1);
        mExtraBuffers = uniforms;
        mExtraSets = des;
        cmdBuffer->barrierSource(mKernel->buffer(), 0, mKernel->size());
    }
    std::pair<const VulkanBuffer*, size_t> bias;
    size_t biasSize;
    if (inputs.size() >= 3) {
        bias = extra->getTensorBuffer(inputs[2]);
        biasSize = extra->getTensorSize(inputs[2]);
    } else {
        bias.first = mBias.get();
        bias.second = 0;
        biasSize = mBias->size();
    }
    /*Write Command Buffer*/
    auto outputBuffer = extra->getBuffer(outputs[0]);
    auto inputBuffer = extra->getBuffer(input);
    mConvSet->writeBuffer(outputBuffer, 0);
    mConvSet->writeBuffer(inputBuffer, 1);
    mConvSet->writeBuffer(mKernel->buffer(), 2, mKernel->size());
    mConvSet->writeBuffer(bias.first->buffer(), 3, biasSize, bias.second);
    mConvSet->writeBuffer(convCons->buffer(), 4, convCons->size());
    mConvPipeline->bind(cmdBuffer->get(), mConvSet->get());
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
