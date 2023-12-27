//
//  VulkanDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanDeconvolution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {
static void _initKernelRegion() {
    
}
VulkanDeconvolution::VulkanDeconvolution(Backend* bn) : VulkanBasicExecution(bn) {
    // Donthing
}

VulkanDeconvolution* VulkanDeconvolution::create(Backend* bn, const Convolution2D* conv, OpType type, bool multiInputs) {
    auto exeRes = new VulkanDeconvolution(bn);
    exeRes->mConvCommonOption = conv->common();
    auto vkBn         = (VulkanBackend*)bn;
    int outputC4      = UP_DIV(exeRes->mConvCommonOption->outputCount(), 4);
    auto biasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, outputC4 * 4 * sizeof(float));
    auto biasPtr    = biasBuffer->map();
    ::memset(biasPtr, 0, outputC4 * 4 * sizeof(float));
    if (conv->bias() != nullptr) {
        ::memcpy(biasPtr, conv->bias()->data(), conv->bias()->size() * sizeof(float));
    }
    biasBuffer->unmap();
    exeRes->mBias = biasBuffer;
    exeRes->mConvParam = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false,
                                                sizeof(VulkanConvolutionCommon::ConvolutionParameter), nullptr,
                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    int kh     = exeRes->mConvCommonOption->kernelY();
    int kw     = exeRes->mConvCommonOption->kernelX();
    int co     = exeRes->mConvCommonOption->outputCount();
    int coC4   = UP_DIV(co, 4);
    int ci     = exeRes->mConvCommonOption->inputCount();
    if (type == OpType_DeconvolutionDepthwise) {
        ci = 1;
    }
    const float* tempWeight = nullptr;
    int tempWeightSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (!multiInputs) {
        ConvolutionCommon::getConvParameters(&quanCommon, bn, conv, &tempWeight, &tempWeightSize);
        MNN_ASSERT(nullptr != tempWeight);
        if (0 >= ci) {
            ci = tempWeightSize / co / kw / kh;
        }
    }

    int ciC4   = UP_DIV(ci, 4);
    if (type == OpType_Deconvolution) {
        exeRes->mKernel.reset(MNN::Tensor::createDevice<float>({kw*kh, coC4, ciC4, 16}));
    } else {
        exeRes->mKernel.reset(MNN::Tensor::createDevice<float>({kw*kh, coC4, 4}));
    }
    exeRes->mKernelReorder = VulkanRaster::create(exeRes->mKernel.get(), vkBn);
    auto des = TensorUtils::getDescribe(exeRes->mKernel.get());
    int pack = 4;
    if (OpType_DeconvolutionDepthwise == type) {
        for (int i=0; i<pack; ++i) {
            auto oSize = (co + pack - 1 - i) / pack;
            if (oSize <= 0) {
                continue;
            }
            Tensor::InsideDescribe::Region reg;
            reg.size[0] = 1;
            reg.size[1] = oSize;
            reg.size[2] = kh * kw;
            reg.dst.offset = i;
            reg.dst.stride[0] = 0;
            reg.dst.stride[1] = pack * kh * kw;
            reg.dst.stride[2] = pack;

            reg.src.offset = kh * kw * i;
            reg.src.stride[0] = 0;
            reg.src.stride[1] = pack * kh * kw;
            reg.src.stride[2] = 1;
            des->regions.emplace_back(std::move(reg));
        }
    } else {
        for (int i=0; i<pack; ++i) {
            auto oSize = (co + pack - 1 - i) / pack;
            if (oSize <= 0) {
                continue;
            }
            for (int j=0; j<pack; ++j) {
                int cSize = (ci + pack - 1 - j) / pack;
                if (cSize <= 0) {
                    continue;
                }
                Tensor::InsideDescribe::Region reg;
                reg.size[0] = oSize;
                reg.size[1] = cSize;
                reg.size[2] = kh * kw;
                reg.dst.offset = i + j * pack;
                reg.dst.stride[0] = pack * pack * ciC4 * kh * kw;
                reg.dst.stride[1] = pack * pack;
                reg.dst.stride[2] = pack * pack * ciC4;

                reg.src.offset = kh * kw * i + kh * kw * co * j;
                reg.src.stride[0] = pack * kh * kw;
                reg.src.stride[1] = pack * kh * kw * co;
                reg.src.stride[2] = 1;
                des->regions.emplace_back(std::move(reg));
            }
        }
    }

    if (!multiInputs) {
        MNN_ASSERT(nullptr != tempWeight);
        auto res = vkBn->onAcquireBuffer(exeRes->mKernel.get(), Backend::STATIC);
        if (!res) {
            return nullptr;
        }
        std::shared_ptr<Tensor> tempWeightTensor(Tensor::createDevice<float>({tempWeightSize}));
        res = vkBn->onAcquireBuffer(tempWeightTensor.get(), Backend::STATIC);
        if (!res) {
            return nullptr;
        }
        auto tempWeightBuffer = reinterpret_cast<VulkanBuffer*>(tempWeightTensor->deviceId());
        vkBn->copyToGPUBuffer(tempWeight, tempWeightBuffer->buffer(), tempWeightSize * sizeof(float), TensorUtils::getDescribe(tempWeightTensor.get())->extra.offset);
        std::shared_ptr<VulkanCommandPool::Buffer> prearrangeCmd( vkBn->getPool().allocBuffer());
        for (auto& reg : des->regions) {
            reg.origin = tempWeightTensor.get();
        }
        prearrangeCmd->begin(0);
        exeRes->mKernelReorder.exe->onEncode({}, {exeRes->mKernel.get()}, prearrangeCmd.get());
        prearrangeCmd->end();
        vkBn->pushCommand(prearrangeCmd->get());
        vkBn->onExecuteEnd();
        exeRes->mKernelReorder.exe = nullptr;
    }
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    std::string macro = VulkanConvolutionCommon::getPostTreatMacro(exeRes->mConvCommonOption);

    if (type == OpType_Deconvolution) {
        exeRes->mPipeline = vkBn->getPipeline("glsl_deconvolution_" + macro + "comp", types);
    } else {
        MNN_ASSERT(type == OpType_DeconvolutionDepthwise);
        exeRes->mPipeline = vkBn->getPipeline("glsl_deconvolutionDepthwise_" + macro + "comp", types);
    }
    exeRes->mPipelineSet.reset(exeRes->mPipeline->createSet());
    return exeRes;
}

ErrorCode VulkanDeconvolution::onEncode(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs,
                                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto src         = inputs[0];
    auto dst         = outputs[0];
    const int ocDiv4 = UP_DIV(dst->channel(), 4);
    auto common      = mConvCommonOption;
    auto extra = static_cast<VulkanBackend*>(backend());
    if (inputs.size() >= 2) {
        auto res = extra->onAcquireBuffer(mKernel.get(), Backend::DYNAMIC);
        if (!res) {
            return NO_ERROR;
        }
        auto kernelBuffer = extra->getBuffer(mKernel.get());
        auto des = TensorUtils::getDescribe(mKernel.get());
        for (auto& reg : des->regions) {
            reg.origin = inputs[1];
        }
        auto rasterCode = mKernelReorder.exe->onEncode({}, {mKernel.get()}, cmdBuffer);
        if (NO_ERROR != rasterCode) {
            return rasterCode;
        }
        cmdBuffer->barrierSource(kernelBuffer);
    }
    {
        auto convCons = reinterpret_cast<VulkanConvolutionCommon::ConvolutionParameter*>(mConvParam->map());
        VulkanConvolutionCommon::writeDeconvolution(convCons, common, src, dst);
        mConvParam->unmap();
    }
    auto dstBuffer = extra->getBuffer(dst);
    auto srcBuffer = extra->getBuffer(src);
    auto kernelBuffer = extra->getBuffer(mKernel.get());

    mPipelineSet->writeBuffer(dstBuffer, 0);
    mPipelineSet->writeBuffer(srcBuffer, 1);
    mPipelineSet->writeBuffer(kernelBuffer, 2);
    if (inputs.size() >= 3) {
        auto biasBuffer = extra->getBuffer(inputs[2]);
        mPipelineSet->writeBuffer(biasBuffer, 3);
    } else {
        mPipelineSet->writeBuffer(mBias->buffer(), 3, mBias->size());
    }
    mPipelineSet->writeBuffer(mConvParam->buffer(), 4, mConvParam->size());
    mPipeline->bind(cmdBuffer->get(), mPipelineSet->get());
    auto totalCount = dst->width() * dst->height() * ocDiv4 * dst->batch();

    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalCount, 64), 1, 1);
    if (inputs.size() >= 2) {
        extra->onReleaseBuffer(mKernel.get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

class VulkanDeconvolutionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return VulkanDeconvolution::create(backend, op->main_as_Convolution2D(), op->type(), inputs.size() > 1);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_DeconvolutionDepthwise, new VulkanDeconvolutionCreator);
    VulkanBackend::addCreator(OpType_Deconvolution, new VulkanDeconvolutionCreator);
    return true;
}();
} // namespace MNN
