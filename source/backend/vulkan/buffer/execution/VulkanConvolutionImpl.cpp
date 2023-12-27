//
//  VulkanConvolutionImpl.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanConvolutionImpl.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "VulkanConvolution.hpp"
#include "VulkanRaster.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {

class VulkanConvolutionSlideWindows : public VulkanConvolutionCommon {
private:
    const VulkanPipeline* mSlideWindow;

    std::shared_ptr<VulkanBuffer> mBias;
    std::shared_ptr<VulkanLayout::DescriptorSet> mConvSet;
    const Convolution2DCommon* mConvCommonOption;
    VulkanRaster::Componet mKernelReorder;
    std::shared_ptr<Tensor> mKernel;
    std::pair<int, int> mChannels;
public:

    VulkanConvolutionSlideWindows(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr,
                            const float* biasPtr, int ci, int co) : VulkanConvolutionCommon(convOption, backend) {
        auto kw = convOption->kernelX();
        auto kh = convOption->kernelY();
        auto vkBn = (VulkanBackend*)backend;
        mChannels = std::make_pair(ci, co);
        // Create Pipeline
        std::vector<VkDescriptorType> convTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        auto macro = getPostTreatMacro(convOption);
        mSlideWindow = vkBn->getPipeline("glsl_convolution_" + macro + "comp", convTypes);
        mConvSet.reset(mSlideWindow->createSet());
        auto common = convOption;
        auto extra = vkBn;
        mBias = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * ALIGN_UP4(common->outputCount()));
        auto bias = mBias->map();
        ::memset(bias, 0, ALIGN_UP4(common->outputCount()) * sizeof(float));
        if (nullptr != biasPtr) {
            ::memcpy(bias, biasPtr, common->outputCount() * sizeof(float));
        }
        mBias->unmap();
        int ciC4 = UP_DIV(ci, 4);
        int coC4 = UP_DIV(co, 4);
        int kernelSize = common->kernelY() * common->kernelX();
        mKernel.reset(Tensor::createDevice<float>({coC4, kernelSize, ciC4, (4 * 4)}));
        mKernelReorder = VulkanRaster::create(mKernel.get(), vkBn);
        auto des = TensorUtils::getDescribe(mKernel.get());
        int pack = 4;
        for (int i=0; i<pack; ++i) {
            auto cSize = (ci + pack - 1 - i) / pack;
            if (cSize <= 0) {
                continue;
            }
            auto srcCIOffset = i * kernelSize;
            auto dstCIOffset = i * pack;
            for (int j=0; j<pack; ++j) {
                auto oSize = (co + pack - 1 - j) / pack;
                if (oSize <= 0) {
                    continue;
                }
                auto srcCOOffset = srcCIOffset + j * kernelSize * ci;
                auto dstCOOffset = dstCIOffset + j;
                Tensor::InsideDescribe::Region reg;
                reg.size[0] = oSize;
                reg.size[1] = cSize;
                reg.size[2] = kernelSize;
                reg.dst.offset = dstCOOffset;
                reg.dst.stride[0] = pack * pack * ciC4 * kernelSize;
                reg.dst.stride[1] = pack * pack;
                reg.dst.stride[2] = pack * pack * ciC4;

                reg.src.offset = srcCOOffset;
                reg.src.stride[0] = pack * ci * kernelSize;
                reg.src.stride[1] = pack * kernelSize;
                reg.src.stride[2] = 1;
                des->regions.emplace_back(std::move(reg));
            }
        }
        if (nullptr != weightPtr) {
            auto res = vkBn->onAcquireBuffer(mKernel.get(), Backend::STATIC);
            if (!res) {
                return;
            }
            std::shared_ptr<Tensor> sourceWeight(Tensor::createDevice<float>({ci * co * kernelSize}));
            res = vkBn->onAcquireBuffer(sourceWeight.get(), Backend::STATIC);
            if (!res) {
                return;
            }
            {
                auto vkTensor = extra->getBuffer(sourceWeight.get());
                extra->copyToGPUBuffer(weightPtr, std::get<0>(vkTensor), sourceWeight->size(), std::get<2>(vkTensor));
            }
            std::shared_ptr<VulkanCommandPool::Buffer> prearrangeCmd( vkBn->getPool().allocBuffer());
            for (auto& reg : des->regions) {
                reg.origin = sourceWeight.get();
            }
            prearrangeCmd->begin(0);
            mKernelReorder.exe->onEncode({}, {mKernel.get()}, prearrangeCmd.get());
            prearrangeCmd->end();
            vkBn->pushCommand(prearrangeCmd->get());
            vkBn->onExecuteEnd();
            mKernelReorder.exe = nullptr;
        }
    }
    ~VulkanConvolutionSlideWindows() {
        // Do nothing
    }
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto res = new VulkanConvolutionSlideWindows((VulkanBackend*)bn, op->main_as_Convolution2D()->common(), nullptr, nullptr, mChannels.first, mChannels.second);
        res->mBias = mBias;
        res->mKernel = mKernel;
        *dst = res;
        return true;
    }
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer) override {
        auto src         = inputs[0];
        auto dst         = outputs[0];
        const int icDiv4 = UP_DIV(src->channel(), 4);
        const int ocDiv4 = UP_DIV(dst->channel(), 4);
        auto vkBn = (VulkanBackend*)backend();
        auto extra = static_cast<VulkanBackend*>(backend());
        if (inputs.size() >= 2) {
            auto res = vkBn->onAcquireBuffer(mKernel.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            auto des = TensorUtils::getDescribe(mKernel.get());
            for (auto& reg : des->regions) {
                reg.origin = inputs[1];
            }
            auto rasterCode = mKernelReorder.exe->onEncode({}, {mKernel.get()}, cmdBuffer);
            if (NO_ERROR != rasterCode) {
                return rasterCode;
            }
            auto kernelBuffer = extra->getTensorBuffer(mKernel.get());
            auto kernelSize = extra->getTensorSize(mKernel.get());
            cmdBuffer->barrierSource(kernelBuffer.first->buffer(), kernelBuffer.second, kernelSize);
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
        auto outputBuffer = extra->getTensorBuffer(outputs[0]);
        auto inputBuffer = extra->getTensorBuffer(inputs[0]);
        mConvSet->writeBuffer(outputBuffer.first->buffer(), 0, extra->getTensorSize(outputs[0]), outputBuffer.second);
        mConvSet->writeBuffer(inputBuffer.first->buffer(), 1, extra->getTensorSize(inputs[0]), inputBuffer.second);
        auto kernelBuffer = extra->getTensorBuffer(mKernel.get());
        mConvSet->writeBuffer(kernelBuffer.first->buffer(), 2, extra->getTensorSize(mKernel.get()), kernelBuffer.second);
        mConvSet->writeBuffer(bias.first->buffer(), 3, biasSize, bias.second);
        mConvSet->writeBuffer(constConvBuffer->buffer(), 4, constConvBuffer->size());
        mSlideWindow->bind(cmdBuffer->get(), mConvSet->get());
        int totalSize = ocDiv4 * outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 64), 1, 1);
        if (inputs.size() >= 2) {
            vkBn->onReleaseBuffer(mKernel.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }
};

VulkanBasicExecution* VulkanConvolutionImpl::create(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                                         const std::vector<Tensor*>& inputs, const Tensor* output,
                                                         const float* weightPtr, const float* biasPtr, int ci, int co) {
    AUTOTIME;
    return new VulkanConvolutionSlideWindows(backend, convOption, weightPtr, biasPtr, ci, co);
}

} // namespace MNN
