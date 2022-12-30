//
//  VulkanMatMul.cpp
//  MNN
//
//  Created by MNN on 2020/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMatMul.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {
VulkanMatMul::VulkanMatMul(bool transposeA, bool transposeB, Backend* bn, bool hasBias) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend*)bn;
    mTransposeA = transposeA;
    mBlitPipeline = vkBn->getPipeline("glsl_blit_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
    mTransposeB = transposeB;
    if (hasBias) {
        mComputePipeline = vkBn->getPipeline("glsl_gemm16x16_bias_comp", {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        });
    } else {
        mComputePipeline = vkBn->getPipeline("glsl_gemm16x16_comp", {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        });
    }
    mComputeSet.reset(mComputePipeline->createSet());
}
ErrorCode VulkanMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    mTempBuffer.clear();
    Tensor* C       = outputs[0];
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto input0T = vkBn->getBuffer(inputs[0]);
    auto input1T = vkBn->getBuffer(inputs[1]);
    auto outputT = vkBn->getBuffer(outputs[0]);
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    int ePack = 8;
    int hPack = 4;
    auto eUnit = UP_DIV(e, ePack);
    auto hUnit = UP_DIV(h, hPack);
    std::shared_ptr<Tensor> inputTemp(Tensor::createDevice<float>({eUnit * l * ePack}));
    std::shared_ptr<Tensor> kernelTemp(Tensor::createDevice<float>({hUnit * l * hPack}));
    std::shared_ptr<Tensor> outputTemp(Tensor::createDevice<float>({eUnit * h * ePack}));
    mInput = VulkanRaster::create(inputTemp.get(), backend());
    {
        // Input: e, l -> e/ePack, l, ePack
        auto des = TensorUtils::getDescribe(mInput.real);
        des->regions.clear();
        des->regions.reserve(ePack);
        for (int i=0; i<ePack; ++i) {
            auto sizeTemp = (e + ePack - 1 - i) / ePack;
            if (sizeTemp <= 0) {
                continue;
            }
            Tensor::InsideDescribe::Region reg;
            reg.origin = inputs[0];
            reg.size[0] = 1;
            reg.size[1] = l;
            reg.size[2] = sizeTemp;
            reg.dst.offset = i;
            reg.dst.stride[0] = 0;
            reg.dst.stride[1] = ePack;
            reg.dst.stride[2] = l * ePack;

            reg.src.stride[0] = 0;
            if (mTransposeA) {
                reg.src.offset = i;
                reg.src.stride[1] = e;
                reg.src.stride[2] = ePack;
            } else {
                reg.src.offset = i * l;
                reg.src.stride[1] = 1;
                reg.src.stride[2] = ePack * l;
            }
            des->regions.emplace_back(std::move(reg));
        }
        //FUNC_PRINT(des->regions.size());
    }
    mKernel = VulkanRaster::create(kernelTemp.get(), backend());
    {
        // Kernel: l, h -> h/hPack, l, hPack
        auto des = TensorUtils::getDescribe(mKernel.real);
        des->regions.clear();
        des->regions.reserve(hPack);
        for (int i=0; i<hPack; ++i) {
            Tensor::InsideDescribe::Region reg;
            reg.size[2] = (h + hPack - 1 - i) / hPack;
            if (reg.size[2] <= 0) {
                continue;
            }
            reg.origin = inputs[1];
            reg.size[0] = 1;
            reg.size[1] = l;
            reg.dst.offset = i;
            reg.dst.stride[0] = 0;
            reg.dst.stride[1] = hPack;
            reg.dst.stride[2] = l * hPack;

            reg.src.stride[0] = 0;
            if (!mTransposeB) {
                reg.src.offset = i;
                reg.src.stride[1] = h;
                reg.src.stride[2] = hPack;
            } else {
                reg.src.offset = i * l;
                reg.src.stride[1] = 1;
                reg.src.stride[2] = hPack * l;
            }
            des->regions.emplace_back(std::move(reg));
        }
        //FUNC_PRINT(des->regions.size());
    }
    mOutput = VulkanRaster::create(outputs[0], backend());
    {
        // Output: e/Pack, h, ePack -> e, h
        auto des = TensorUtils::getDescribe(mOutput.real);
        des->regions.clear();
        des->regions.reserve(ePack);
        for (int i=0; i<ePack; ++i) {
            auto sizeTemp = (e + ePack - 1 - i) / ePack;
            if (sizeTemp <= 0) {
                continue;
            }
            Tensor::InsideDescribe::Region reg;
            reg.origin = outputTemp.get();
            reg.size[0] = 1;
            reg.size[1] = h;
            reg.size[2] = sizeTemp;
            reg.src.offset = i;
            reg.src.stride[0] = 0;
            reg.src.stride[1] = ePack;
            reg.src.stride[2] = h * ePack;

            reg.dst.offset = i * h;
            reg.dst.stride[0] = 0;
            reg.dst.stride[1] = 1;
            reg.dst.stride[2] = ePack * h;
            des->regions.emplace_back(std::move(reg));
        }
    }

    {
        // Pretreat B
        auto res = backend()->onAcquireBuffer(kernelTemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        auto code = mKernel.exe->onEncode({}, {mKernel.real}, cmdBuffer);
        if (NO_ERROR != code) {
            return code;
        }
    }
    {
        // Pretreat A
        auto res = backend()->onAcquireBuffer(inputTemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        auto code = mInput.exe->onEncode({}, {mInput.real}, cmdBuffer);
        if (NO_ERROR != code) {
            return code;
        }
    }
    {
        auto res = backend()->onAcquireBuffer(outputTemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        ivec4 gemmSize;
        gemmSize[0] = eUnit;
        gemmSize[1] = hUnit;
        gemmSize[2] = h;
        gemmSize[3] = l;
        std::shared_ptr<VulkanBuffer> gemmParameter(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(gemmSize), &gemmSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        mTempBuffer.emplace_back(gemmParameter);
        mComputeSet->writeBuffer(vkBn->getBuffer(mInput.real), 1);
        mComputeSet->writeBuffer(vkBn->getBuffer(mKernel.real), 2);
        mComputeSet->writeBuffer(vkBn->getBuffer(outputTemp.get()), 0);
        if (inputs.size() >= 3) {
            mComputeSet->writeBuffer(vkBn->getBuffer(inputs[2]), 4);
        }
        mComputeSet->writeBuffer(gemmParameter->buffer(), 3, gemmParameter->size());
        mComputePipeline->bind(cmdBuffer->get(), mComputeSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(eUnit, 2), UP_DIV(hUnit, 4), 1);
    }
    backend()->onReleaseBuffer(kernelTemp.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(inputTemp.get(), Backend::DYNAMIC);
    {
        // Posttreat C
        auto code = mOutput.exe->onEncode({}, {outputs[0]}, cmdBuffer);
        if (NO_ERROR != code) {
            return code;
        }
    }
    backend()->onReleaseBuffer(outputTemp.get(), Backend::DYNAMIC);
    return NO_ERROR;
}
class VulkanMatMulCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        const auto mat = op->main_as_MatMul();
        return new VulkanMatMul(mat->transposeA(), mat->transposeB(), bn, inputs.size() > 2);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MatMul, new VulkanMatMulCreator);
    return true;
}();

}
