//
//  VulkanUnary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanUnary.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

struct Param {
    int len;
};

VulkanUnary::VulkanUnary(const Op* op, Backend* bn) : VulkanBasicExecution(bn), mOp(op) {
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam         = std::make_shared<VulkanBuffer>(vkbackend->getMemoryPool(), false, sizeof(Param), nullptr,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanUnary::~VulkanUnary() {
}

ErrorCode VulkanUnary::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(MNN_DATA_FORMAT_NHWC == TensorUtils::getDescribe(inputs[0])->dimensionFormat);
    MNN_ASSERT(inputs[0]->buffer().type.code == halide_type_float && inputs[0]->buffer().type.bits == 32);
    // get pipeline
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    if (mOp->type() == OpType_TanH) {
        mUnaryPipeline = vkbackend->getPipeline("glsl_tanh_comp", /*glsl_tanh_comp, glsl_tanh_comp_len,*/ types);
    } else {
        // unary op
        auto unaryType = mOp->main_as_UnaryOp()->opType();
        switch (unaryType) {
            case UnaryOpOperation_RSQRT:
                mUnaryPipeline =
                    vkbackend->getPipeline("glsl_rsqrt_comp", /*glsl_rsqrt_comp, glsl_rsqrt_comp_len,*/ types);
                break;
            case UnaryOpOperation_ABS:
                mUnaryPipeline = vkbackend->getPipeline("glsl_abs_comp", /*glsl_abs_comp, glsl_abs_comp_len,*/ types);
                break;
            case UnaryOpOperation_EXP:
                mUnaryPipeline = vkbackend->getPipeline("glsl_exp_comp", /*glsl_exp_comp, glsl_exp_comp_len,*/ types);
                break;
            case UnaryOpOperation_SQRT:
                mUnaryPipeline =
                    vkbackend->getPipeline("glsl_sqrt_comp", /*glsl_sqrt_comp, glsl_sqrt_comp_len,*/ types);
                break;
            default:
                break;
        }
    }

    // set param
    auto paramPtr = reinterpret_cast<Param*>(mParam->map());
    paramPtr->len = inputs[0]->elementSize();
    mParam->unmap();

    mDesSet.reset(mUnaryPipeline->createSet());
    mDesSet->writeBuffer(reinterpret_cast<VkBuffer>(outputs[0]->deviceId()), 0, outputs[0]->size());
    mDesSet->writeBuffer(reinterpret_cast<VkBuffer>(inputs[0]->deviceId()), 1, inputs[0]->size());
    mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
    mUnaryPipeline->bind(cmdBuffer->get(), mDesSet->get());
    cmdBuffer->barrierSource(reinterpret_cast<VkBuffer>(inputs[0]->deviceId()), 0, inputs[0]->size());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(inputs[0]->elementSize(), 16), 1, 1);

    return NO_ERROR;
}

class VulkanUnaryCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanUnary(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_UnaryOp, new VulkanUnaryCreator);
    VulkanBackend::addCreator(OpType_TanH, new VulkanUnaryCreator);
    return true;
}();

} // namespace MNN
