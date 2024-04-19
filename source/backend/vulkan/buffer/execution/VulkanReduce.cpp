//
//  VulkanReduce.cpp
//  MNN
//
//  Created by MNN on 2020/03/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanReduce.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"
namespace MNN {
struct constBuffer {
    int w;//inside
    int h;//axis
    int c;//outside
    float k;//For mean
    int reduceAxis;
};
#define MAX_VALUE 10001.f
VulkanReduce::VulkanReduce(const std::string& name, const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend*)backend();
    mOp = op;
    mPipeline = vkBn->getPipeline(name, {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
    mConstBuffer = vkBn->allocUniform();
    mDescriptorSet.reset(mPipeline->createSet());
}
VulkanReduce::~VulkanReduce() {
    auto vkBn = (VulkanBackend*)backend();
    vkBn->recycleUniform(mConstBuffer);
}

ErrorCode VulkanReduce::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    auto ptr = reinterpret_cast<constBuffer*>(mConstBuffer->map());
    ::memset(ptr, 0, sizeof(constBuffer));
    auto axisPos = mOp->main_as_ReductionParam()->dim()->data()[0];
    auto axis = inputs[0]->length(axisPos);
    int inside = 1;
    for (int i=axisPos+1; i<inputs[0]->dimensions(); ++i) {
        inside *= inputs[0]->length(i);
    }
    int outside = 1;
    for (int i=0; i<axisPos; ++i) {
        outside *= inputs[0]->length(i);
    }
    ptr->c = outside;
    ptr->h = axis;
    ptr->w = inside;
    ptr->k = 1.0f/(float)axis;
    auto total = outside * inside;
    int outsideParallel = 1;
    if (total >= 256) {
        ptr->reduceAxis = 1;
        outsideParallel = 256;
    } else if (total < 16) {
        ptr->reduceAxis = 256;
        outsideParallel = 1;
    } else {
        ptr->reduceAxis = 16;
        outsideParallel = 16;
    }
    //MNN_PRINT("o, i, axis: %d - %d - %d => op %d, ra %d\n", outside, inside, axis, outsideParallel, ptr->reduceAxis);
    mConstBuffer->unmap();
    // Encode
    mDescriptorSet->writeBuffer(outputTensor, 0);
    mDescriptorSet->writeBuffer(inputTensor, 1);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
    cmdBuffer->barrierSource(inputTensor);
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, outsideParallel), 1, 1);
    return NO_ERROR;
}

static std::string _getShaderName(const Op* op, bool isInt) {
    std::string prefix = "glsl_reduce_";
    if (isInt) {
        prefix = "glsl_reduce_int_";
    }
    std::string posfix = "_comp";
    std::string mid = "";
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_SUM:
            mid = "SUM";
            break;
        case ReductionType_MEAN:
            mid = "MEAN";
            break;
        case ReductionType_MAXIMUM:
            mid = "VMAX";
            break;
        case ReductionType_MINIMUM:
            mid = "VMIN";
            break;
        case ReductionType_PROD:
            mid = "PROD";
            break;
        default:
            break;
    }
    if (mid.empty()) {
        return mid;
    }
    return prefix + mid + posfix;
}
class VulkanReduceCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input0 = inputs[0];
        bool isint = input0->getType().code == halide_type_int;
        auto shader = _getShaderName(op, isint);
        if (shader.empty()) {
            return nullptr;
        }
        return new VulkanReduce(shader, op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Reduction, new VulkanReduceCreator);
    return true;
}();
}
