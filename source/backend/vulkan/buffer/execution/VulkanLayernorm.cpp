//
//  VulkanLayernorm.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanLayernorm.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct Param {
    ivec4 size;
    vec4 eps;
};

struct FP16Param {
    ivec4 size;
    f16vec4 eps;
};

VulkanLayernorm::VulkanLayernorm(const Op* op, Backend* backend, Tensor * tensor) : VulkanBasicExecution(backend) {
    auto layer_norm_param = op->main_as_LayerNorm();
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    if (nullptr != layer_norm_param->axis()) {
        mAxisSize = layer_norm_param->axis()->size();
    }
    mGroup = layer_norm_param->group();

    mParam = vkbackend->allocUniform();
    mEps = layer_norm_param->epsilon();

    mFP16 = tensor->getType().code == halide_type_float && vkbackend->useFP16();

    if (layer_norm_param->gamma() && layer_norm_param->beta()) {
        mHasScale = true;
        int size = layer_norm_param->gamma()->size();
        auto prepareParam = [&](std::shared_ptr<Tensor>& paramTensor, const float* sourceData, const char* errorName) {
            paramTensor.reset(Tensor::createDevice<float>({size}));
            auto status = backend->onAcquireBuffer(paramTensor.get(), Backend::STATIC);
            if (!status) {
                MNN_ERROR("Out of memory when %s is acquired in LayerNorm.\n", errorName);
                return false;
            }
            const void * paramData;
            std::vector<int16_t> paramFP16;
            if (mFP16) {
                paramFP16.resize(size);
                FLOAT_TO_HALF(sourceData, paramFP16.data(), size);
                paramData = paramFP16.data();
            } else {
                paramData = (const void *) sourceData;
            }
            auto paramBuffer = vkbackend->getBuffer(paramTensor.get());
            vkbackend->copyToGPUBuffer(paramData, std::get<0>(paramBuffer), std::get<1>(paramBuffer), std::get<2>(paramBuffer));
            return true;
        };

        if (!prepareParam(mGamma, layer_norm_param->gamma()->data(), "gamma")) {
            return;
        }

        if (layer_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in LayerNorm.\n");
            return;
        }

        if (!prepareParam(mBias, layer_norm_param->beta()->data(), "beta")) {
            return;
        }
    }

    std::string pKey = "glsl_norm_";
    std::vector<VkDescriptorType> types;
    if (!mHasScale) {
        types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
    } else {
        types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        pKey += "LAYERNORM_SCALE_";
    }
    if (mFP16) {
        pKey += "FP16_";
    }
    pKey += "comp";
    mPipeline = vkbackend->getPipeline(pKey, types);
    mDesSet.reset(mPipeline->createSet());
}

VulkanLayernorm::~VulkanLayernorm() {
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    vkbackend->recycleUniform(mParam);
}


ErrorCode VulkanLayernorm::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    // set param
    auto vkBn = (VulkanBackend*)backend();
    auto outside = 1;
    auto inside = 1;
    int rank = inputs.at(0)->dimensions();
    if (mGroup > 1) {
        outside = inputs.at(0)->length(0) * mGroup;
        for (int i = 1; i < rank; i++) {
            inside *= inputs.at(0)->length(i);
        }
        inside /= mGroup;
    } else {
        for (int i = 0; i < rank - mAxisSize; ++i) {
            outside *= inputs.at(0)->length(i);
        }
        for (int i = rank - mAxisSize; i < rank; ++i) {
            inside *= inputs.at(0)->length(i);
        }
    }
    if (mFP16) {
        auto param = reinterpret_cast<FP16Param*>(mParam->map());
        param->size[0] = inside;
        param->size[1] = outside;
        param->size[2] = 1;
        param->size[3] = outside;
        int16_t eps;
        FLOAT_TO_HALF(&mEps, &eps, 1);
        auto epsPtr = reinterpret_cast<int16_t*>(&param->eps);
        epsPtr[0] = eps;
        epsPtr[1] = eps;
        epsPtr[2] = eps;
        epsPtr[3] = eps;
    } else {
        auto param = reinterpret_cast<Param*>(mParam->map());
        param->size[0] = inside;
        param->size[1] = outside;
        param->size[2] = 1;
        param->size[3] = outside;
        param->eps[0] = mEps;
        param->eps[1] = mEps;
        param->eps[2] = mEps;
        param->eps[3] = mEps;
    }
    mParam->unmap();
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    mDesSet->writeBuffer(outputTensor, 0);
    mDesSet->writeBuffer(inputTensor, 1);
    mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
    if (mHasScale) {
        mDesSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 3);
        mDesSet->writeBuffer(vkBn->getBuffer(mBias.get()), 4);
    }
    mPipeline->bind(cmdBuffer->get(), mDesSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(outside, 64), 1, 1);

    return NO_ERROR;
}

class VulkanLayernormCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanLayernorm(op, bn, inputs[0]);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LayerNorm, new VulkanLayernormCreator);
    return true;
}();

} // namespace MNN
