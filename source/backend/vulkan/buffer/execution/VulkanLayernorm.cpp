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

VulkanLayernorm::VulkanLayernorm(const Op* op, Backend* backend) : VulkanBasicExecution(backend) {
    auto layer_norm_param = op->main_as_LayerNorm();
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    if (nullptr != layer_norm_param->axis()) {
        mAxisSize = layer_norm_param->axis()->size();
    }
    mGroup = layer_norm_param->group();

    mParam = vkbackend->allocUniform();
    mEps = layer_norm_param->epsilon();

    if (layer_norm_param->gamma() && layer_norm_param->beta()) {
        mHasScale = true;
        int size = layer_norm_param->gamma()->size();
        mGamma.reset(Tensor::createDevice<float>({size}));
        auto status = backend->onAcquireBuffer(mGamma.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in LayerNorm.\n");
            return;
        }
        const float* gamma_data = layer_norm_param->gamma()->data();
        auto gammaBuffer = vkbackend->getBuffer(mGamma.get());
        vkbackend->copyToGPUBuffer(gamma_data, std::get<0>(gammaBuffer), std::get<1>(gammaBuffer), std::get<2>(gammaBuffer));

        if (layer_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in LayerNorm.\n");
            return;
        }
        mBias.reset(Tensor::createDevice<float>({size}));
        status = backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when beta is acquired in LayerNorm.\n");
            return;
        }
        const float* beta_data = layer_norm_param->beta()->data();
        auto betaBuffer = vkbackend->getBuffer(mBias.get());
        vkbackend->copyToGPUBuffer(beta_data, std::get<0>(betaBuffer), std::get<1>(betaBuffer), std::get<2>(betaBuffer));
    }

    if (!mHasScale) {
        auto types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mPipeline = vkbackend->getPipeline("glsl_norm_comp", types);
    } else {
        auto types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        mPipeline = vkbackend->getPipeline("glsl_norm_LAYERNORM_SCALE_comp", types);
    }
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
    }
    for (int i = 0; i < rank - mAxisSize; ++i) {
        outside *= inputs.at(0)->length(i);
    }
    for (int i = rank - mAxisSize; i < rank; ++i) {
        inside *= inputs.at(0)->length(i);
    }
    auto param = reinterpret_cast<Param*>(mParam->map());
    param->size[0] = inside;
    param->size[1] = outside;
    param->size[2] = 1;
    param->size[3] = outside;
    param->eps[0] = mEps;
    param->eps[1] = mEps;
    param->eps[2] = mEps;
    param->eps[3] = mEps;
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
        return new VulkanLayernorm(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LayerNorm, new VulkanLayernormCreator);
    return true;
}();

} // namespace MNN
