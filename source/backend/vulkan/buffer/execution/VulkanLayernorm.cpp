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
    float eps;
};

static uint32_t _selectLocalSize(int inside, uint32_t maxSize) {
    if (inside <= 1) {
        return 1;
    }
    uint32_t target = (uint32_t)inside;
    if (target > 256) {
        target = 256;
    }
    if (target > maxSize) {
        target = maxSize;
    }
    uint32_t localSize = 1;
    while ((localSize << 1) <= target) {
        localSize <<= 1;
    }
    return localSize;
}

VulkanLayernorm::VulkanLayernorm(const Op* op, Backend* backend, Tensor * tensor) : VulkanBasicExecution(backend) {
    auto layer_norm_param = op->main_as_LayerNorm();
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    if (nullptr != layer_norm_param->axis()) {
        mAxisSize = layer_norm_param->axis()->size();
    }
    mGroup = layer_norm_param->group();
    mUseRMSNorm = layer_norm_param->useRMSNorm();

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

    mKey = "glsl_norm_";
    if (!mHasScale) {
        mDesTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
    } else {
        mDesTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        mKey += "LAYERNORM_SCALE_";
    }
    if (mFP16) {
        mKey += "FP16_";
    }
    mKey += "comp";
    std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u};
    mPipeline = vkbackend->getPipeline(mKey, mDesTypes, {}, spec);
    mDesSet.reset(mPipeline->createSet());

    mOptKey = "glsl_norm_opt_";
    if (mHasScale) {
        mOptKey += "LAYERNORM_SCALE_";
    }
    if (mFP16) {
        mOptKey += "FP16_";
    }
    mOptKey += "comp";
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
    auto param = reinterpret_cast<Param*>(mParam->map());
    param->size[0] = inside;
    param->size[1] = outside;
    param->size[2] = mUseRMSNorm ? 1 : 0;
    param->size[3] = outside;
    param->eps = mEps;
    mParam->unmap();
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    auto maxGroupCountX = (int)vkBn->getDevice().proty().limits.maxComputeWorkGroupCount[0];
    auto maxGroupSizeX = (uint32_t)vkBn->getDevice().proty().limits.maxComputeWorkGroupSize[0];
    // LLM-oriented fast path: 1 workgroup per row (outside), parallel reduce over inside.
    // Requires inside % 4 == 0; fallback when dispatch count might exceed device limits.
    bool useOpt = (outside <= maxGroupCountX) && ((inside & 3) == 0);
    if (useOpt) {
        auto inside4 = inside >> 2;
        auto localSize = _selectLocalSize(inside4, maxGroupSizeX);
        auto sharedSize = localSize;
        if (mOptPipeline == nullptr || mOptLocalSize != localSize) {
            std::vector<uint32_t> localSizeVec = {localSize};
            std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u, sharedSize};
            mOptPipeline = vkBn->getPipeline(mOptKey, mDesTypes, localSizeVec, spec);
            mOptDesSet.reset(mOptPipeline->createSet());
            mOptLocalSize = localSize;
        }
        mOptDesSet->writeBuffer(outputTensor, 0);
        mOptDesSet->writeBuffer(inputTensor, 1);
        mOptDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
        if (mHasScale) {
            mOptDesSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 3);
            mOptDesSet->writeBuffer(vkBn->getBuffer(mBias.get()), 4);
        }
        mOptPipeline->bind(cmdBuffer->get(), mOptDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), outside, 1, 1);
    } else {
        mDesSet->writeBuffer(outputTensor, 0);
        mDesSet->writeBuffer(inputTensor, 1);
        mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
        if (mHasScale) {
            mDesSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 3);
            mDesSet->writeBuffer(vkBn->getBuffer(mBias.get()), 4);
        }
        mPipeline->bind(cmdBuffer->get(), mDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(outside, 64), 1, 1);
    }

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
