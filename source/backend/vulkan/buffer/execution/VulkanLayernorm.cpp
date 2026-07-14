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
#include "backend/vulkan/vulkan/vulkan_wrapper.h"

namespace MNN {

struct Param {
    ivec4 size;
    float eps;
};

static bool _supportLayerNormSubgroup(const VulkanDevice& device) {
    const auto& subgroup = device.getSubgroupInfo();
    if (subgroup.size < 16) {
        return false;
    }
    if (0 == (subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        return false;
    }
    const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    return (subgroup.ops & required) == required;
}

VulkanLayernorm::VulkanLayernorm(const Op* op, Backend* backend, Tensor* tensor) : VulkanBasicExecution(backend) {
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
            const void* paramData;
            std::vector<int16_t> paramFP16;
            if (mFP16) {
                paramFP16.resize(size);
                FLOAT_TO_HALF(sourceData, paramFP16.data(), size);
                paramData = paramFP16.data();
            } else {
                paramData = (const void*)sourceData;
            }
            auto paramBuffer = vkbackend->getBuffer(paramTensor.get());
            vkbackend->copyToGPUBuffer(paramData, std::get<0>(paramBuffer), std::get<1>(paramBuffer),
                                       std::get<2>(paramBuffer));
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
    mOptPipeline = vkbackend->getPipeline(mOptKey, mDesTypes, {}, spec);
    mOptDesSet.reset(mOptPipeline->createSet());

    if (_supportLayerNormSubgroup(vkbackend->getDevice())) {
        mSubgroupSize = vkbackend->getDevice().getSubgroupSize();
        if (mSubgroupSize > 0) {
            mOptSubgroupKey = "glsl_norm_opt_subgroup_";
            if (mHasScale) {
                mOptSubgroupKey += "LAYERNORM_SCALE_";
            }
            if (mFP16) {
                mOptSubgroupKey += "FP16_";
            }
            mOptSubgroupKey += "comp";
            mOptSubgroupPipeline = vkbackend->getPipeline(mOptSubgroupKey, mDesTypes, {mSubgroupSize}, spec);
            if (nullptr != mOptSubgroupPipeline) {
                mOptSubgroupDesSet.reset(mOptSubgroupPipeline->createSet());
            }
        }
    }

    mC4Key = "glsl_norm_c4_";
    mC4DesTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    if (mHasScale) {
        mC4Key += "LAYERNORM_SCALE_";
        mC4DesTypes.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        mC4DesTypes.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    if (mFP16) {
        mC4Key += "FP16_";
    }
    mC4Key += "comp";
    mC4Pipeline = vkbackend->getPipeline(mC4Key, mC4DesTypes, {}, spec);
    if (nullptr != mC4Pipeline) {
        mC4DesSet.reset(mC4Pipeline->createSet());
    }
    if (mSubgroupSize > 0) {
        mC4SubgroupKey = "glsl_norm_c4_subgroup_";
        if (mHasScale) {
            mC4SubgroupKey += "LAYERNORM_SCALE_";
        }
        if (mFP16) {
            mC4SubgroupKey += "FP16_";
        }
        mC4SubgroupKey += "comp";
        mC4SubgroupPipeline = vkbackend->getPipeline(mC4SubgroupKey, mC4DesTypes, {mSubgroupSize}, spec);
        if (nullptr != mC4SubgroupPipeline) {
            mC4SubgroupDesSet.reset(mC4SubgroupPipeline->createSet());
        }
    }
}

VulkanLayernorm::VulkanLayernorm(Backend* bn, const VulkanLayernorm* src)
    : VulkanBasicExecution(bn),
      mEps(src->mEps),
      mHasScale(src->mHasScale),
      mUseRMSNorm(src->mUseRMSNorm),
      mGroup(src->mGroup),
      mAxisSize(src->mAxisSize),
      mFP16(src->mFP16),
      mSubgroupSize(src->mSubgroupSize),
      mKey(src->mKey),
      mOptKey(src->mOptKey),
      mOptSubgroupKey(src->mOptSubgroupKey),
      mC4Key(src->mC4Key),
      mC4SubgroupKey(src->mC4SubgroupKey),
      mDesTypes(src->mDesTypes),
      mC4DesTypes(src->mC4DesTypes) {
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam = vkbackend->allocUniform();
    mGamma = src->mGamma;
    mBias = src->mBias;
    std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u};
    mPipeline = vkbackend->getPipeline(mKey, mDesTypes, {}, spec);
    mDesSet.reset(mPipeline->createSet());
    mOptPipeline = vkbackend->getPipeline(mOptKey, mDesTypes, {}, spec);
    mOptDesSet.reset(mOptPipeline->createSet());
    if (!mOptSubgroupKey.empty() && mSubgroupSize > 0) {
        mOptSubgroupPipeline = vkbackend->getPipeline(mOptSubgroupKey, mDesTypes, {mSubgroupSize}, spec);
        if (nullptr != mOptSubgroupPipeline) {
            mOptSubgroupDesSet.reset(mOptSubgroupPipeline->createSet());
        }
    }
    mC4Pipeline = vkbackend->getPipeline(mC4Key, mC4DesTypes, {}, spec);
    if (nullptr != mC4Pipeline) {
        mC4DesSet.reset(mC4Pipeline->createSet());
    }
    if (!mC4SubgroupKey.empty() && mSubgroupSize > 0) {
        mC4SubgroupPipeline = vkbackend->getPipeline(mC4SubgroupKey, mC4DesTypes, {mSubgroupSize}, spec);
        if (nullptr != mC4SubgroupPipeline) {
            mC4SubgroupDesSet.reset(mC4SubgroupPipeline->createSet());
        }
    }
}

VulkanLayernorm::~VulkanLayernorm() {
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    vkbackend->recycleUniform(mParam);
}

bool VulkanLayernorm::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanLayernorm(bn, this);
    *dst = res;
    return true;
}

ErrorCode VulkanLayernorm::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* cmdBuffer) {
    // set param
    auto vkBn = (VulkanBackend*)backend();
    const bool needUnpackC4 =
        !inputs.empty() && TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    if (needUnpackC4) {
        if (nullptr == mC4Pipeline || nullptr == mC4DesSet || inputs.empty() || outputs.empty()) {
            return NOT_SUPPORT;
        }
        const bool hasResidual = inputs.size() == 2 && outputs.size() == 2;
        int channel = inputs[0]->length(1);
        int outside = inputs[0]->length(0);
        for (int i = 2; i < inputs[0]->dimensions(); ++i) {
            outside *= inputs[0]->length(i);
        }
        auto param = reinterpret_cast<Param*>(mParam->map());
        param->size[0] = channel;
        param->size[1] = outside;
        param->size[2] = mUseRMSNorm ? 1 : 0;
        param->size[3] = hasResidual ? 1 : 0;
        param->eps = mEps;
        mParam->unmap();

        auto input0 = vkBn->getBuffer(inputs[0]);
        auto input1 = hasResidual ? vkBn->getBuffer(inputs[1]) : input0;
        auto rawOutput = vkBn->getBuffer(outputs[0]);
        auto normOutput = hasResidual ? vkBn->getBuffer(outputs[1]) : rawOutput;
        auto& c4Set = (nullptr != mC4SubgroupPipeline && nullptr != mC4SubgroupDesSet) ? mC4SubgroupDesSet : mC4DesSet;
        const auto* c4Pipeline =
            (nullptr != mC4SubgroupPipeline && nullptr != mC4SubgroupDesSet) ? mC4SubgroupPipeline : mC4Pipeline;
        c4Set->writeBuffer(input0, 0);
        c4Set->writeBuffer(input1, 1);
        c4Set->writeBuffer(rawOutput, 2);
        c4Set->writeBuffer(normOutput, 3);
        c4Set->writeBuffer(mParam->buffer(), 4, mParam->size());
        if (mHasScale) {
            c4Set->writeBuffer(vkBn->getBuffer(mGamma.get()), 5);
            c4Set->writeBuffer(vkBn->getBuffer(mBias.get()), 6);
        }
        c4Pipeline->bind(cmdBuffer->get(), c4Set->get());
        vkCmdDispatch(cmdBuffer->get(), outside, 1, 1);
        return NO_ERROR;
    }
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
    // LLM-oriented fast path: 1 workgroup per row (outside), parallel reduce over inside.
    // Requires inside % 4 == 0; fallback when dispatch count might exceed device limits.
    bool useOpt = (outside <= maxGroupCountX) && ((inside & 3) == 0);
    if (useOpt) {
        auto& optSet =
            (nullptr != mOptSubgroupPipeline && nullptr != mOptSubgroupDesSet) ? mOptSubgroupDesSet : mOptDesSet;
        const auto* optPipeline =
            (nullptr != mOptSubgroupPipeline && nullptr != mOptSubgroupDesSet) ? mOptSubgroupPipeline : mOptPipeline;
        optSet->writeBuffer(outputTensor, 0);
        optSet->writeBuffer(inputTensor, 1);
        optSet->writeBuffer(mParam->buffer(), 2, mParam->size());
        if (mHasScale) {
            optSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 3);
            optSet->writeBuffer(vkBn->getBuffer(mBias.get()), 4);
        }
        optPipeline->bind(cmdBuffer->get(), optSet->get());
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
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* bn) const override {
        return new VulkanLayernorm(op, bn, inputs[0]);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LayerNorm, new VulkanLayernormCreator);
    return true;
}();

} // namespace MNN
