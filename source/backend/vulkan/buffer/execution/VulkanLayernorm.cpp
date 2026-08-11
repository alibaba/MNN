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

VulkanLayernorm::VulkanLayernorm(const Op* op, Backend* backend, Tensor* tensor, bool binaryC4)
    : VulkanBasicExecution(backend), mIsBinaryC4(binaryC4) {
    auto layer_norm_param = op->main_as_LayerNorm();
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    if (nullptr != layer_norm_param->axis()) {
        mAxisSize = layer_norm_param->axis()->size();
    }
    mGroup = layer_norm_param->group();
    mUseRMSNorm = layer_norm_param->useRMSNorm();
    mIsNC4HW4 = TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;

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

    mKey = mIsBinaryC4 ? "glsl_norm_binary_" : "glsl_norm_";
    if (mIsBinaryC4) {
        mDesTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        if (mHasScale) {
            mDesTypes.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            mDesTypes.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            mKey += "LAYERNORM_SCALE_";
        }
        if (mFP16) {
            mKey += "FP16_";
        }
        mKey += "comp";
        std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u};
        mPipeline = vkbackend->getPipeline(mKey, mDesTypes, {}, spec);
        mDesSet.reset(mPipeline->createSet());
        return;
    }
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
}

VulkanLayernorm::VulkanLayernorm(Backend* bn, const VulkanLayernorm* src)
    : VulkanBasicExecution(bn),
      mEps(src->mEps),
      mHasScale(src->mHasScale),
      mUseRMSNorm(src->mUseRMSNorm),
      mGroup(src->mGroup),
      mAxisSize(src->mAxisSize),
      mFP16(src->mFP16),
      mIsNC4HW4(src->mIsNC4HW4),
      mIsBinaryC4(src->mIsBinaryC4),
      mKey(src->mKey),
      mOptKey(src->mOptKey),
      mDesTypes(src->mDesTypes) {
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam = vkbackend->allocUniform();
    mGamma = src->mGamma;
    mBias = src->mBias;
    std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u};
    mPipeline = vkbackend->getPipeline(mKey, mDesTypes, {}, spec);
    mDesSet.reset(mPipeline->createSet());
    if (!mIsBinaryC4) {
        mOptPipeline = vkbackend->getPipeline(mOptKey, mDesTypes, {}, spec);
        mOptDesSet.reset(mOptPipeline->createSet());
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
    auto outside = 1;
    auto inside = 1;
    auto area = 1;
    int rank = inputs.at(0)->dimensions();
    if (mIsNC4HW4) {
        inside = inputs[0]->length(1);
        for (int i = 0; i < rank; ++i) {
            if (i != 1) {
                outside *= inputs[0]->length(i);
            }
        }
        for (int i = 2; i < rank; ++i) {
            area *= inputs[0]->length(i);
        }
    } else if (mGroup > 1) {
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
    if (mIsNC4HW4 && (mGroup > 1 || rank < 2 || inputs[0]->length(1) <= 0 || area != 1 ||
                      (mIsBinaryC4 && inside % 4 != 0))) {
        MNN_ERROR("Vulkan LayerNorm: unsupported C4 shape or axis.\n");
        return NOT_SUPPORT;
    }
    auto param = reinterpret_cast<Param*>(mParam->map());
    param->size[0] = inside;
    param->size[1] = outside;
    param->size[2] = mIsNC4HW4 ? 1 : 0;
    param->size[3] = outside;
    param->eps = mEps;
    mParam->unmap();
    if (mIsBinaryC4) {
        if (inputs.size() != 2 || outputs.size() != 2 ||
            TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            TensorUtils::getDescribe(outputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            TensorUtils::getDescribe(outputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            inputs[0]->shape() != inputs[1]->shape() || inputs[0]->shape() != outputs[0]->shape() ||
            inputs[0]->shape() != outputs[1]->shape()) {
            MNN_ERROR("Vulkan LayerNorm: invalid binary C4 inputs or outputs.\n");
            return NOT_SUPPORT;
        }
        mDesSet->writeBuffer(vkBn->getBuffer(outputs[0]), 0);
        mDesSet->writeBuffer(vkBn->getBuffer(outputs[1]), 1);
        mDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 2);
        mDesSet->writeBuffer(vkBn->getBuffer(inputs[1]), 3);
        mDesSet->writeBuffer(mParam->buffer(), 4, mParam->size());
        if (mHasScale) {
            mDesSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 5);
            mDesSet->writeBuffer(vkBn->getBuffer(mBias.get()), 6);
        }
        mPipeline->bind(cmdBuffer->get(), mDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), outside, 1, 1);
        return NO_ERROR;
    }
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    auto maxGroupCountX = (int)vkBn->getDevice().proty().limits.maxComputeWorkGroupCount[0];
    // LLM-oriented fast path: 1 workgroup per row (outside), parallel reduce over inside.
    // Requires inside % 4 == 0; fallback when dispatch count might exceed device limits.
    bool useOpt = (outside <= maxGroupCountX) && ((inside & 3) == 0);
    if (useOpt) {
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
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* bn) const override {
        const bool single = inputs.size() == 1 && outputs.size() == 1;
        const bool binary = inputs.size() == 2 && outputs.size() == 2;
        bool binaryC4 = false;
        if (single && (op->defaultDimentionFormat() == MNN_DATA_FORMAT_NC4HW4 ||
                       TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4)) {
            TensorUtils::getDescribe(inputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        } else if (binary && op->defaultDimentionFormat() == MNN_DATA_FORMAT_NC4HW4) {
            if (inputs[0]->dimensions() < 2 ||
                (inputs[0]->length(1) > 0 && inputs[0]->length(1) % 4 != 0)) {
                return nullptr;
            }
            binaryC4 = true;
            for (auto input : inputs) {
                TensorUtils::getDescribe(input)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
            for (auto output : outputs) {
                TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
        }
        if ((inputs.size() != 1 || outputs.size() != 1) && !binaryC4) {
            return nullptr;
        }
        return new VulkanLayernorm(op, bn, inputs[0], binaryC4);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LayerNorm, new VulkanLayernormCreator);
    return true;
}();

} // namespace MNN
