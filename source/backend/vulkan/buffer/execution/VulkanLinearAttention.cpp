//
//  VulkanLinearAttention.cpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanLinearAttention.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

VulkanLinearAttention::VulkanLinearAttention(const MNN::Op* op, Backend* backend)
    : VulkanBasicExecution(backend) {
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();

    auto vkBn = static_cast<VulkanBackend*>(backend);

    // Create pipelines - always use float (not FP16) for numerical stability
    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // qkv
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // conv_state
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // conv_weight
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // conv_out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        mConvSiluPipeline = vkBn->getPipeline("glsl_linear_attn_conv_silu_comp", types);
    }
    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // qkv
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // conv_state
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        mConvStateUpdatePipeline = vkBn->getPipeline("glsl_linear_attn_conv_state_update_comp", types);
    }
    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // conv_out
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // gate
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // beta
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // recurrent_state
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // attn_out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        mGatedDeltaRulePipeline = vkBn->getPipeline("glsl_linear_attn_gated_delta_rule_comp", types);
    }

    // Allocate uniform buffers
    mConvSiluParam = vkBn->allocUniform();
    mConvStateUpdateParam = vkBn->allocUniform();
    mGatedDeltaRuleParam = vkBn->allocUniform();

    // Create descriptor sets
    mConvSiluDesSet.reset(mConvSiluPipeline->createSet());
    mConvStateUpdateDesSet.reset(mConvStateUpdatePipeline->createSet());
    mGatedDeltaRuleDesSet.reset(mGatedDeltaRulePipeline->createSet());
}

VulkanLinearAttention::~VulkanLinearAttention() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    vkBn->recycleUniform(mConvSiluParam);
    vkBn->recycleUniform(mConvStateUpdateParam);
    vkBn->recycleUniform(mGatedDeltaRuleParam);
}

ErrorCode VulkanLinearAttention::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());

    auto qkv = inputs[0];
    int batch = qkv->length(0);
    int convDim = qkv->length(1);
    int seqLen = qkv->length(2);
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int key_dim = mNumKHeads * dk;
    int val_dim = mNumVHeads * dv;
    int gqa_factor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;
    float qScale = 1.0f / sqrtf((float)dk);

    // Allocate persistent states on first call
    if (mFirstResize) {
        // Conv state: [B, D, convStateSize]
        if (convStateSize > 0) {
            mConvState.reset(Tensor::createDevice<float>({batch * convDim * convStateSize}));
            bool success = backend()->onAcquireBuffer(mConvState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            // Zero-initialize via copyToGPUBuffer
            int bufferBytes = batch * convDim * convStateSize * sizeof(float);
            std::vector<float> zeros(batch * convDim * convStateSize, 0.0f);
            auto buf = vkBn->getBuffer(mConvState.get());
            vkBn->copyToGPUBuffer(zeros.data(), std::get<0>(buf), bufferBytes, std::get<2>(buf));
        }

        // Recurrent state: [B, H, d_k, d_v]
        int rnnSize = batch * H * dk * dv;
        mRecurrentState.reset(Tensor::createDevice<float>({rnnSize}));
        bool success = backend()->onAcquireBuffer(mRecurrentState.get(), Backend::STATIC);
        if (!success) return OUT_OF_MEMORY;
        {
            std::vector<float> zeros(rnnSize, 0.0f);
            auto buf = vkBn->getBuffer(mRecurrentState.get());
            vkBn->copyToGPUBuffer(zeros.data(), std::get<0>(buf), rnnSize * sizeof(float), std::get<2>(buf));
        }

        mFirstResize = false;
    }

    // Allocate temporary conv output
    mConvOut.reset(Tensor::createDevice<float>({batch * convDim * seqLen}));
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    // Kernel 1: Conv1D + SiLU
    {
        int totalConvSilu = batch * convDim * seqLen;
        auto paramPtr = reinterpret_cast<int*>(mConvSiluParam->map());
        paramPtr[0] = batch;
        paramPtr[1] = convDim;
        paramPtr[2] = seqLen;
        paramPtr[3] = K_conv;
        paramPtr[4] = convStateSize;
        paramPtr[5] = totalConvSilu;
        paramPtr[6] = 0;
        paramPtr[7] = 0;
        mConvSiluParam->unmap();

        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);      // qkv
        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(mConvState.get()), 1); // conv_state
        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(inputs[3]), 2);      // conv_weight
        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 3); // conv_out
        mConvSiluDesSet->writeBuffer(mConvSiluParam->buffer(), 4, mConvSiluParam->size());

        mConvSiluPipeline->bind(cmdBuffer->get(), mConvSiluDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalConvSilu, 256), 1, 1);

        // Memory barrier: conv_out write -> read, conv_state read -> write
        cmdBuffer->barrierSource(vkBn->getBuffer(mConvOut.get()));
    }

    // Kernel 2: Conv state update
    if (convStateSize > 0) {
        int totalUpdate = batch * convDim * convStateSize;
        auto paramPtr = reinterpret_cast<int*>(mConvStateUpdateParam->map());
        paramPtr[0] = batch;
        paramPtr[1] = convDim;
        paramPtr[2] = seqLen;
        paramPtr[3] = convStateSize;
        paramPtr[4] = totalUpdate;
        paramPtr[5] = 0;
        paramPtr[6] = 0;
        paramPtr[7] = 0;
        mConvStateUpdateParam->unmap();

        mConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);      // qkv
        mConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(mConvState.get()), 1); // conv_state
        mConvStateUpdateDesSet->writeBuffer(mConvStateUpdateParam->buffer(), 2, mConvStateUpdateParam->size());

        mConvStateUpdatePipeline->bind(cmdBuffer->get(), mConvStateUpdateDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalUpdate, 256), 1, 1);

        // Memory barrier: conv_state update complete before next use
        cmdBuffer->barrierSource(vkBn->getBuffer(mConvState.get()));
    }

    // Kernel 3: Gated Delta Rule
    {
        int totalHeads = batch * H;
        auto paramPtr = reinterpret_cast<int*>(mGatedDeltaRuleParam->map());
        paramPtr[0] = batch;
        paramPtr[1] = convDim;
        paramPtr[2] = seqLen;
        paramPtr[3] = mNumKHeads;
        paramPtr[4] = mNumVHeads;
        paramPtr[5] = dk;
        paramPtr[6] = dv;
        paramPtr[7] = key_dim;
        paramPtr[8] = val_dim;
        paramPtr[9] = gqa_factor;
        paramPtr[10] = mUseQKL2Norm ? 1 : 0;
        paramPtr[11] = totalHeads;
        // q_scale as float at offset 12 (vec4 size3)
        auto floatPtr = reinterpret_cast<float*>(paramPtr + 12);
        floatPtr[0] = qScale;
        floatPtr[1] = 0.0f;
        floatPtr[2] = 0.0f;
        floatPtr[3] = 0.0f;
        mGatedDeltaRuleParam->unmap();

        mGatedDeltaRuleDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 0);           // conv_out
        mGatedDeltaRuleDesSet->writeBuffer(vkBn->getBuffer(inputs[1]), 1);                // gate
        mGatedDeltaRuleDesSet->writeBuffer(vkBn->getBuffer(inputs[2]), 2);                // beta
        mGatedDeltaRuleDesSet->writeBuffer(vkBn->getBuffer(mRecurrentState.get()), 3);    // recurrent_state
        mGatedDeltaRuleDesSet->writeBuffer(vkBn->getBuffer(outputs[0]), 4);               // attn_out
        mGatedDeltaRuleDesSet->writeBuffer(mGatedDeltaRuleParam->buffer(), 5, mGatedDeltaRuleParam->size());

        mGatedDeltaRulePipeline->bind(cmdBuffer->get(), mGatedDeltaRuleDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalHeads, 256), 1, 1);
    }

    return NO_ERROR;
}

bool VulkanLinearAttention::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new VulkanLinearAttention(op, bn);
    return true;
}

class VulkanLinearAttentionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) const override {
        return new VulkanLinearAttention(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LinearAttention, new VulkanLinearAttentionCreator);
    return true;
}();

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
