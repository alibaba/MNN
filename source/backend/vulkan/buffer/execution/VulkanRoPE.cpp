#include "VulkanRoPE.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace MNN {

struct RoPEParam {
    int outerSize;
    int halfD;
    int ropeHalfD;
    int headDim;
    int numHead;
    int kvNumHead;
    float qEps;
    float kEps;
};

static bool _supportRoPESubgroup(const VulkanDevice& device) {
    const auto& subgroup = device.getSubgroupInfo();
    if (subgroup.size < 64) {
        return false;
    }
    if (0 == (subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        return false;
    }
    const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    return (subgroup.ops & required) == required;
}

static std::shared_ptr<VulkanBuffer> createGammaBuffer(VulkanBackend* vkBn, const LayerNorm* param) {
    if (nullptr == vkBn || nullptr == param || nullptr == param->gamma()) {
        return nullptr;
    }
    const int size = param->gamma()->size();
    if (size <= 0) {
        return nullptr;
    }
    std::vector<float> gamma(ALIGN_UP4(size), 0.0f);
    ::memcpy(gamma.data(), param->gamma()->data(), size * sizeof(float));
    std::shared_ptr<VulkanBuffer> buffer(
        new VulkanBuffer(vkBn->getMemoryPool(), false, gamma.size() * sizeof(float), nullptr,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    vkBn->copyToGPUBuffer(gamma.data(), buffer->buffer(), gamma.size() * sizeof(float), 0);
    return buffer;
}

VulkanRoPE::VulkanRoPE(const Op* op, Backend* backend) : VulkanBasicExecution(backend) {
    parseAttrs(op);
    initPipeline();
}

VulkanRoPE::VulkanRoPE(Backend* backend, int ropeCutHeadDim, std::shared_ptr<VulkanBuffer> qGamma, float qEps,
                       std::shared_ptr<VulkanBuffer> kGamma, float kEps)
    : VulkanBasicExecution(backend),
      mRopeCutHeadDim(ropeCutHeadDim),
      mQGamma(std::move(qGamma)),
      mKGamma(std::move(kGamma)),
      mQEps(qEps),
      mKEps(kEps) {
    initPipeline();
}

VulkanRoPE::~VulkanRoPE() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (nullptr != vkBn && nullptr != mConstBuffer) {
        vkBn->recycleUniform(mConstBuffer);
    }
}

bool VulkanRoPE::useNorm() const {
    return nullptr != mQGamma || nullptr != mKGamma;
}

void VulkanRoPE::parseAttrs(const Op* op) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (nullptr == op || OpParameter_Extra != op->main_type()) {
        return;
    }
    auto extra = op->main_as_Extra();
    if (nullptr == extra || nullptr == extra->attr()) {
        return;
    }
    for (int i = 0; i < extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (nullptr == attr || nullptr == attr->key()) {
            continue;
        }
        const auto key = attr->key()->str();
        if (key == "rope_cut_head_dim") {
            mRopeCutHeadDim = attr->i();
            continue;
        }
        if ((key == "q_norm" || key == "k_norm") && nullptr != attr->tensor() && nullptr != attr->tensor()->int8s()) {
            auto normOp = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
            if (nullptr == normOp || nullptr == normOp->main_as_LayerNorm()) {
                continue;
            }
            auto param = normOp->main_as_LayerNorm();
            if (key == "q_norm") {
                mQEps = param->epsilon();
                mQGamma = createGammaBuffer(vkBn, param);
            } else {
                mKEps = param->epsilon();
                mKGamma = createGammaBuffer(vkBn, param);
            }
        }
    }
}

void VulkanRoPE::initPipeline() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    mConstBuffer = vkBn->allocUniform();
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    std::string key = "glsl_rope_";
    if (mQGamma && mKGamma) {
        key += "Q_NORM_K_NORM_";
        types.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        types.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    } else if (mQGamma) {
        key += "Q_NORM_";
        types.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    } else if (mKGamma) {
        key += "K_NORM_";
        types.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    if (vkBn->useFP16()) {
        key += "FP16_";
    }
    key += "comp";
    mPipeline = vkBn->getPipeline(key, types);
    if (nullptr != mPipeline) {
        mDescriptorSet.reset(mPipeline->createSet());
    }

    if (useNorm() && _supportRoPESubgroup(vkBn->getDevice())) {
        mSubgroupSize = vkBn->getDevice().getSubgroupSize();
        if (mSubgroupSize > 0) {
            std::string subgroupKey = "glsl_rope_subgroup_";
            if (mQGamma && mKGamma) {
                subgroupKey += "Q_NORM_K_NORM_";
            } else if (mQGamma) {
                subgroupKey += "Q_NORM_";
            } else if (mKGamma) {
                subgroupKey += "K_NORM_";
            }
            if (vkBn->useFP16()) {
                subgroupKey += "FP16_";
            }
            subgroupKey += "comp";
            mSubgroupPipeline = vkBn->getPipeline(subgroupKey, types, {mSubgroupSize});
            if (nullptr != mSubgroupPipeline) {
                mSubgroupDescriptorSet.reset(mSubgroupPipeline->createSet());
            }
        }
    }
}

bool VulkanRoPE::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return nullptr != mPipeline && nullptr != mDescriptorSet;
    }
    *dst = new VulkanRoPE(bn, mRopeCutHeadDim, mQGamma, mQEps, mKGamma, mKEps);
    return true;
}

ErrorCode VulkanRoPE::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    if (inputs.size() != 6 || outputs.size() != 2 || nullptr == mPipeline || nullptr == mDescriptorSet ||
        nullptr == mConstBuffer) {
        return NOT_SUPPORT;
    }
    auto q = inputs[0];
    auto k = inputs[1];
    const int batch = q->length(0);
    const int seqLen = q->length(1);
    const int numHead = q->length(2);
    const int headDim = q->length(3);
    const int kvNumHead = k->length(2);
    const int halfD = headDim / 2;
    int ropeDim = mRopeCutHeadDim;
    if (ropeDim <= 0 || ropeDim > headDim) {
        ropeDim = headDim;
    }
    ropeDim = (ropeDim / 2) * 2;
    int ropeHalfD = std::min(ropeDim / 2, halfD);
    const int outerSize = batch * seqLen;

    auto param = reinterpret_cast<RoPEParam*>(mConstBuffer->map());
    param->outerSize = outerSize;
    param->halfD = halfD;
    param->ropeHalfD = ropeHalfD;
    param->headDim = headDim;
    param->numHead = numHead;
    param->kvNumHead = kvNumHead;
    param->qEps = mQEps;
    param->kEps = mKEps;
    mConstBuffer->unmap();

    auto vkBn = static_cast<VulkanBackend*>(backend());
    const bool useSubgroup = nullptr != mSubgroupPipeline && nullptr != mSubgroupDescriptorSet;
    const auto* pipeline = useSubgroup ? mSubgroupPipeline : mPipeline;
    auto& descriptorSet = useSubgroup ? mSubgroupDescriptorSet : mDescriptorSet;
    descriptorSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
    descriptorSet->writeBuffer(vkBn->getBuffer(inputs[1]), 1);
    descriptorSet->writeBuffer(vkBn->getBuffer(inputs[2]), 2);
    descriptorSet->writeBuffer(vkBn->getBuffer(inputs[3]), 3);
    descriptorSet->writeBuffer(vkBn->getBuffer(inputs[4]), 4);
    descriptorSet->writeBuffer(vkBn->getBuffer(inputs[5]), 5);
    descriptorSet->writeBuffer(vkBn->getBuffer(outputs[0]), 6);
    descriptorSet->writeBuffer(vkBn->getBuffer(outputs[1]), 7);
    descriptorSet->writeBuffer(mConstBuffer->buffer(), 8, mConstBuffer->size());
    uint32_t binding = 9;
    if (mQGamma) {
        descriptorSet->writeBuffer(mQGamma->buffer(), binding++, mQGamma->size());
    }
    if (mKGamma) {
        descriptorSet->writeBuffer(mKGamma->buffer(), binding++, mKGamma->size());
    }

    const uint32_t gx = useNorm() ? 1u : static_cast<uint32_t>(halfD);
    const uint32_t gy = static_cast<uint32_t>(outerSize);
    const uint32_t gz = static_cast<uint32_t>(numHead + kvNumHead);
    pipeline->bind(cmdBuffer->get(), descriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(gx, 64u), gy, gz);
    return NO_ERROR;
}

class VulkanRoPECreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) const override {
        for (auto input : inputs) {
            TensorUtils::setTensorSupportPack(input, false);
        }
        for (auto output : outputs) {
            TensorUtils::setTensorSupportPack(output, false);
        }
        return new VulkanRoPE(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_RoPE, new VulkanRoPECreator);
    return true;
}();

} // namespace MNN
