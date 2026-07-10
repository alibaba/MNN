#include "VulkanRoPE.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

VulkanRoPE::VulkanRoPE(const Op* op, Backend* backend) : VulkanBasicExecution(backend) {
    auto vkBn = static_cast<VulkanBackend*>(backend);
    auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
    if (param == nullptr) {
        return;
    }
    mRopeCutHeadDim = param->rope_cut_head_dim();
    mNumHead = param->num_head();
    mKvNumHead = param->kv_num_head();
    mHeadDim = param->head_dim();
    mUseFP16 = vkBn->useFP16();
    mQNorm = prepareGamma(param->q_norm(), mQGamma, mQEps);
    mKNorm = prepareGamma(param->k_norm(), mKGamma, mKEps);
    mValid = mNumHead > 0 && mKvNumHead > 0 && mHeadDim > 0 && (param->q_norm() == nullptr || mQNorm) &&
             (param->k_norm() == nullptr || mKNorm);
    mParam = vkBn->allocUniform(nullptr, sizeof(GpuParam));
    createPipeline();
}

VulkanRoPE::VulkanRoPE(Backend* backend, const VulkanRoPE* source)
    : VulkanBasicExecution(backend),
      mRopeCutHeadDim(source->mRopeCutHeadDim),
      mNumHead(source->mNumHead),
      mKvNumHead(source->mKvNumHead),
      mHeadDim(source->mHeadDim),
      mQEps(source->mQEps),
      mKEps(source->mKEps),
      mQNorm(source->mQNorm),
      mKNorm(source->mKNorm),
      mValid(source->mValid),
      mUseFP16(source->mUseFP16),
      mQGamma(source->mQGamma),
      mKGamma(source->mKGamma) {
    auto vkBn = static_cast<VulkanBackend*>(backend);
    mParam = vkBn->allocUniform(nullptr, sizeof(GpuParam));
    createPipeline();
}

VulkanRoPE::~VulkanRoPE() {
    if (mParam) {
        static_cast<VulkanBackend*>(backend())->recycleUniform(mParam);
    }
}

bool VulkanRoPE::prepareGamma(const LayerNorm* norm, std::shared_ptr<Tensor>& gamma, float& eps) {
    if (norm == nullptr) {
        return false;
    }
    eps = norm->epsilon();
    if (norm->gamma() == nullptr || norm->gamma()->size() != mHeadDim || !norm->useRMSNorm()) {
        MNN_ERROR("Vulkan RoPE: q/k norm must be RMSNorm with headDim gamma.\n");
        return false;
    }
    gamma.reset(Tensor::createDevice<float>({mHeadDim}));
    if (!backend()->onAcquireBuffer(gamma.get(), Backend::STATIC)) {
        gamma.reset();
        return false;
    }
    const void* gammaData = norm->gamma()->data();
    std::vector<int16_t> gammaFP16;
    if (mUseFP16) {
        gammaFP16.resize(mHeadDim);
        FLOAT_TO_HALF(norm->gamma()->data(), gammaFP16.data(), mHeadDim);
        gammaData = gammaFP16.data();
    }
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto buffer = vkBn->getBuffer(gamma.get());
    vkBn->copyToGPUBuffer(gammaData, std::get<0>(buffer), std::get<1>(buffer), std::get<2>(buffer));
    return true;
}

void VulkanRoPE::createPipeline() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    std::string name = "glsl_rope_";
    if (mUseFP16) {
        name += "FP16_";
    }
    name += "comp";
    std::vector<VkDescriptorType> types(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    types.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    mPipeline = vkBn->getPipeline(name, types);
    if (mPipeline != nullptr) {
        mDescriptorSet.reset(mPipeline->createSet());
    }
}

bool VulkanRoPE::onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) {
    if (dst == nullptr) {
        return true;
    }
    *dst = new VulkanRoPE(backend, this);
    return true;
}

ErrorCode VulkanRoPE::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    if (!mValid || mPipeline == nullptr || mDescriptorSet == nullptr || inputs.size() != 4 || outputs.size() != 2) {
        return NOT_SUPPORT;
    }
    auto q = inputs[0];
    auto k = inputs[1];
    const bool valid = q != nullptr && k != nullptr && inputs[2] != nullptr && inputs[3] != nullptr &&
                       outputs[0] != nullptr && outputs[1] != nullptr && q->dimensions() == 4 && k->dimensions() == 4 &&
                       q->length(0) == k->length(0) && q->length(1) == mNumHead * mHeadDim &&
                       k->length(1) == mKvNumHead * mHeadDim && q->length(2) == 1 && q->length(3) == 1 &&
                       k->length(2) == 1 && k->length(3) == 1 &&
                       TensorUtils::getDescribe(q)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                       TensorUtils::getDescribe(k)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                       TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC &&
                       TensorUtils::getDescribe(outputs[1])->dimensionFormat == MNN_DATA_FORMAT_NHWC;
    if (!valid) {
        MNN_ERROR("Vulkan RoPE: invalid q/k layout or head configuration.\n");
        return NOT_SUPPORT;
    }
    const int seqLen = q->length(0);
    int ropeDim = mRopeCutHeadDim;
    if (ropeDim <= 0 || ropeDim > mHeadDim) {
        ropeDim = mHeadDim;
    }
    ropeDim = ropeDim / 2 * 2;
    if (ropeDim <= 0 || inputs[2]->elementSize() < seqLen * ropeDim || inputs[3]->elementSize() < seqLen * ropeDim) {
        MNN_ERROR("Vulkan RoPE: invalid rotary table shape.\n");
        return NOT_SUPPORT;
    }

    auto gpuParam = reinterpret_cast<GpuParam*>(mParam->map());
    gpuParam->size0[0] = seqLen;
    gpuParam->size0[1] = mHeadDim;
    gpuParam->size0[2] = mNumHead;
    gpuParam->size0[3] = mKvNumHead;
    gpuParam->size1[0] = ropeDim / 2;
    gpuParam->size1[1] = mQNorm ? 1 : 0;
    gpuParam->size1[2] = mKNorm ? 1 : 0;
    gpuParam->size1[3] = 0;
    gpuParam->eps[0] = mQEps;
    gpuParam->eps[1] = mKEps;
    gpuParam->eps[2] = 0.0f;
    gpuParam->eps[3] = 0.0f;
    mParam->unmap();

    auto vkBn = static_cast<VulkanBackend*>(backend());
    mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[1]), 1);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[2]), 2);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[3]), 3);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputs[0]), 4);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputs[1]), 5);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(mQGamma ? mQGamma.get() : inputs[0]), 6);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(mKGamma ? mKGamma.get() : inputs[1]), 7);
    mDescriptorSet->writeBuffer(mParam->buffer(), 8, mParam->size());
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), static_cast<uint32_t>(mNumHead + mKvNumHead), static_cast<uint32_t>(seqLen), 1);
    return NO_ERROR;
}

class VulkanRoPECreator : public VulkanBackend::Creator {
public:
    VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const Op* op, Backend* backend) const override {
        if (inputs.size() != 4 || outputs.size() != 2 ||
            TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            TensorUtils::getDescribe(inputs[1])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        return new VulkanRoPE(op, backend);
    }
};

static bool gRegisterVulkanRoPE = []() {
    VulkanBackend::addCreator(OpType_RoPE, new VulkanRoPECreator);
    return true;
}();

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
