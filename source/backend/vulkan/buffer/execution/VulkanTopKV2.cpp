//
//  VulkanTopKV2.cpp
//  MNN
//
//  Vulkan buffer-mode implementation of TopKV2.
//

#include "VulkanTopKV2.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef ENABLE_VULKAN_TIME_PROFILE
#include "backend/vulkan/component/VulkanTimeProfiler.hpp"
#endif

namespace MNN {

struct TopKV2ConstBuffer {
    int rowSize;
    int k;
    int numRows;
    int pad;
};

static int _selectTop1ThreadNumber(const VulkanBackend* backend) {
    if (nullptr == backend) {
        return 256;
    }
    const auto& properties = backend->getDevice().proty();
    if (backend->gpuType() == VulkanRuntime::MALI || properties.vendorID == 0x13B5) {
        return 256;
    }
    if (properties.limits.maxComputeWorkGroupInvocations < 1024 ||
        properties.limits.maxComputeWorkGroupSize[0] < 1024) {
        return 256;
    }
    std::string deviceName = properties.deviceName;
    return deviceName.find("Mali") == std::string::npos ? 1024 : 256;
}

static int _topKFromOutput(const Tensor* output) {
    if (nullptr == output || output->dimensions() <= 0) {
        return 0;
    }
    return output->length(output->dimensions() - 1);
}

VulkanTopKV2::VulkanTopKV2(const Op* op, Backend* bn, Tensor* input, Tensor* output) : VulkanBasicExecution(bn) {
    MNN_ASSERT(input != nullptr);
    MNN_ASSERT(_topKFromOutput(output) > 0);
    auto vkBn = (VulkanBackend*)backend();
    mLargest = true;
    auto param = op->main_as_TopKV2();
    if (nullptr != param) {
        mLargest = param->largest();
    }

    mConstBuffer = vkBn->allocUniform();
    _createPipeline(_topKFromOutput(output), input);
}

bool VulkanTopKV2::_createPipeline(int k, Tensor* input) {
    if (k <= 0 || nullptr == input) {
        return false;
    }
    auto vkBn = (VulkanBackend*)backend();
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::string pKey;
    if (k == 1) {
        mTop1ThreadNumber = _selectTop1ThreadNumber(vkBn);
        pKey = "glsl_topkv2_top1_";
    } else {
        pKey = "glsl_topkv2_";
    }
    if (mLargest) {
        pKey += "SORT_DESC_";
    }
    if (input->getType().code == halide_type_float && vkBn->useFP16()) {
        pKey += "FP16_";
    }
    pKey += "comp";

    if (mPipeline != nullptr && mPipelineName == pKey && mK == k) {
        return true;
    }
    mPipelineName = pKey;
    if (k == 1) {
        mPipeline = vkBn->getPipeline(pKey, types, {(uint32_t)mTop1ThreadNumber});
    } else {
        mPipeline = vkBn->getPipeline(pKey, types);
    }
    if (nullptr == mPipeline) {
        mDescriptorSet.reset();
        return false;
    }
    mK = k;
    mDescriptorSet.reset(mPipeline->createSet());
    return mDescriptorSet != nullptr;
}

VulkanTopKV2::~VulkanTopKV2() {
    auto vkBn = (VulkanBackend*)backend();
    vkBn->recycleUniform(mConstBuffer);
}

ErrorCode VulkanTopKV2::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto outputValue = outputs[0];
    auto outputIndex = outputs[1];

    const int rowSize = input->length(input->dimensions() - 1);
    if (rowSize <= 0) {
        return NO_ERROR;
    }
    const int numRows = input->elementSize() / rowSize;
    const int k = _topKFromOutput(outputValue);
    if (k <= 0) {
        return NO_ERROR;
    }

    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (!_createPipeline(k, input)) {
        return NOT_SUPPORT;
    }

    // Set GPU params
    auto topkParam = reinterpret_cast<TopKV2ConstBuffer*>(mConstBuffer->map());
    topkParam->rowSize = rowSize;
    topkParam->k = k;
    topkParam->numRows = numRows;
    topkParam->pad = 0;
    mConstBuffer->unmap();

    // Bind buffers
    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputValue), 0);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(outputIndex), 1);
    mDescriptorSet->writeBuffer(vkBn->getBuffer(input), 2);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());

#ifdef ENABLE_VULKAN_TIME_PROFILE
    auto* profiler = vkBn->timeProfiler();
    if (nullptr != profiler) {
        VulkanTimeProfileScope scope(profiler, cmdBuffer->get(), mPipelineName.c_str(),
                                     VulkanTimeProfiler::Kind::Shader);
        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), 1, numRows, 1);
        return NO_ERROR;
    }
#endif
    // Dispatch: x=1 (one workgroup scans each row), y=numRows, z=1
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), 1, numRows, 1);

    return NO_ERROR;
}

class VulkanTopKV2Creator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() < 2 || outputs.size() != 2) {
            return nullptr;
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        const int k = _topKFromOutput(outputs[0]);
        if (k <= 0) {
            return nullptr;
        }
        return new VulkanTopKV2(op, backend, inputs[0], outputs[0]);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_TopKV2, new VulkanTopKV2Creator);
    return true;
}();

} // namespace MNN
