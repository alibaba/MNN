//
//  Vulkan.cpp
//  MNN
//
//  Created by MNN on 2023/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
struct ConstBuffer {
    ivec4 inShape;  // inW, inH
};

class VulkanFuse : public VulkanBasicExecution {
public:
    VulkanFuse(const Extra* extra, Backend* bn, int inputSize, int outputSize) : VulkanBasicExecution(bn) {
        auto vkBn = static_cast<VulkanBackend*>(bn);
        auto factory = vkBn->getPipelineFactory();
        mOutputBinding.resize(outputSize);
        mInputBinding.resize(inputSize);
        mGroupSize.resize(3);
        mGlobalSize.resize(3);
        // Find shader
        const uint8_t* data = nullptr;
        size_t dataSize = 0;
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "spirv") {
                data = (uint8_t*)attr->tensor()->int8s()->data();
                dataSize = attr->tensor()->int8s()->size();
                break;
            }
        }
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "global_size") {
                // Use Auto set group size
                auto ptr = attr->tensor()->int32s()->data();
                mGlobalSize[0] = ptr[0];
                mGlobalSize[1] = ptr[1];
                mGlobalSize[2] = ptr[2];
                mNeedAutoTuning = true;
                break;
            }
        }
        // If Has group_size, can't auto tuning
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "group_size") {
                auto ptr = attr->tensor()->int32s()->data();
                mGroupSize[0] = ptr[0];
                mGroupSize[1] = ptr[1];
                mGroupSize[2] = ptr[2];
                mNeedAutoTuning = false;
                break;
            }
        }

        std::vector<VkDescriptorType> types;
        int maxIndex = -1;
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "input") {
                maxIndex = ALIMAX(maxIndex, attr->i());
            } else if (attr->key()->str() == "const") {
                maxIndex = ALIMAX(maxIndex, attr->i());
            }
        }
        types.resize(maxIndex+1);
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "input") {
                auto list = attr->list()->i()->data();
                if (list[1] >= 0) {
                    if (0 == list[0]) {
                        mInputBinding[list[1]] = attr->i();
                    } else {
                        mOutputBinding[list[1]] = attr->i();
                    }
                }
                if (attr->b()) {
                    types[attr->i()] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                } else {
                    types[attr->i()] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                }
                continue;
            }
            if (attr->key()->str() == "const") {
                auto usageBit = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                if (attr->b()) {
                    types[attr->i()] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                } else {
                    usageBit = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                    types[attr->i()] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                }
                auto b = attr->tensor();
                void* result = nullptr;
                size_t bufferSize = 0;
                switch (b->dataType()) {
                    case DataType_DT_FLOAT:
                        result = (void*)b->float32s()->Data();
                        bufferSize = b->float32s()->size() * sizeof(float);
                        break;
                    case DataType_DT_INT32:
                        result = (void*)b->int32s()->Data();
                        bufferSize = b->int32s()->size() * sizeof(float);
                        break;
                    default:
                        MNN_ASSERT(false);
                        break;
                }
                std::shared_ptr<VulkanBuffer> vkBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, bufferSize, nullptr, usageBit, VK_SHARING_MODE_EXCLUSIVE, 0));
                vkBn->copyToGPUBuffer(result, vkBuffer->buffer(), bufferSize, 0);
                mConstIndides.emplace_back(std::make_pair(attr->i(), vkBuffer));
                continue;
            }
        }
        mPipeline = factory->createComputePipeline(data, dataSize, types, std::vector<uint32_t>{});
        mDescriptorSet = mPipeline->createSet();
    }
    virtual ~VulkanFuse() {
        // Remove set firstly before destroy pipeline
        mDescriptorSet = nullptr;
    }
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        auto vkBn = static_cast<VulkanBackend*>(backend());
        for (int i=0; i<inputs.size(); ++i) {
            mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[i]), mInputBinding[i]);
        }
        for (int i=0; i<outputs.size(); ++i) {
            mDescriptorSet->writeBuffer(vkBn->getBuffer(outputs[i]), mOutputBinding[i]);
        }
        for (auto& iter : mConstIndides) {
            mDescriptorSet->writeBuffer(iter.second->buffer(), iter.first, iter.second->size());
        }
        if (mNeedAutoTuning) {
            auto localSize = vkBn->autoTunePipeline(mPipeline.get(), mDescriptorSet, mGlobalSize);
            mPipeline->changePipeline(localSize);
            mGroupSize[0] = UP_DIV(mGlobalSize[0], localSize[0]);
            mGroupSize[1] = UP_DIV(mGlobalSize[1], localSize[1]);
            mGroupSize[2] = UP_DIV(mGlobalSize[2], localSize[2]);
        }
        mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), mGroupSize[0], mGroupSize[1], mGroupSize[2]);
        return NO_ERROR;
    }
private:
    std::vector<int> mGroupSize;
    std::vector<int> mGlobalSize;
    std::vector<int> mInputBinding;
    std::vector<int> mOutputBinding;
    std::vector<std::pair<int, std::shared_ptr<VulkanBuffer>>> mConstIndides;
    SharedPtr<VulkanPipeline> mPipeline;
    SharedPtr<VulkanLayout::DescriptorSet> mDescriptorSet;
    bool mNeedAutoTuning = false;
};

class VulkanFuseCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto extra = op->main_as_Extra();
        if (nullptr == extra) {
            return nullptr;
        }
        if (nullptr == extra->attr()) {
            return nullptr;
        }
        return new VulkanFuse(extra, backend, (int)inputs.size(), (int)outputs.size());
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Extra, new VulkanFuseCreator);
    return true;
}();

} // namespace MNN

