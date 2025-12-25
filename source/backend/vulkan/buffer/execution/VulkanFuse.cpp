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
        std::vector<std::tuple<int, void*, size_t>> constStoragePtrs;
        std::vector<std::tuple<int, void*, size_t>> constUniformPtrs;
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
                if (attr->b()) {
                    types[attr->i()] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    constUniformPtrs.emplace_back(std::make_tuple(attr->i(), result, bufferSize));
                } else {
                    types[attr->i()] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    constStoragePtrs.emplace_back(std::make_tuple(attr->i(), result, bufferSize));
                }
                continue;
            }
        }
        auto alignSize = vkBn->device().proty().limits.minMemoryMapAlignment;
        size_t offset = 0;
        std::shared_ptr<VulkanCommandPool::Buffer> cmdbuffer( vkBn->getPool().allocBuffer());
        cmdbuffer->begin(0);
        auto merge = [&](const std::vector<std::tuple<int, void*, size_t>>& constPtrs, VkDescriptorType type) {
            if (constPtrs.empty()) {
                return std::make_tuple(std::vector<std::tuple<int, size_t, size_t>>{}, std::shared_ptr<VulkanBuffer>(nullptr), std::shared_ptr<VulkanBuffer>(nullptr));
            }
            std::vector<std::tuple<int, size_t, size_t>> mConstOffset;
            for (auto& constAttr : constPtrs) {
                auto size = UP_DIV(std::get<2>(constAttr), alignSize) * alignSize;
                mConstOffset.emplace_back(std::make_tuple(std::get<0>(constAttr), size, offset));
                offset += size;
            }
            std::shared_ptr<VulkanBuffer> hostBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, offset, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
            auto ptr = (uint8_t*)hostBuffer->map();
            for (int i=0; i<constPtrs.size(); ++i) {
                ::memcpy(ptr + std::get<2>(mConstOffset[i]), std::get<1>(constPtrs[i]), std::get<2>(constPtrs[i]));
            }
            hostBuffer->unmap();
            std::shared_ptr<VulkanBuffer> vkBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, offset, nullptr, type, VK_SHARING_MODE_EXCLUSIVE, 0));
            VkBufferCopy bufferCopy;
            bufferCopy.size = offset;
            bufferCopy.dstOffset = 0;
            bufferCopy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuffer->buffer(), vkBuffer->buffer(),
                            1, &bufferCopy);
            return std::make_tuple(mConstOffset, vkBuffer, hostBuffer);
        };
        mConstStorageOffset.clear();
        mConstUniformOffset.clear();
        auto uniforms = merge(constUniformPtrs, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        mConstUniformOffset = std::get<0>(uniforms);
        mConstUniformBuffer = std::get<1>(uniforms);
        auto storages = merge(constStoragePtrs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        mConstStorageOffset = std::get<0>(storages);
        mConstStorageBuffer = std::get<1>(storages);
        cmdbuffer->end();
        auto fence = vkBn->getPool().submit(cmdbuffer->get());

        mPipeline = factory->createComputePipeline(data, dataSize, types, std::vector<uint32_t>{});
        mDescriptorSet = mPipeline->createSet();
        fence->wait();
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
        for (auto& iter : mConstStorageOffset) {
            mDescriptorSet->writeBuffer(mConstStorageBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
        }
        for (auto& iter : mConstUniformOffset) {
            mDescriptorSet->writeBuffer(mConstUniformBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
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
    std::shared_ptr<VulkanBuffer> mConstStorageBuffer;
    std::shared_ptr<VulkanBuffer> mConstUniformBuffer;
    // Index, offset, size
    std::vector<std::tuple<int, size_t, size_t>> mConstStorageOffset;
    std::vector<std::tuple<int, size_t, size_t>> mConstUniformOffset;
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
        if(extra->type()->str() == "ExtraConvolution2DPrelu"){
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

