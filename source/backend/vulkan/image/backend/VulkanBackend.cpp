//
//  VulkanBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBackend.hpp"
#include <mutex>
#include "core/Execution.hpp"
#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "component/VulkanDevice.hpp"
#include "execution/VulkanImageConverter.hpp"
#include "component/VulkanInstance.hpp"
#include "execution/VulkanBasicExecution.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
//#define MNN_OP_SUPPORT_LOG
//#define MNN_VULKAN_DUMP_MEMORY_USAGE
#define MNN_VULKAN_MAX_CACHE_CONVSIZE 50
namespace MNN {

static std::map<OpType, VulkanBackend::Creator*>* gCreator = nullptr;

// Creator
static inline std::map<OpType, VulkanBackend::Creator*>* getCreatorMap() {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, VulkanBackend::Creator*>();
    }
    return gCreator;
}

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source) {
    auto sourcePtr   = source->map();
    auto dataType    = dest->getType();
    //TODO: Support other kind of dataType
    MNN_ASSERT(dataType.bits == 32);
    ::memcpy(dest->host<float>(), sourcePtr, dest->size());
    source->unmap();
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest) {
    auto destPtr     = dest->map();
    auto dataType    = source->getType();
    //TODO: Support other kind of dataType
    MNN_ASSERT(dataType.bits == 32);
    ::memcpy(destPtr, source->host<float>(), source->size());
    dest->unmap();
}

VulkanBackend::VulkanBackend(const VulkanRuntime* runtime, const Backend::Info& info) : Backend(MNN_FORWARD_VULKAN) {
    mRuntime = runtime;
    mDirect = Backend::Info::INDIRECT != info.mode;
    mDynamicMemoryPool.reset(new VulkanMemoryPool(runtime->mMemoryPool.get()));

    auto& dev              = device();
    mFence                 = std::make_shared<VulkanFence>(dev);
    if (!mDirect) {
        mCmdBuffer.reset(runtime->mCmdPool->allocBuffer());
    }
    mInitBuffer.reset(runtime->mCmdPool->allocBuffer());
}

VulkanBackend::~VulkanBackend() {
    /*keep release order*/
    mCmdBuffer = nullptr;

    mAllBuffers.clear();
    mHostBuffer = nullptr;
    mCmdBuffers.clear();
    mFence = nullptr;
    mConverters.clear();
}
void VulkanBackend::pushCommand(VkCommandBuffer buffer) const {
    mCmdBuffers.emplace_back(buffer);
//    _finish();
}

const VulkanPipeline* VulkanBackend::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                 const std::vector<uint32_t>& localSize) const {
    return mRuntime->mPipelineFactory->getPipeline(key, types, localSize);
}

bool VulkanBackend::_supportImageSize(const Tensor* MTensor) {
    if (MTensor->getType().code != halide_type_float) {
        return false;
    }
    auto format = TensorUtils::getDescribe(MTensor)->dimensionFormat;
    if (format != MNN_DATA_FORMAT_NC4HW4) {
        return true;
    }
    auto nhwc = VulkanTensor::tensorShapeFormat(MTensor);
    auto width = UP_DIV(nhwc[3], 4) * nhwc[2];
    auto height = nhwc[0] * nhwc[1];
    int unit = device().proty().limits.maxImageDimension2D;
    if (width > unit || height > unit) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_PRINT("Not support size: %d - %d\n", width, height);
#endif
        return false;
    }
    return true;
}
void VulkanBackend::onResizeBegin() {
    mInitBuffer->begin(0);
    if (!mDirect) {
        mCmdBuffer->begin(0);
    }
}
ErrorCode VulkanBackend::onResizeEnd() {
    if (!mDirect) {
        mCmdBuffer->end();
    }
    mInitBuffer->end();
    mCmdBuffers.emplace_back(mInitBuffer->get());
    _finish();
    return NO_ERROR;
}
class VulkanMemRelease : public Backend::MemObj {
public:
    VulkanMemRelease(std::shared_ptr<VulkanTensor> t) {
        mTensor = t;
    }
    virtual ~ VulkanMemRelease() {
        mTensor->release();
    }
private:
    std::shared_ptr<VulkanTensor> mTensor;
};

static VkFormat _getFormat(halide_type_t type) {
    switch (type.code) {
        case halide_type_float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case halide_type_int: {
            if (8 == type.bits) {
                return VK_FORMAT_R8G8B8A8_SINT;
            } else if (type.bits == 16) {
                return VK_FORMAT_R16G16B16A16_SINT;
            }
            return VK_FORMAT_R32G32B32A32_SINT;
        }
        case halide_type_uint: {
            if (8 == type.bits) {
                return VK_FORMAT_R8G8B8A8_UINT;
            } else if (type.bits == 16) {
                return VK_FORMAT_R16G16B16A16_UINT;
            }
            return VK_FORMAT_R32G32B32A32_UINT;
        }
        default:
            break;
    }
    return VK_FORMAT_R32G32B32A32_SFLOAT;
}

Backend::MemObj* VulkanBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    //FUNC_PRINT_ALL(tensor, p);

    auto MTensor     = const_cast<Tensor*>(tensor);
    auto format = _getFormat(tensor->getType());
    if (Backend::STATIC == storageType) {
        auto newBuffer           = std::make_shared<VulkanTensor>(MTensor, format, getMemoryPool(), device().proty().limits);
        MTensor->buffer().device = (uint64_t)(newBuffer.get());
        return new VulkanMemRelease(newBuffer);
    }
    bool separate  = storageType == Backend::DYNAMIC_SEPERATE;
    auto newBuffer = std::make_shared<VulkanTensor>(MTensor, format, getDynamicMemoryPool(), device().proty().limits, separate);
    MTensor->buffer().device = (uint64_t)(newBuffer.get());
    mAllBuffers.insert(std::make_pair(MTensor->buffer().device, newBuffer));
    return new VulkanMemRelease(newBuffer);;
}

bool VulkanBackend::onClearBuffer() {
    mAllBuffers.clear();
    mConverters.clear();
    mDynamicMemoryPool->clear();
    return true;
}
Execution* VulkanBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
    auto creator = getCreatorMap();
    auto iter    = creator->find(op->type());
    std::string name = "";
    if (nullptr != op->name()) {
        name = op->name()->str();
    }
    if (iter == creator->end()) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_PRINT("Vulkan don't support %d, %s: %s\n", op->type(), EnumNameOpType(op->type()),
                name.c_str());
#endif
        return nullptr;
    }
    bool valid = true;
    for (int i=0; i<inputs.size(); ++i) {
        if (!OpCommonUtils::opNeedContent(op, i)) {
            continue;
        }
        auto t = inputs[i];
        if (!_supportImageSize(t)) {
            valid = false;
            break;
        }
    }
    for (auto t : outputs) {
        if (!_supportImageSize(t)) {
            valid = false;
            break;
        }
    }
    if (!valid) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Tensor not support\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }
    auto originExecution = (VulkanBasicExecution*)iter->second->onCreate(inputs, outputs, op, this);
    if (nullptr == originExecution) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Special case\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }
    if (mDirect) {
        return new VulkanBasicExecutionDirect(std::shared_ptr<VulkanBasicExecution>(originExecution));
    }
    return new VulkanBasicExecutionInDirect(std::shared_ptr<VulkanBasicExecution>(originExecution));
}

void VulkanBackend::onExecuteBegin() const {
    if (!mDirect) {
        mCmdBuffers.push_back(mCmdBuffer->get());
    }
    // FUNC_PRINT_ALL(mDynamicMemoryPool->computeSize(), f);
}
void VulkanBackend::onExecuteEnd() const {
    _finish();
}
void VulkanBackend::_finish() const {
    if (mCmdBuffers.empty()) {
        return;
    }
    VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                /* .pNext                = */ nullptr,
                                /* .waitSemaphoreCount   = */ 0,
                                /* .pWaitSemaphores      = */ nullptr,
                                /* .pWaitDstStageMask    = */ nullptr,
                                /* .commandBufferCount   = */ (uint32_t)mCmdBuffers.size(),
                                /* .pCommandBuffers      = */ mCmdBuffers.data(),
                                /* .signalSemaphoreCount = */ 0,
                                /* .pSignalSemaphores    = */ nullptr};
    auto fenceReal           = mFence->get();
    mFence->reset();
    CALL_VK(vkQueueSubmit(device().acquireDefaultDevQueue(), 1, &submit_info, fenceReal));

    auto res = mFence->wait();
    MNN_VK_CHECK(res);
    mCmdBuffers.clear();
}

const VulkanDevice& VulkanBackend::device() const {
    return (* mRuntime->mDevice);
}

void VulkanBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef MNN_VULKAN_DEBUG
#ifdef MNN_VULKAN_DEBUG_COPY
    AUTOTIME;
    MNN_PRINT("Src: ");
    for (int i=0; i<srcTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", srcTensor->length(i));
    }
    MNN_PRINT("\n");
    MNN_PRINT("Dst: ");
    for (int i=0; i<dstTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", dstTensor->length(i));
    }
    MNN_PRINT("\n");
#endif
#endif
    if (srcTensor->host<float>() != nullptr) {
        MNN_ASSERT(nullptr == dstTensor->host<float>());
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        std::shared_ptr<Tensor> tempTensor(new Tensor);
        TensorUtils::copyShape(dstTensor, tempTensor.get(), true);
        tempTensor->buffer().type = dstTensor->buffer().type;
        auto size = VulkanTensor::getAlignSize(tempTensor.get()) * sizeof(float);
        // host->gpu
        _allocHostBuffer(size);
        tempTensor->buffer().host = (uint8_t*)mHostBuffer->map();
        MNNCPUCopyBuffer(srcTensor, tempTensor.get());
        mHostBuffer->unmap();
        auto key    = std::make_tuple(TensorUtils::getDescribe(dstTensor), true, format);
        auto iter   = mConverters.find(key);
        if (iter == mConverters.end()) {
            if (mConverters.size() > MNN_VULKAN_MAX_CACHE_CONVSIZE) {
                mConverters.clear();
            }
            auto converter = std::make_shared<VulkanImageConverter>(this);
            std::shared_ptr<VulkanCommandPool::Buffer> convertorBuffer(
                                                                       const_cast<VulkanCommandPool::Buffer*>(getPool().allocBuffer()));
            convertorBuffer->begin(0);
            auto vkTensor = reinterpret_cast<VulkanTensor*>(dstTensor->deviceId());
            for (int i=0; i<vkTensor->imageSize(); ++i) {
                vkTensor->image(i)->barrierWrite(convertorBuffer->get());
            }
            converter->encodeBufferToTensor(mHostBuffer->buffer(), dstTensor, mHostBuffer->size(), 0,
                                            format,
                                            convertorBuffer.get());
            for (int i=0; i<vkTensor->imageSize(); ++i) {
                vkTensor->image(i)->barrierRead(convertorBuffer->get());
            }
            convertorBuffer->end();
            mConverters.insert(std::make_pair(key, std::make_pair(converter, convertorBuffer)));
            iter = mConverters.find(key);
        }
        mCmdBuffers.push_back(iter->second.second->get());
    } else if (dstTensor->host<void>() != nullptr) {
        // gpu->host
        auto size = VulkanTensor::getAlignSize(srcTensor) * sizeof(float);
        _finish();
        _allocHostBuffer(size);
        auto format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
        auto key    = std::make_tuple(TensorUtils::getDescribe(srcTensor), false, format);

        auto iter = mConverters.find(key);
        if (iter == mConverters.end()) {
            if (mConverters.size() > MNN_VULKAN_MAX_CACHE_CONVSIZE) {
                mConverters.clear();
            }
            auto converter = std::make_shared<VulkanImageConverter>(this);
            std::shared_ptr<VulkanCommandPool::Buffer> convertorBuffer(
                                                                       const_cast<VulkanCommandPool::Buffer*>(getPool().allocBuffer()));
            convertorBuffer->begin(0);
            converter->encodeTensorToBuffer(srcTensor, mHostBuffer->buffer(), mHostBuffer->size(), 0,
                                            format,
                                            convertorBuffer.get());
            convertorBuffer->end();
            mConverters.insert(std::make_pair(key, std::make_pair(converter, convertorBuffer)));
            iter = mConverters.find(key);
        }
        mCmdBuffers.push_back(iter->second.second->get());
        _finish();
        std::shared_ptr<Tensor> tempTensor(new Tensor);
        TensorUtils::copyShape(srcTensor, tempTensor.get(), true);
        tempTensor->buffer().type = srcTensor->buffer().type;
        tempTensor->buffer().host = (uint8_t*)mHostBuffer->map();
        MNNCPUCopyBuffer(tempTensor.get(), dstTensor);
        mHostBuffer->unmap();
    } else {
        // Device to device
        _finish();
        auto srcVkTensor = reinterpret_cast<VulkanTensor*>(srcTensor->deviceId());
        auto dstVkTensor = reinterpret_cast<VulkanTensor*>(dstTensor->deviceId());
        MNN_ASSERT(TensorUtils::getDescribe(srcTensor)->dimensionFormat == TensorUtils::getDescribe(dstTensor)->dimensionFormat);
        if (nullptr == srcVkTensor || nullptr == dstVkTensor) {
            return;
        }

        MNN_ASSERT(srcVkTensor->imageSize() == dstVkTensor->imageSize());
        int n = srcVkTensor->imageSize();
        auto types = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        // get pipeline
        auto unaryPipeline = getPipeline("glsl_unaryImage_comp", types);
        struct Param {
            ivec4 size;
            ivec4 srcOffset;
            ivec4 srcStride;
            ivec4 dstOffset;
            ivec4 dstStride;
        };
        std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mDesSet(srcVkTensor->imageSize());
        auto needSize = sizeof(Param);
        if (needSize < proty().limits.nonCoherentAtomSize) {
            needSize = proty().limits.nonCoherentAtomSize;
        }
        auto mParam = std::make_shared<VulkanBuffer>(getMemoryPool(), false, needSize * srcVkTensor->imageSize(), nullptr,
                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::shared_ptr<VulkanCommandPool::Buffer> cmdBuffer(mRuntime->mCmdPool->allocBuffer());
        cmdBuffer->begin(0);
        auto paramOrigin = (Param*)mParam->map();
        for (int n=0; n<srcVkTensor->imageSize(); ++n) {
            auto paramPtr = (Param*)((uint8_t*)paramOrigin + n * needSize);
            mDesSet[n].reset(unaryPipeline->createSet());
            auto inputT = srcVkTensor->image(n);
            auto outputT = dstVkTensor->image(n);
            auto totalSize = inputT->depth() * inputT->height() * inputT->width();
            paramPtr->size[0] = inputT->depth() * inputT->height() * inputT->width();
            paramPtr->size[1] = inputT->depth();
            paramPtr->size[2] = inputT->height();
            paramPtr->size[3] = inputT->width();
            paramPtr->dstOffset[0] = 0;
            paramPtr->dstOffset[1] = 0;
            paramPtr->srcOffset[0] = 0;
            paramPtr->srcOffset[1] = 0;
            paramPtr->dstStride[0] = 1;
            paramPtr->dstStride[1] = 1;
            paramPtr->srcStride[0] = 1;
            paramPtr->srcStride[1] = 1;
            inputT->barrierRead(cmdBuffer->get());
            outputT->barrierWrite(cmdBuffer->get());
            mDesSet[n]->writeImage(outputT->view(), getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
            mDesSet[n]->writeImage(inputT->view(), getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
            mDesSet[n]->writeBuffer(mParam->buffer(), 2, sizeof(Param), n * needSize);
            unaryPipeline->bind(cmdBuffer->get(), mDesSet[n]->get());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 256), 1, 1);
            outputT->barrierRead(cmdBuffer->get());
        }
        mParam->unmap();
        cmdBuffer->end();
        mCmdBuffers.push_back(cmdBuffer->get());
        _finish();
    }
}

void VulkanBackend::_allocHostBuffer(size_t size) const {
    if (mHostBuffer.get() == nullptr || mHostBuffer->size() < size) {
        mHostBuffer.reset(new VulkanBuffer(getMemoryPool(), false, size, nullptr,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
        mConverters.clear();
    }
}
bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

void VulkanBackend::copyBufferToImage(const VulkanBuffer* buffer, const VulkanImage* image, VkImageLayout finalLayout) const {
    std::vector<int> dimVector = image->dims();
    const VulkanPipeline* transformPipeline = nullptr;
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    int localX = 16;
    int localY = 16;
    int localZ = 1;
    switch (dimVector.size()) {
        case 1:
            transformPipeline = getPipeline("glsl_buffer2Image1D_comp",
                                            /*glsl_buffer2Image1D_comp, glsl_buffer2Image1D_comp_len,*/ types);
            localX            = 256;
            localY            = 1;
            break;
        case 2:
            transformPipeline = getPipeline("glsl_buffer2Image2D_comp",
                                            /*glsl_buffer2Image2D_comp, glsl_buffer2Image2D_comp_len,*/ types);
            break;
        case 3:
            transformPipeline = getPipeline("glsl_buffer2Image3D_comp",
                                            /*glsl_buffer2Image3D_comp, glsl_buffer2Image3D_comp_len,*/ types);
            break;
        default:
            break;
    }

    std::unique_ptr<VulkanLayout::DescriptorSet> sets(transformPipeline->createSet());
    auto constBuffer = std::make_shared<VulkanBuffer>(getMemoryPool(), false, dimVector.size() * sizeof(int),
                                                      dimVector.data(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    sets->writeImage(image->view(), mRuntime->mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    sets->writeBuffer(buffer->buffer(), 1, buffer->size());
    sets->writeBuffer(constBuffer->buffer(), 2, constBuffer->size());

    std::unique_ptr<VulkanCommandPool::Buffer> cmdbuffer(
        const_cast<VulkanCommandPool::Buffer*>(mRuntime->mCmdPool->allocBuffer()));
    cmdbuffer->begin(0);
    image->barrierWrite(cmdbuffer->get());
    transformPipeline->bind(cmdbuffer->get(), sets->get());
    vkCmdDispatch(cmdbuffer->get(), UP_DIV(image->width(), localX), UP_DIV(image->height(), localY),
                  UP_DIV(image->depth(), localZ));
    image->barrierRead(cmdbuffer->get());
    cmdbuffer->end();
    mRuntime->mCmdPool->submitAndWait(cmdbuffer->get());
}


} // namespace MNN
