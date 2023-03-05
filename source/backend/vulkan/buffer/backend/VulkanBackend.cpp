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
#include "component/VulkanDevice.hpp"
#include "component/VulkanInstance.hpp"
#include "execution/VulkanBasicExecution.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#define MNN_OP_SUPPORT_LOG
//#define MNN_VULKAN_DUMP_MEMORY_USAGE

namespace MNN {

static std::map<OpType, VulkanBackend::Creator*>* gCreator = nullptr;

// Creator
static inline std::map<OpType, VulkanBackend::Creator*>* getCreatorMap() {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, VulkanBackend::Creator*>();
    }
    return gCreator;
}

template<typename T>
void _copyIn(const T* src, float* dst, size_t size) {
    for (int i=0; i<size; ++i) {
        dst[i] = src[i];
    }
}

template<typename T>
void _copyOut(const float* src, T* dst, size_t size) {
    for (int i=0; i<size; ++i) {
        dst[i] = src[i];
    }
}

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source, size_t offset) {
    auto sourcePtr   = (const float*)source->map(offset);
    auto dataType    = dest->getType();
    auto eleSize = dest->elementSize();
    if (dataType == halide_type_of<float>()) {
        ::memcpy(dest->host<float>(), sourcePtr, dest->size());
    }
    else if (dataType == halide_type_of<uint8_t>()) {
        _copyOut(sourcePtr, dest->host<uint8_t>(), eleSize);
    }
    else if (dataType == halide_type_of<int8_t>()) {
        _copyOut(sourcePtr, dest->host<int8_t>(), eleSize);
    }
    else if (dataType == halide_type_of<int32_t>()) {
        _copyOut(sourcePtr, dest->host<int32_t>(), eleSize);
    }
    else if (dataType == halide_type_of<uint32_t>()) {
        _copyOut(sourcePtr, dest->host<uint32_t>(), eleSize);
    } else {
        MNN_PRINT("Don't support typecode = %d, bits = %d\n", dataType.code, dataType.bits);
    }
    source->unmap();
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest, size_t offset) {
    auto destPtr     = (float*)dest->map(offset);
    auto dataType    = source->getType();
    auto eleSize = source->elementSize();
    if (dataType == halide_type_of<float>()) {
        ::memcpy(destPtr, source->host<float>(), source->size());
    }
    else if (dataType == halide_type_of<uint8_t>()) {
        _copyIn(source->host<uint8_t>(), destPtr, eleSize);
    }
    else if (dataType == halide_type_of<int8_t>()) {
        _copyIn(source->host<int8_t>(), destPtr, eleSize);
    }
    else if (dataType == halide_type_of<int32_t>()) {
        _copyIn(source->host<int32_t>(), destPtr, eleSize);
    }
    else if (dataType == halide_type_of<uint32_t>()) {
        _copyIn(source->host<uint32_t>(), destPtr, eleSize);
    } else {
        MNN_PRINT("Don't support typecode = %d, bits = %d\n", dataType.code, dataType.bits);
    }
    dest->unmap();
}

VulkanBackend::VulkanBackend(const VulkanRuntime* runtime, const Backend::Info& info) : Backend(MNN_FORWARD_VULKAN) {
    mRuntime = runtime;
    mDirect = Backend::Info::INDIRECT != info.mode;
    std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(runtime->mBufferPool.get());

    mDynamicBufferPool.reset(new BufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));

    auto& dev              = device();
    mFence                 = std::make_shared<VulkanFence>(dev);
    if (!mDirect) {
        mCmdBuffer.reset(runtime->mCmdPool->allocBuffer());
    }
}

VulkanBackend::~VulkanBackend() {
    /*keep release order*/
    mCmdBuffer = nullptr;
    mCmdBuffers.clear();
    mFence = nullptr;
}
void VulkanBackend::pushCommand(VkCommandBuffer buffer) const {
    mCmdBuffers.emplace_back(buffer);
}

const VulkanPipeline* VulkanBackend::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                 const std::vector<uint32_t>& localSize) const {
    return mRuntime->mPipelineFactory->getPipeline(key, types, localSize);
}

void VulkanBackend::onResizeBegin() {
    if (!mDirect) {
        mCmdBuffer->begin(0);
    }
}
void VulkanBackend::onResizeEnd() {
    if (!mDirect) {
        mCmdBuffer->end();
    }
}
class VulkanMemRelease : public Backend::MemObj {
public:
    VulkanMemRelease(BufferAllocator* allocator, std::pair<void*, int> points, int size) {
        mPoint = std::move(points);
        mAllocator = allocator;
        mSize = size;
    }
    virtual ~ VulkanMemRelease() {
        mAllocator->free(mPoint);
    }
    inline int getSize() const {
        return mSize;
    }
    inline std::pair<void*, int> points() const {
        return mPoint;
    }
private:
    BufferAllocator* mAllocator;
    std::pair<void*, int> mPoint;
    int mSize;
};
VULKAN_TENSOR VulkanBackend::getBuffer(const Tensor* tensor) const {
    auto b = getTensorBuffer(tensor);
    return std::make_tuple(b.first->buffer(), getTensorSize(tensor), b.second);
}

std::pair<const VulkanBuffer*, size_t> VulkanBackend::getTensorBuffer(const Tensor* tensor) const {
    auto mem = (VulkanBuffer*)(tensor->deviceId());
    MNN_ASSERT(nullptr != mem);
    return std::make_pair(mem, TensorUtils::getDescribe(tensor)->extra.offset);
}

size_t VulkanBackend::getTensorSize(const Tensor* tensor) const {
    auto elementSize = tensor->elementSize();
    auto alignSize = UP_DIV(elementSize, 4) * 4 * sizeof(float);
    return alignSize;
}

Backend::MemObj* VulkanBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    //FUNC_PRINT_ALL(tensor, p);
    auto alignSize = getTensorSize(tensor);
    auto MTensor     = const_cast<Tensor*>(tensor);
    auto des = TensorUtils::getDescribe(tensor);
    if (Backend::STATIC == storageType) {
        auto newBuffer = mRuntime->mBufferPool->alloc(alignSize);
        auto mem = new VulkanMemRelease(mRuntime->mBufferPool.get(), newBuffer, alignSize);
        MTensor->buffer().device = (uint64_t)(newBuffer.first);
        des->extra.offset = newBuffer.second;
        return mem;
    }
    bool seperate  = storageType == Backend::DYNAMIC_SEPERATE;
    auto newBuffer = mDynamicBufferPool->alloc(alignSize, seperate);
    auto mem = new VulkanMemRelease(mDynamicBufferPool.get(), newBuffer, alignSize);
    MTensor->buffer().device = (uint64_t)(newBuffer.first);
    des->extra.offset = newBuffer.second;
    return mem;
}

std::shared_ptr<VulkanBuffer> VulkanBackend::allocUniform(const void* src, int size) {
    auto rt = const_cast<VulkanRuntime*>(mRuntime);
    return rt->allocUniform(src, size);
}
void VulkanBackend::recycleUniform(std::shared_ptr<VulkanBuffer> buffer) {
    auto rt = const_cast<VulkanRuntime*>(mRuntime);
    rt->recycleUniform(buffer);
}

bool VulkanBackend::onClearBuffer() {
    mDynamicBufferPool->release(false);
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
    std::shared_ptr<VulkanBasicExecution> originExecution ((VulkanBasicExecution*)iter->second->onCreate(inputs, outputs, op, this));
    if (nullptr == originExecution) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Special case\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }
    if (mDirect) {
        return new VulkanBasicExecutionDirect(originExecution);
    }
    return new VulkanBasicExecutionInDirect(originExecution);
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

static Tensor::DimensionType _convert(MNN_DATA_FORMAT format) {
    switch (format) {
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        default:
            break;
    }
    return Tensor::CAFFE;
}
void VulkanBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef MNN_VULKAN_DEBUG
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
    std::shared_ptr<Tensor> tempTensor;
    if (srcTensor->host<float>() != nullptr) {
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        auto buffer = reinterpret_cast<VulkanBuffer*>(dstTensor->deviceId());
        auto offset = TensorUtils::getDescribe(dstTensor)->extra.offset;
        // host->gpu
        if(format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(dstTensor->shape(), dstTensor->getType(), nullptr, _convert(format)));
            MNNCPUCopyBuffer(srcTensor, tempTensor.get());
            srcTensor = tempTensor.get();
        }
        _copyTensorToBuffer(srcTensor, buffer, offset);
    } else if (dstTensor->host<float>() != nullptr) {
        // gpu->host
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        if (format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(srcTensor->shape(), dstTensor->getType(), nullptr, _convert(TensorUtils::getDescribe(srcTensor)->dimensionFormat)), [dstTensor](void* t) {
                Tensor* temp = (Tensor*)t;
                MNNCPUCopyBuffer(temp, dstTensor);
            });
            dstTensor = tempTensor.get();
        }
        auto buffer = reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId());
        auto offset = TensorUtils::getDescribe(srcTensor)->extra.offset;

        _copyBufferToTensor(dstTensor, buffer, offset);
    } else if (srcTensor->deviceId() != 0 && dstTensor->deviceId() != 0) {
        // gpu->gpu
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        // host->gpu
        MNN_ASSERT(format == TensorUtils::getDescribe(srcTensor)->dimensionFormat);
        std::shared_ptr<VulkanCommandPool::Buffer> buffer( mRuntime->mCmdPool->allocBuffer());
        buffer->begin(0);
        VkBufferCopy bufferCopy;
        bufferCopy.size = srcTensor->elementSize() * sizeof(float);
        bufferCopy.dstOffset = TensorUtils::getDescribe(dstTensor)->extra.offset;
        bufferCopy.srcOffset = TensorUtils::getDescribe(srcTensor)->extra.offset;
        vkCmdCopyBuffer(buffer->get(), reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId())->buffer(), reinterpret_cast<VulkanBuffer*>(dstTensor->deviceId())->buffer(),
                        1, &bufferCopy);
        buffer->end();
        pushCommand(buffer->get());
        _finish();
    }
}

bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

} // namespace MNN
