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

template<typename T0, typename T1>
void _copy(const T0* src, T1* dst, size_t size) {
    for (int i=0; i<size; ++i) {
        dst[i] = src[i];
    }
}

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source, size_t offset) {
    auto sourcePtr   = (const float*)source->map(offset);
    ::memcpy(dest->host<float>(), sourcePtr, dest->usize());
    source->unmap();
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest, size_t offset) {
    auto destPtr     = (float*)dest->map(offset);
    ::memcpy(destPtr, source->host<float>(), source->usize());
    dest->unmap();
}

VulkanBackend::VulkanBackend(const VulkanRuntime* runtime, const Backend::Info& info) : Backend(MNN_FORWARD_VULKAN) {
    mRuntime = runtime;
    mDirect = Backend::Info::INDIRECT != info.mode;
    std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(runtime->mBufferPool.get());
    mDynamicBufferPool.resize(2);
    mDynamicBufferPool[0].reset(new EagerBufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));
    mCurrentDynamicBufferPool = mDynamicBufferPool[0].get();

    auto& dev              = device();
    mFence                 = std::make_shared<VulkanFence>(dev);
    if (!mDirect) {
        mCmdBuffer.reset(runtime->mCmdPool->allocBuffer());
    }
    std::string deviceName = dev.proty().deviceName;
    if(deviceName.find("Apple") != std::string::npos){
        mUseAutoTune = false;
    }
    mCmdBufferForCopy.reset(runtime->mCmdPool->allocBuffer());
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
ErrorCode VulkanBackend::onResizeEnd() {
    if (!mDirect) {
        mCmdBuffer->end();
    }
    return NO_ERROR;
}
class VulkanMemRelease : public Backend::MemObj {
public:
    VulkanMemRelease(BufferAllocator* allocator, MemChunk points, int size) {
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
    inline MemChunk points() const {
        return mPoint;
    }
private:
    BufferAllocator* mAllocator;
    MemChunk mPoint;
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
    auto newBuffer = mCurrentDynamicBufferPool->alloc(alignSize, seperate);
    auto mem = new VulkanMemRelease(mCurrentDynamicBufferPool, newBuffer, alignSize);
    MTensor->buffer().device = (uint64_t)(newBuffer.first);
    des->extra.offset = newBuffer.second;
    return mem;
}
bool VulkanBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2 || index >= 2 || index < 0) {
        return false;
    }
    if (mDynamicBufferPool[1].get() == nullptr) {
        std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(mRuntime->mBufferPool.get());
        mDynamicBufferPool[1].reset(new EagerBufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));
    }
    mCurrentDynamicBufferPool = mDynamicBufferPool[index].get();
    return true;
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
    mCurrentDynamicBufferPool->release(false);
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
const VulkanPipelineFactory* VulkanBackend::getPipelineFactory() const {
    return mRuntime->mPipelineFactory.get();
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
void VulkanBackend::copyToGPUBuffer(const void* src, VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset) const {
    _requireHostBuffer(size);
    ::memcpy(mHostBuffer->map(), src, size);
    mHostBuffer->unmap();
    auto cmdbuffer = mCmdBufferForCopy;
    cmdbuffer->begin(0);
    VkBufferCopy bufferCopy;
    bufferCopy.size = size;
    bufferCopy.dstOffset = offset;
    bufferCopy.srcOffset = 0;
    vkCmdCopyBuffer(cmdbuffer->get(), mHostBuffer->buffer(), buffer,
                    1, &bufferCopy);
    cmdbuffer->end();
    pushCommand(cmdbuffer->get());
    _finish();
    mHostBuffer.reset();
}
void VulkanBackend::_requireHostBuffer(size_t size) const {
    _finish();
    if (nullptr == mHostBuffer || mHostBuffer->size() < size) {
        mHostBuffer.reset(new VulkanBuffer(*mRuntime->mMemoryPool, false, size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
    }
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
        auto eleSize = srcTensor->elementSize();
        _requireHostBuffer(eleSize * sizeof(float));
        _copyTensorToBuffer(srcTensor, mHostBuffer.get(), 0);
        auto cmdbuffer = mCmdBufferForCopy;
        cmdbuffer->begin(0);
        VkBufferCopy bufferCopy;
        bufferCopy.size = eleSize * sizeof(float);
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;
        vkCmdCopyBuffer(cmdbuffer->get(), mHostBuffer->buffer(), buffer->buffer(),
                        1, &bufferCopy);
        cmdbuffer->end();
        pushCommand(cmdbuffer->get());
        _finish();
    } else if (dstTensor->host<float>() != nullptr) {
        // gpu->host
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        if (format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(srcTensor->shape(), dstTensor->getType(), nullptr, _convert(TensorUtils::getDescribe(srcTensor)->dimensionFormat)), [dstTensor](void* t) {
                Tensor* temp = (Tensor*)t;
                MNNCPUCopyBuffer(temp, dstTensor);
                delete temp;
            });
            dstTensor = tempTensor.get();
        }
        auto eleSize = dstTensor->elementSize();
        _requireHostBuffer(eleSize * sizeof(float));
        auto buffer = reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId());
        auto offset = TensorUtils::getDescribe(srcTensor)->extra.offset;
        auto cmdbuffer = mCmdBufferForCopy;
        cmdbuffer->begin(0);
        VkBufferCopy bufferCopy;
        bufferCopy.size = eleSize * sizeof(float);
        bufferCopy.dstOffset = 0;
        bufferCopy.srcOffset = offset;
        vkCmdCopyBuffer(cmdbuffer->get(), buffer->buffer(), mHostBuffer->buffer(),
                        1, &bufferCopy);
        cmdbuffer->end();
        pushCommand(cmdbuffer->get());
        _finish();
        _copyBufferToTensor(dstTensor, mHostBuffer.get(), 0);
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
int VulkanBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    _finish();
    return 0;
}

bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

bool VulkanBackend::onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
    if (nullptr == dstInfo) {
        return true;
    }
    auto vkBuffer = getBuffer(tensor);
    auto dst = (MNNVulkanTensorContent*)dstInfo;
    dst->buffer = std::get<0>(vkBuffer);
    dst->offset = std::get<2>(vkBuffer);
    dst->size = std::get<1>(vkBuffer);
    return true;
}

float VulkanBackend::getPipelineTime(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> groupSize){
    std::shared_ptr<VulkanCommandPool::Buffer> cmd;
    cmd.reset(const_cast<VulkanCommandPool::Buffer *>(getPool().allocBuffer()));
    cmd->begin(0);
    mRuntime->mQueryPool->VulkanCmdResetQueryPool(cmd.get()->get());
    mRuntime->mQueryPool->VulkanCmdWriteTimestamp(cmd.get()->get(), 0);
    pipeline->bind(cmd.get()->get(), des->get());
    vkCmdDispatch(cmd.get()->get(), groupSize[0], groupSize[1], groupSize[2]);
    mRuntime->mQueryPool->VulkanCmdWriteTimestamp(cmd.get()->get(), 1);
    cmd->end();
    getPool().submitAndWait(cmd.get()->get());
    float time = mRuntime->mQueryPool->VulkanGetQueryPoolResults();
    return time;
}

std::vector<uint32_t> VulkanBackend::autoTunePipeline(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> gws){
    std::vector<uint32_t> lws(3, 1);
    std::vector<int> groupSize(3, 1);
    std::vector<int> maxGroups(3, 1);
    int maxGroupSize = mRuntime->mDevice->getMaxComputeWorkGroupInvocations();
    mRuntime->mDevice->getMaxComputeWorkGroupSize(maxGroups);
    
    std::vector<uint32_t> lws_prefer(3, 1);
    uint32_t min_cost = UINT_MAX;
    
    while(lws[2] <= gws[2] && lws[2] <= maxGroups[2]) {
        lws[1] = 1;
        while(lws[1] <= gws[1] && lws[1] <= maxGroups[1]) {
            lws[0] = 1;
            while(lws[0] <= gws[0] && lws[0] <= maxGroups[0]) {
                if(lws[0]*lws[1]*lws[2] <= maxGroupSize) {
                    groupSize[0] = UP_DIV(gws[0], lws[0]);
                    groupSize[1] = UP_DIV(gws[1], lws[1]);
                    groupSize[2] = UP_DIV(gws[2], lws[2]);
                    
                    pipeline->changePipeline(lws);
                    int cost_time = (int)getPipelineTime(pipeline, des, groupSize);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                        lws_prefer[2] = lws[2];
                    }
                }
                lws[0]*=2;
            }
            lws[1]*=2;
        }
        lws[2]*=2;
    }
    
    pipeline->changePipeline(lws_prefer);
    return lws_prefer;
}

} // namespace MNN
