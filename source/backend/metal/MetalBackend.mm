//
//  MetalBackend.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalBackend.hpp"
#define MNN_METAL
#import <MNN/MNNSharedContext.h>
#define METAL_CONST_BUFFER_LIMIT 128
#if MNN_METAL_ENABLED
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/TensorUtils.hpp"
#include "MetalCache_generated.h"
int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor) {
    if (nullptr == content || nullptr == tensor) {
        return 0;
    }
    auto t = (MNN::Tensor*)tensor;
    auto des = MNN::TensorUtils::getDescribe(t);
    content->buffer = ((MNN::MetalRuntimeAllocator::MetalBufferAlloc*)t->deviceId())->getBuffer();
    content->texture = nil;
    content->offset = des->extra.offset;
    return 0;
}

namespace MNN {

static void _MetalApplyTensor(uint8_t* host, size_t offset, Tensor* t) {
    // ptr of MetalBufferAlloc
    t->buffer().device = (uint64_t)host;
    auto des = TensorUtils::getDescribe(t);
    des->extra.offset = offset;
}
static BufferAllocator* _createBufferAllocator(const Runtime* runtime, BufferAllocator* origin, bool secondResize) {
    if (runtime->getAllocatorType() == Runtime::Allocator_Defer && secondResize) {
        return new DeferBufferAllocator(BufferAllocator::Allocator::createRecurse(origin), 1024, _MetalApplyTensor);
    }
    return new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(origin), 1024);
}
struct TunedInfo {
    std::vector<std::unique_ptr<MetalCache::OpInfoT>> mInfos;
};

void registerMetalOps();
#ifdef MNN_SUPPORT_RENDER
extern void registerMetalRenderOps();
#endif

static inline std::map<OpType, MetalBackend::Creator *> *getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, MetalBackend::Creator *> *ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, MetalBackend::Creator *>; });
    return ret;
}

void MetalBackend::addCreator(OpType t, Creator *c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
    }
    map->insert(std::make_pair(t, c));
}

MetalBackend::MetalBackend(std::shared_ptr<EagerBufferAllocator> staticMem, const MetalRuntime* runtime, bool usefp16AsFp32) : Backend(MNN_FORWARD_METAL),
    mEmptyMem(nil)
    {
    mRuntime = runtime;
    auto ctx = (__bridge MNNMetalContext *)runtime->context();
    mBufferPool.reset(_createBufferAllocator(runtime, staticMem.get(), false));
    mCurrentAllocator = mBufferPool.get();
    mStaticBufferPool = staticMem;
    mShapeH2D = getConstBuffer(4 * sizeof(int));
    mShapeD2H = getConstBuffer(4 * sizeof(int));
    mUseFloatAsFp16 = usefp16AsFp32;
    mIsIphone = ctx.isIphone;
    if (runtime->getCommandQueue() == nil) {
        // one command queue can create only a few command buffer, so let each backend own a command queue
        _commandQueue = [[ctx device] newCommandQueue];
        mSupportDeferEncode = true;
    } else {
        // otherwise forbid defer encode optimize
        _commandQueue = runtime->getCommandQueue();
        mSupportDeferEncode = false;
    }
    _commandBuffer = nil;
    _commandBuffer_net = nil;
    _waiting = nil;
}
MetalBackend::~MetalBackend() {
    flushEncoder();
}

id<MTLComputeCommandEncoder> MetalBackend::encoder_net() const {
    id<MTLComputeCommandEncoder> result = [getCommandBufferForNet() computeCommandEncoder];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    result.label = nil;
#endif
    return result;
}

void *MetalBackend::context() const {
    return mRuntime->context();
}

class MetalMemRelease : public Backend::MemObj {
public:
    MetalMemRelease(MemChunk buffer, BufferAllocator* allocator) {
        mBuffer = buffer;
        mAllocator = allocator;
    }
    virtual ~ MetalMemRelease() {
        mAllocator->free(mBuffer);
    }
    MemChunk chunk() override {
        return mBuffer;
    }
private:
    MemChunk mBuffer;
    BufferAllocator* mAllocator;
};
size_t MetalBackend::getTensorSizeInBytes(const Tensor* tensor) const {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    size_t size;
    if (MNN_DATA_FORMAT_NC4HW4 == format && tensor->dimensions() >= 2) {
        int width = 1;
        int height = 1;
        int batch    = tensor->length(0);
        int channel  = tensor->length(1);
        if (tensor->dimensions() >= 3) {
            height = tensor->length(2);
        }
        for (int i=3; i<tensor->dimensions(); ++i) {
            width *= tensor->length(i);
        }
        int alignC = ROUND_UP(channel, 4);
        int hR = ROUND_UP(height, 4) - height;
        // width parallel 4, may exceed 3 elements
        int wR = ROUND_UP(width + 3, 4) - width;
        int bhw = batch * width * height;
        int bhwR = UP_DIV(bhw, 16) * 16 - bhw;
        int extraPadding = ALIMAX(bhwR, (hR * width + wR));
        size = batch * alignC * width * height;
        size = size + extraPadding * 4;
    } else {
        size = 1;
        for (int i=0; i<tensor->dimensions(); ++i) {
            size *= tensor->length(i);
        }
        size = ROUND_UP(size, 4);
    }
    if (0 == size) {
        return 0;
    }
    // use metal_float when meets float
    if (halide_type_float == tensor->buffer().type.code && tensor->buffer().type.bits == 32 && mUseFloatAsFp16) {
        size *= 2;
    } else {
        size *= tensor->getType().bytes();
    }
    size_t align = 4 * sizeof(int);
    size = ROUND_UP(size, align);
    return size;
}

Backend::MemObj* MetalBackend::onAcquire(const Tensor *_tensor, StorageType storageType) {
    auto tensor  = const_cast<Tensor *>(_tensor);
    size_t size = getTensorSizeInBytes(_tensor);
    if (0 == size) {
        return nullptr;
    }
    // reuse if possible
    MemChunk buffer;
    BufferAllocator* allocator = nullptr;
    switch (storageType) {
        case Backend::STATIC: {
            buffer = mStaticBufferPool->alloc(size, false);
            allocator = mStaticBufferPool.get();
        } break;
        case Backend::DYNAMIC: {
            buffer = mCurrentAllocator->alloc(size, false);
            allocator = mCurrentAllocator;
        } break;
        case Backend::DYNAMIC_SEPERATE: {
            buffer = mCurrentAllocator->alloc(size, true);
            allocator = mCurrentAllocator;
        } break;
    }
    if (storageType == Backend::STATIC) {
        if(nullptr == buffer.first) {
            MNN_ERROR("onAcquireBuffer error!\n");
            return nullptr;
        }
    } else {
        buffer.attach(tensor);
    }
    if (nullptr == buffer.first) {
        _MetalApplyTensor((uint8_t*)(&mEmptyMem), 0, (Tensor*)_tensor);
    } else {
        _MetalApplyTensor((uint8_t*)buffer.first, buffer.second, (Tensor*)_tensor);
    }
    return new MetalMemRelease(buffer, allocator);
}

bool MetalBackend::onClearBuffer() {
    mCurrentAllocator->release(true);
    return true;
}

Execution *MetalBackend::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                  const Op *op) {
    auto map  = getCreatorMap();
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        mSupportDeferEncode = false;
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type [%s], %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type [%s]\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, op, this, outputs);
    if (NULL == exe) {
        mSupportDeferEncode = false;
        MNN_PRINT("The Creator Don't support type [%s], %s\n", MNN::EnumNameOpType(op->type()), op->name() ? op->name()->c_str() : "");
        return NULL;
    }
    return exe;
}
void MetalBackend::flushEncoder() const {
    if (nil != mComputeEncoder) {
        [mComputeEncoder endEncoding];
        mComputeEncoder = nil;
    }
}

void MetalBackend::onExecuteBegin() const {
    mEncoderCount = 0;
}
void MetalBackend::onExecuteEnd() const {
    flushEncoder();
    commit_net();

    if(mFrameEncodeCache) {
        // Prepare for next execute
        for(auto opEncoder : mOpEncoders) {
            opEncoder();
        }
        mOpEncoderSet = true;
    }
}
BufferAllocator* MetalBackend::getBufferPool() const {
    return mCurrentAllocator;
}

bool MetalBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2) {
        return false;
    }
    if (maxIndex == 2 && mBufferPoolShapeImmutable.get() == nullptr) {
        mBufferPoolShapeImmutable.reset(_createBufferAllocator(mRuntime, mStaticBufferPool.get(), true));
        mBufferPool.reset(_createBufferAllocator(mRuntime, mStaticBufferPool.get(), true));
    }
    if (1 == index) {
        mCurrentAllocator = mBufferPoolShapeImmutable.get();
    } else {
        mCurrentAllocator = mBufferPool.get();
    }
    return true;
}

bool MetalBackend::onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
    if (nullptr == dstInfo) {
        return true;
    }
    auto dst = (MNNMetalTensorContent*)dstInfo;
    dst->type.code = halide_type_float;
    if (mUseFloatAsFp16) {
        dst->type.bits = 16;
    } else {
        dst->type.bits = 32;
    }
    MNNMetalGetTensorContent(dst, (void*)tensor);
    return true;
}

bool MetalBackend::isCommandEncoderSet() {
    return mOpEncoderSet;// !isCommitEachShader & mOpFullSupport
}

bool MetalBackend::isCmdBufferCommit() {
    auto ctx = (__bridge MNNMetalContext *)context();
    if(!ctx.isCommitEachShader) {
        return false;
    }
    
    //TODO: set magic number
    const int magicNum = 2;
    mEncoderCount++;
    if(mEncoderCount != 0 && mEncoderCount % magicNum == 0) {
        return true;
    }
    return false;
}

void MetalBackend::addOpEncoder(std::function<void(void)> opEncoder) {
    if(mFrameEncodeCache) {
        mOpEncoders.push_back(opEncoder);
    }
}

id<MTLBuffer> MetalBackend::getHostBuffer(size_t size) const {
    // reuse
    if (nullptr != mHostBuffer && mHostBuffer.length >= size) {
        return mHostBuffer;
    }

    // create larger
    auto context = (__bridge MNNMetalContext *)this->context();
    mHostBuffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return mHostBuffer;
}

id<MTLBuffer> MetalBackend::getConstBuffer(size_t size) const {
    if (size < METAL_CONST_BUFFER_LIMIT) {
        if (!mHoldBuffers.empty()) {
            auto res = mHoldBuffers.front();
            mHoldBuffers.pop();
            return res;
        }
        size = METAL_CONST_BUFFER_LIMIT;
    }
    auto context = (__bridge MNNMetalContext *)this->context();
    auto buffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return buffer;
}
void MetalBackend::returnConstBuffer(id<MTLBuffer> buffer) const {
    mHoldBuffers.push(buffer);
}

MTLSize getTensorShape(id<MTLBuffer> shape, const Tensor *tensor) {
    int s = 1, c = 1, b = 1;
    if (tensor->dimensions() == 4) {
        s = tensor->width() * tensor->height();
        c = tensor->channel();
        b = tensor->batch();
    } else if (tensor->dimensions() >= 2){
        for (int i=2; i<tensor->dimensions(); ++i) {
            s *= tensor->length(i);
        }
        c = tensor->length(1);
        b = tensor->length(0);
    }

    int z = UP_DIV(c, 4);

    // shape
    ((int *)shape.contents)[0] = s;
    ((int *)shape.contents)[1] = c;
    ((int *)shape.contents)[2] = b;
    ((int *)shape.contents)[3] = b * z;

    // threads
    MTLSize threads = {(NSUInteger)s, (NSUInteger)b * z, 1};
    return threads;
}

enum MetalCastType : int {
    // no cast
    None = 0,
    // metal float to float
    Up,
    // float to metal float
    Down
};

static NSString *kernelForConvert(halide_type_t type, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to, MetalCastType cast) {
    if (type.code == halide_type_float) {
        NSString *map[3][MNN_DATA_FORMAT_MAX + 1][MNN_DATA_FORMAT_MAX + 1] = {
            // none
            {
                // from MNN_DATA_FORMAT_NCHW
                {nil, nil, @"cvt_f_NCHW_to_NC4HW4", nil, nil},
                // from MNN_DATA_FORMAT_NHWC
                {nil, nil, @"cvt_f_NHWC_to_NC4HW4", nil, nil},
                // from MNN_DATA_FORMAT_NC4HW4
                {@"cvt_f_NC4HW4_to_NCHW", @"cvt_f_NC4HW4_to_NHWC", nil, nil, nil},
                // from MNN_DATA_FORMAT_NHWC4
                {nil, nil, nil, nil, nil},
                // from MNN_DATA_FORMAT_UNKNOWN
                {nil, nil, nil, nil, nil},
            },
            // up
            {
                // from MNN_DATA_FORMAT_NCHW
                {nil, nil, @"upcast_f_NCHW_to_NC4HW4", nil, nil},
                // from MNN_DATA_FORMAT_NHWC
                {@"upcast_f_NHWC_to_NCHW", nil, @"upcast_f_NHWC_to_NC4HW4", nil, nil},
                // from MNN_DATA_FORMAT_NC4HW4
                {@"upcast_f_NC4HW4_to_NCHW", @"upcast_f_NC4HW4_to_NHWC", nil, nil, nil},
                // from MNN_DATA_FORMAT_NHWC4
                {nil, nil, nil, nil, nil},
                // from MNN_DATA_FORMAT_UNKNOWN
                {nil, nil, nil, nil, nil},
            },
            // down
            {
                // from MNN_DATA_FORMAT_NCHW
                {nil, @"downcast_f_NCHW_to_NHWC", @"downcast_f_NCHW_to_NC4HW4", nil, nil},
                // from MNN_DATA_FORMAT_NHWC
                {nil, nil, @"downcast_f_NHWC_to_NC4HW4", nil, nil},
                // from MNN_DATA_FORMAT_NC4HW4
                {@"downcast_f_NC4HW4_to_NCHW", @"downcast_f_NC4HW4_to_NHWC", nil, nil, nil},
                // from MNN_DATA_FORMAT_NHWC4
                {nil, nil, nil, nil, nil},
                // from MNN_DATA_FORMAT_UNKNOWN
                {nil, nil, nil, nil, nil},
            },
        };
        return map[cast][from][to];
    } else {
        NSString *map[MNN_DATA_FORMAT_MAX + 1][MNN_DATA_FORMAT_MAX + 1] = {
            // from MNN_DATA_FORMAT_NCHW
            {nil, nil, @"cvt_u_NCHW_to_NC4HW4", nil, nil},
            // from MNN_DATA_FORMAT_NHWC
            {nil, nil, @"cvt_u_NHWC_to_NC4HW4", nil, nil},
            // from MNN_DATA_FORMAT_NC4HW4
            {@"cvt_u_NC4HW4_to_NCHW", @"cvt_u_NC4HW4_to_NHWC", nil, nil, nil},
            // from MNN_DATA_FORMAT_NHWC4
            {nil, nil, nil, nil, nil},
            // from MNN_DATA_FORMAT_UNKNOWN
            {nil, nil, nil, nil, nil},
        };
        return map[from][to];
    }
}

void MetalBackend::onResizeBegin() {
    mFrameEncodeCache = false;
    mOpEncoderSet = false;
    mOpEncoders.clear();
    
    // Abort last inference task if needed
    flushEncoder();
    _commandBuffer_net = nil;
    _commandBuffer = nil;
    wait();
    mCurrentAllocator->reset();
}

ErrorCode MetalBackend::onResizeEnd() {
    auto ctx = (__bridge MNNMetalContext *)context();
    mFrameEncodeCache = (!ctx.isCommitEachShader && mSupportDeferEncode);
    return mCurrentAllocator->compute();
}

void MetalBackend::onCopyHostToDevice(const Tensor *src, const Tensor *dst) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *) (dst->deviceId()))->getBuffer();
    auto floats  = src->getType().code == halide_type_float;
    // For command queue from user, need user to make sure last frame's gpu work is ready
    bool needWait = !mRuntime->userSync();
    // cast
    if (sfmt == dfmt || src->dimensions() <= 1) {
        if (floats && mUseFloatAsFp16) {
            NSUInteger size = src->elementSize();
            auto sizeC4 = UP_DIV(size, 4);
            auto host = this->getHostBuffer(sizeC4 * 4 * sizeof(float));
            if (needWait) {
                wait();
            }
            memcpy(host.contents, src->host<float>(), src->size());
            unsigned int limits[] = {
                (unsigned int)sizeC4,
                1,
                1,
                1
            };
            ::memcpy(mShapeH2D.contents, limits, sizeof(limits));
            auto encoder    = [getCommandBufferForBufferCopy() computeCommandEncoder];
            auto pipeline = [ctx pipelineWithName:@"downcast_float4" fp16:mUseFloatAsFp16];
            [encoder setComputePipelineState:pipeline];
            
            [encoder setBuffer:host offset:0 atIndex:0];
            [encoder setBuffer:device offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
            [encoder setBuffer:mShapeH2D offset:0 atIndex:2];
            //[ctx dispatchEncoder:encoder threads:{sizeC4, 1, 1} bandwidth:bandwidth];
            std::pair<MTLSize, MTLSize> threads;
            threads.first = {sizeC4, 1, 1};
            threads.second = {[pipeline maxTotalThreadsPerThreadgroup], 1, 1};
            threads.second.width = threads.second.width <= threads.first.width ? threads.second.width : threads.first.width;
            threads.first.width = UP_DIV(threads.first.width, threads.second.width);
            [encoder dispatchThreadgroups:threads.first threadsPerThreadgroup:threads.second];
            [encoder endEncoding];
            commit();
            //[ctx wait];
        } else {
            if (needWait) {
                wait();
            }
            memcpy((uint8_t*)device.contents + TensorUtils::getDescribe(dst)->extra.offset, src->host<uint8_t>(), src->size());
        }
    }
    // convert
    else {

        auto buffer = getHostBuffer(src->elementSize() * sizeof(float));
        if (needWait) {
            wait();
        }
        auto size = getTensorShape(mShapeH2D, src);
        memcpy(buffer.contents, src->host<float>(), src->size());
        auto encoder = [getCommandBufferForBufferCopy() computeCommandEncoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Down);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt
        auto pipeline = [ctx pipelineWithName:kernel fp16:mUseFloatAsFp16];
        [encoder setComputePipelineState:pipeline];
        
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBuffer:device offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
        [encoder setBuffer:mShapeH2D offset:0 atIndex:2];
        auto gl = [ctx computeBestGroupAndLocal:pipeline threads:size];
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
        [encoder endEncoding];
        commit();
        //[ctx wait];
    }
}

void MetalBackend::onCopyDeviceToHost(const Tensor *src, const Tensor *dst) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId())->getBuffer();
    auto floats  = src->getType().code == halide_type_float;
    // cast
    if (sfmt == dfmt || src->dimensions() <= 1) {
        if (floats && mUseFloatAsFp16) {
            auto eleSize = dst->elementSize();
            eleSize = UP_DIV(eleSize, 4) * 4;
            auto buffer = getHostBuffer(eleSize * dst->getType().bytes());

            NSUInteger size = src->elementSize();
            auto encoder    = [getCommandBufferForBufferCopy() computeCommandEncoder];
            auto pipeline = [ctx pipelineWithName:@"upcast_float4" fp16:mUseFloatAsFp16];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:device offset:TensorUtils::getDescribe(src)->extra.offset atIndex:0];
            [encoder setBuffer:buffer offset:0 atIndex:1];
            auto sizeC4 = UP_DIV(size, 4);
            unsigned int limits[] = {
                (unsigned int)sizeC4,
                1,
                1,
                1
            };
            ::memcpy(mShapeD2H.contents, limits, sizeof(limits));
            [encoder setBuffer:mShapeD2H offset:0 atIndex:2];
            //[ctx dispatchEncoder:encoder threads:{sizeC4, 1, 1} bandwidth:bandwidth];
            std::pair<MTLSize, MTLSize> threads;
            threads.first = {sizeC4, 1, 1};
            threads.second = {[pipeline maxTotalThreadsPerThreadgroup], 1, 1};
            threads.second.width = threads.second.width <= threads.first.width ? threads.second.width : threads.first.width;
            threads.first.width = UP_DIV(threads.first.width, threads.second.width);
            [encoder dispatchThreadgroups:threads.first threadsPerThreadgroup:threads.second];
            
            [encoder endEncoding];
            commit();
            wait();

            memcpy(dst->host<float>(), buffer.contents, dst->size());
        } else {
            commit();
            wait();
            memcpy(dst->host<uint8_t>(), (uint8_t*)device.contents + TensorUtils::getDescribe(src)->extra.offset, dst->size());
        }
    }
    // convert
    else {
        auto size = getTensorShape(mShapeD2H, src);
        auto buffer  = getHostBuffer(dst->size());
        auto encoder    = [getCommandBufferForBufferCopy() computeCommandEncoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Up);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto pipeline = [ctx pipelineWithName:kernel fp16:mUseFloatAsFp16];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:device offset:TensorUtils::getDescribe(src)->extra.offset atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:1];
        [encoder setBuffer:mShapeD2H offset:0 atIndex:2];
        auto gl = [ctx computeBestGroupAndLocal:pipeline threads:size];
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
        [encoder endEncoding];
        commit();
        wait();
        memcpy(dst->host<float>(), buffer.contents, dst->size());
    }
}
static const char* gCopy = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
kernel void main0(const device int4 *in [[buffer(0)]], device int4 *out [[buffer(1)]], constant uint4& limit [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < limit.x) {
        out[int(gid)] = in[int(gid)];
    }
})metal";
void MetalBackend::onCopyDeviceToDevice(const Tensor *src, const Tensor *dst,
                                        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const {
    auto ctx    = (__bridge MNNMetalContext *)context();
    auto standalone = encoder == nil;
    encoder         = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
    auto sfmt       = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt       = TensorUtils::getDescribe(dst)->dimensionFormat;

    // copy
    if (sfmt == dfmt || src->dimensions() <= 1) {
        auto size      = dst->usize();
        if (mUseFloatAsFp16 && dst->getType().code == halide_type_float) {
            size = size / 2;
        }
        size = UP_DIV(size, (4 * sizeof(float)));
        std::vector<std::string> keys = {
            "copyC4"
        };
        id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = makeComputePipelineWithSourceOption(gCopy, "main0", nil);
            mRuntime->insertPipeline(keys, pipeline);
        }
        [encoder setComputePipelineState:pipeline];
        if (shape == nil) {
            shape = getConstBuffer(4 * sizeof(int));
        }
        ((uint32_t*)[shape contents])[0] = size;
        setTensor(src, encoder, 0);
        setTensor(dst, encoder, 1);
        [encoder setBuffer:shape offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(size, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    // convert
    else {
        auto kernel = kernelForConvert(src->getType(), sfmt, dfmt, None);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt
        if (shape == nil) {
            shape = getConstBuffer(4 * sizeof(int));
        }

        auto size     = getTensorShape(shape, src);
        auto pipeline = [ctx pipelineWithName:kernel fp16:mUseFloatAsFp16];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(src->buffer().device))->getBuffer() offset:TensorUtils::getDescribe(src)->extra.offset atIndex:0];
        [encoder setBuffer:( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(dst->buffer().device))->getBuffer() offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
        [encoder setBuffer:shape offset:0 atIndex:2];
        auto gl = [ctx computeBestGroupAndLocal:pipeline threads:size];
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }

    if (standalone) {
        [encoder endEncoding];
        MNN_PRINT_ENCODER(ctx, encoder);
    }
}

void MetalBackend::onCopyBuffer(const Tensor *src, const Tensor *dst) const {
    flushEncoder();
    auto ctx = (__bridge MNNMetalContext *)context();
    if(!mFrameEncodeCache) {
        commit_net();
    }

    onCopyBuffer(src, dst, nil, nil);
}

id<MTLComputeCommandEncoder> MetalBackend::encoder_for_net() const {
    if (nil == mComputeEncoder) {
        mComputeEncoder = encoder_net();//TO DO :: use which cmdBuffer
    }
    return mComputeEncoder;
}

void MetalBackend::onCopyBuffer(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const {
    MNN_ASSERT(src->buffer().dimensions == dst->buffer().dimensions);
    
    if (!src->buffer().host && !dst->buffer().host) {
        onCopyDeviceToDevice(src, dst, encoder, shape);
    } else if (!src->buffer().host && dst->buffer().host) {
        onCopyDeviceToHost(src, dst);

    } else if (src->buffer().host && !dst->buffer().host) {
        onCopyHostToDevice(src, dst);
        
    } else {
        MNN_ASSERT(false); // should not be handled here
    }
}
int MetalBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    flushEncoder();
    auto ctx = (__bridge MNNMetalContext *)context();
    if(!mOpEncoderSet) {
        commit_net();
    }
    if (toCpu) {
        wait();
    }
    return 0;
}
id<MTLCommandBuffer> MetalBackend::getCommandBufferForBufferCopy() const {
    if (nil == _commandBuffer) {
        _commandBuffer = [_commandQueue commandBuffer];
        if (!mSupportDeferEncode) {
            // In this case _commandBuffer should be the same as _commandBuffer_net
            _commandBuffer_net = _commandBuffer;
        }
    }
    return _commandBuffer;
}
id<MTLCommandBuffer> MetalBackend::getCommandBufferForNet() const {
    if (nil == _commandBuffer_net) {
        _commandBuffer_net = [_commandQueue commandBuffer];
        if (!mSupportDeferEncode) {
            // In this case _commandBuffer should be the same as _commandBuffer_net
            _commandBuffer = _commandBuffer_net;
        }
    }
    return _commandBuffer_net;
}

void MetalBackend::setTensor(const MNN::Tensor* tensor, id<MTLComputeCommandEncoder> encoder, int index) {
    [encoder setBuffer:((MetalRuntimeAllocator::MetalBufferAlloc *)tensor->deviceId())->getBuffer() offset:TensorUtils::getDescribe(tensor)->extra.offset atIndex:index];
}
std::pair<id<MTLBuffer>, int> MetalBackend::getBuffer(MNN::Tensor* tensor) {
    return std::make_pair(((MetalRuntimeAllocator::MetalBufferAlloc *)tensor->deviceId())->getBuffer(), TensorUtils::getDescribe(tensor)->extra.offset);
}


void MetalBackend::commit() const {
    if (nil != _commandBuffer &&  _commandBuffer.status < MTLCommandBufferStatusCommitted) {
        [_commandBuffer commit];
        _waiting = _commandBuffer;
        _commandBuffer = nil;
        if (!mSupportDeferEncode) {
            // In this case _commandBuffer should be the same as _commandBuffer_net
            _commandBuffer_net = nil;
        }
    }
}

void MetalBackend::commit_net() const {
    if (nil != _commandBuffer_net && _commandBuffer_net.status < MTLCommandBufferStatusCommitted) {
        [_commandBuffer_net commit];
        _waiting = _commandBuffer_net;
        _commandBuffer_net = nil;
        if (!mSupportDeferEncode) {
            // In this case _commandBuffer should be the same as _commandBuffer_net
            _commandBuffer = nil;
        }
    }
}

void MetalBackend::wait() const {
    if (nil != _waiting) {
        auto buffer = _waiting;
        if (buffer.status >= MTLCommandBufferStatusCompleted) {
            _waiting = nil;
            return;
        }

#if MNN_METAL_BENCHMARK
        NSTimeInterval begin = [NSDate timeIntervalSinceReferenceDate];
        [buffer waitUntilCompleted];
        NSTimeInterval end = [NSDate timeIntervalSinceReferenceDate];
        if (@available(iOS 10.3, *)) {
            printf("[METAL] commit costs: %.3fms\t(kernel: %.3fms, GPU: %.3fms)\n", (end - begin) * 1000.f,
                   (buffer.kernelEndTime - buffer.kernelStartTime) * 1000.f,
                   (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.f);
        } else {
            printf("[METAL] commit costs: %.3fms\n", (end - begin) * 1000.f);
        }
#else
        [buffer waitUntilCompleted];
#endif

#if MNN_METAL_DEBUG
        if (buffer.error) {
            printf("[METAL] %s\n", buffer.error.localizedDescription.UTF8String);
        }
#endif
    }
    _waiting = nil;
}

id<MTLComputePipelineState> MetalBackend::makeComputePipelineWithSourceOption(const char* csource, const char* cname, MTLCompileOptions *options) const{
    auto ctx = (__bridge MNNMetalContext *)context();
    auto source = [[NSString alloc] initWithUTF8String:csource];
    auto name = [[NSString alloc] initWithUTF8String:cname];
    return [ctx pipelineWithSourceOption:source name:name options:options];
}
void MetalRuntime::setCommandQueue(id<MTLCommandQueue> queue, bool userSync) {
    mQueue = queue;
    mUserSync = userSync;
}
id<MTLComputePipelineState> MetalRuntime::findPipeline(const std::vector<std::string>& keys) const {
    auto iter = mCachePipeine.find(keys);
    if (iter == mCachePipeine.end()) {
        return nil;
    }
    return iter->second;
}
void MetalRuntime::insertPipeline(const std::vector<std::string>& keys, id<MTLComputePipelineState> pipeline) const {
    mCachePipeine.insert(std::make_pair(keys, pipeline));
}

void MetalRuntime::setGpuMode(const int mode_num) {
    int totalSet = 0;
    bool isSet = (mode_num & MNN_GPU_MEMORY_BUFFER);
    if(isSet) {
        totalSet++;
    }
    isSet = (mode_num & MNN_GPU_MEMORY_IMAGE);
    if(isSet) {
        totalSet++;
    }
    if(totalSet > 0) {
        MNN_PRINT("warning: set BUFFER and IMAGE mode is not useful for metal, it doesn't matter, cl_mode:%x！\n", mode_num);
    }
    
    totalSet = 0;
    isSet = (mode_num & MNN_GPU_TUNING_NONE);
    if(isSet) {
        mTuneLevel = Never;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_FAST);
    if(isSet) {
        mTuneLevel = Fast;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_NORMAL);
    if(isSet) {
        mTuneLevel = Normal;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_HEAVY);
    if(isSet) {
        mTuneLevel = Heavy;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_WIDE);
    if(isSet) {
        mTuneLevel = Wide;
        totalSet++;
    }

    if(totalSet != 1) {
        MNN_PRINT("set multi tuning mode is not permitted, please check cl_mode:%x！\n", mode_num);
    }
}

MetalRuntime* MetalRuntime::create(const Backend::Info& info, id<MTLDevice> device) {
    MNNMetalSharedContext sharedContext;
    sharedContext.device = nil;
    sharedContext.queue = nil;
    if (info.user != nullptr) {
        if (info.user->sharedContext != nullptr) {
            sharedContext.device = ((MNNMetalSharedContext*)info.user->sharedContext)->device;
            sharedContext.queue = ((MNNMetalSharedContext*)info.user->sharedContext)->queue;
        }
    }
    if (nil == sharedContext.device) {
        sharedContext.device = device;
    }
    auto mContext = (__bridge_retained void *)[[MNNMetalContext alloc] init];
    auto ctx = (__bridge MNNMetalContext *)mContext;
    BOOL res = [ctx initWithSharedContext:&sharedContext dev:device];
    if (!res) {
        CFRelease(mContext);
        return nullptr;
    }
    auto rt = new MetalRuntime(mContext);
    rt->setGpuMode(info.gpuMode);
    if (nil != sharedContext.queue) {
        rt->setCommandQueue(sharedContext.queue, true);
    }
    bool supportDefer = info.numThread & MNN_GPU_RECORD_BATCH;
    if ((!supportDefer) && nil == sharedContext.queue) {
        id<MTLCommandQueue> queue = [sharedContext.device newCommandQueue];
        rt->setCommandQueue(queue, false);
    }
    if (nullptr != info.user) {
        rt->mDefaultConfig = *info.user;
    }
    return rt;
}

MetalRuntime::MetalRuntime(void* context) {
    mContext = context;
    auto ctx = (__bridge MNNMetalContext *)mContext;
    std::shared_ptr<EagerBufferAllocator::Allocator> allocator(new MetalRuntimeAllocator([ctx device]));
    mStatic.reset(new EagerBufferAllocator(allocator));
    mTunedInfo = new TunedInfo;
}

MetalRuntime::~ MetalRuntime() {
    if(mContext) {
        CFRelease(mContext);
    }
    delete mTunedInfo;
}

bool MetalRuntime::setCache(std::pair<const void*, size_t> cache) {//Get Cache
    auto buffer = cache.first;
    auto size   = cache.second;
    if (nullptr == buffer) {
        mCacheOutside = nullptr;
        mCacheOutsideSize = 0;
        mBuffer.clear();
        return false;//actually get nothing
    }
    mCacheOutsideSize = size;
    mCacheOutside = buffer;
    auto cacheBuffer = GetCache(buffer);
    flatbuffers::Verifier verify((const uint8_t*)cache.first, cache.second);
    if (false == VerifyCacheBuffer(verify)) {
        return false;
    }
    if (nullptr == cacheBuffer->tunings()) {
        return false;
    }

    // Load Auto Tuning Info
    if (nullptr != cacheBuffer->tunings()) {
        auto tuningInfo = cacheBuffer->tunings();
        for (int i=0; i<tuningInfo->size(); ++i) {
            auto tun = tuningInfo->GetAs<Autotuning>(i);
            if (nullptr == tun->threadSize() || nullptr == tun->groupSize() || nullptr == tun->key()) {
                MNN_ERROR("Error tunning info\n");
                continue;
            }
            std::vector<uint32_t> glo(tun->threadSize()->size());
            for (int v=0; v<glo.size(); ++v) {
                glo[v] = tun->threadSize()->data()[v];
            }
            std::vector<uint32_t> grop(tun->groupNum()->size());
            for (int v=0; v<grop.size(); ++v) {
                grop[v] = tun->groupNum()->data()[v];
            }
            std::vector<uint32_t> loc(tun->groupSize()->size());
            for (int v=0; v<loc.size(); ++v) {
                loc[v] = tun->groupSize()->data()[v];
            }
            uint32_t cost = tun->timeCost();
            mTunedThreadGroup.insert(std::make_pair(std::make_pair(tun->key()->str(), glo), std::make_tuple(grop, loc, cost)));
        }
    }
    return true;
}

std::pair<const void*, size_t> MetalRuntime::makeCache(TunedInfo* info) {//make Cache
    std::unique_ptr<CacheT> cache(new CacheT);
    // Get All Autotuning cache
    for (auto& iter : mTunedThreadGroup) {
        std::unique_ptr<AutotuningT> tuning(new AutotuningT);
        tuning->key = iter.first.first;
        tuning->threadSize = iter.first.second;
        
        tuning->groupNum = std::get<0>(iter.second);
        tuning->groupSize = std::get<1>(iter.second);
        tuning->timeCost = std::get<2>(iter.second);

        cache->tunings.emplace_back(std::move(tuning));
    }
    cache->tuned = std::move(info->mInfos);

    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Cache::Pack(builder, cache.get());
    builder.Finish(lastOffset);
    mBuffer.resize(builder.GetSize());
    ::memcpy(mBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
    return std::make_pair(mBuffer.data(), mBuffer.size());
}

float MetalRuntime::onGetMemoryInMB() {
    auto staticMemoryInMB = mStatic->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

void MetalRuntime::onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           const MNN::Op* op) {
    if (nullptr != op->name()) {
        auto dstInfo = mTunedInfo;
        std::unique_ptr<MetalCache::OpInfoT> opInfo(new MetalCache::OpInfoT);;
        opInfo->type = op->type();
        opInfo->name = op->name()->str();
        opInfo->inputs.resize(inputs.size());
        for (int v=0; v<opInfo->inputs.size(); ++v) {
            opInfo->inputs[v].reset(new MetalCache::TensorInfoT);
            opInfo->inputs[v]->shape.resize(inputs[v]->dimensions());
            for (int u=0; u<opInfo->inputs[v]->shape.size(); ++u) {
                opInfo->inputs[v]->shape[u] = inputs[v]->length(u);
            }
        }
        opInfo->outputs.resize(outputs.size());
        for (int v=0; v<opInfo->outputs.size(); ++v) {
            opInfo->outputs[v].reset(new MetalCache::TensorInfoT);
            opInfo->outputs[v]->shape.resize(outputs[v]->dimensions());
            for (int u=0; u<opInfo->outputs[v]->shape.size(); ++u) {
                opInfo->outputs[v]->shape[u] = outputs[v]->length(u);
            }
        }
        dstInfo->mInfos.emplace_back(std::move(opInfo));
    }
}
static bool _checkTensorInfo(const MetalCache::TensorInfoT* dst, const Tensor* src) {
    if (dst->shape.size() != src->dimensions()) {
        return false;
    }
    for (int j=0; j<dst->shape.size(); ++j) {
        if (dst->shape[j] != src->length(j)) {
            return false;
        }
    }
    return true;
}
bool MetalRuntime::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const MNN::Op* op, Runtime::OpInfo& dstInfo) const {
    dstInfo.initCostLong = true;
    if (nullptr == op->name()) {
        dstInfo.initCostLong = false;
        return true;
    }
    for(auto& info : mTunedInfo->mInfos) {
        if (info->type != op->type()) {
            continue;
        }
        if (info->name != op->name()->str()) {
            continue;
        }
        if (info->inputs.size() != inputs.size() || info->outputs.size() != outputs.size()) {
            continue;
        }
        bool match = true;
        for (int i=0; i<inputs.size(); ++i) {
            auto& dst = info->inputs[i];
            auto src = inputs[i];
            if (!_checkTensorInfo(dst.get(), src)) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto& dst = info->outputs[i];
            auto src = outputs[i];
            if (!_checkTensorInfo(dst.get(), src)) {
                match = false;
                break;
            }
        }
        if (match) {
            // All Info is match
            dstInfo.initCostLong = false;
            break;
        }
    }
    return true;
}

Backend* MetalRuntime::onCreate(const BackendConfig* config) const {
    BackendConfig::PrecisionMode precision = mDefaultConfig.precision;
    if (nullptr != config) {
        precision = config->precision;
    }
    bool useFp16AsFp32 = precision != BackendConfig::Precision_High;
    return new MetalBackend(mStatic, this, useFp16AsFp32);
}

void MetalRuntime::onGabageCollect(int level) {
    mStatic->release(false);
}

std::pair<const void*, size_t> MetalRuntime::onGetCache() {//make Cache
    return makeCache(mTunedInfo);
}

bool MetalRuntime::onSetCache(const void* buffer, size_t size) {//set Cache
    if (nullptr == buffer) {
        return false;
    }
    auto cacheBuffer = MetalCache::GetCache(buffer);
    flatbuffers::Verifier verify((const uint8_t*)buffer, size);
    if (false == VerifyCacheBuffer(verify)) {
        return false;
    }
    if(nullptr != cacheBuffer->tuned()) {
        for (int i=0; i<cacheBuffer->tuned()->size(); ++i) {
            auto srcInfo = cacheBuffer->tuned()->GetAs<MetalCache::OpInfo>(i);
            std::unique_ptr<MetalCache::OpInfoT> dst(srcInfo->UnPack());
            mTunedInfo->mInfos.emplace_back(std::move(dst));
        }
    }
    return setCache(std::make_pair(buffer, size));
}

MemChunk MetalRuntimeAllocator::onAlloc(size_t size, size_t align) {
    auto buffer = [mDevice newBufferWithLength:size options:MTLCPUCacheModeDefaultCache];
    auto mMetalBufferAlloc = new MetalBufferAlloc(buffer);
    return MemChunk((void *)mMetalBufferAlloc, 0);
}
void MetalRuntimeAllocator::onRelease(MemChunk ptr) {
    delete (MetalBufferAlloc *)ptr.first;
}

class MetalRuntimeCreator : public RuntimeCreator {
public:
    MetalRuntimeCreator(id<MTLDevice> device) {
        mDevice = device;
    }
    virtual ~ MetalRuntimeCreator() {
        // Do nothing
    }
    virtual Runtime *onCreate(const Backend::Info &info) const {
        auto rt = MetalRuntime::create(info, mDevice);
        return rt;
    }
private:
    id<MTLDevice> mDevice;
};

void registerMetalRuntimeCreator() {
    // according to
    // https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/HardwareGPUInformation/HardwareGPUInformation.html
    // not all device with iOS 8+ supports metal.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (nil != device) {
        registerMetalOps();
#ifdef MNN_SUPPORT_RENDER
        registerMetalRenderOps();
#endif
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_METAL, new MetalRuntimeCreator(device), false);
    } else {
        MNN_ERROR("Init Metal Error\n");
    }
}
} // namespace MNN
#else
namespace MNN {
void registerMetalRuntimeCreator() {
}
};
int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor) {
    return -1;
}

#endif
