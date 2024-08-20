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
    if (runtime->hint().memoryAllocatorType == Runtime::Allocator_Defer && secondResize) {
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
    if (size < METAL_CONST_BUFFER_LIMIT) {
        size = METAL_CONST_BUFFER_LIMIT;
    }
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
static inline void _getNCPlane(const Tensor* tensor, int& s, int& c, int& b) {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    s = 1, c = 1, b = 1;
    b = tensor->length(0);
    if (format == MNN_DATA_FORMAT_NHWC) {
        c = tensor->length(tensor->dimensions()-1);
        for (int i=1; i<tensor->dimensions()-1; ++i) {
            s *= tensor->length(i);
        }
    } else {
        c = tensor->length(1);
        for (int i=2; i<tensor->dimensions(); ++i) {
            s *= tensor->length(i);
        }
    }
}
MTLSize getTensorShape(id<MTLBuffer> shape, const Tensor *tensor) {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    int s, b, c;
    _getNCPlane(tensor, s, c, b);
    int z = UP_DIV(c, 4);

    // shape
    ((int *)shape.contents)[0] = b;
    ((int *)shape.contents)[1] = c;
    ((int *)shape.contents)[2] = s;
    ((int *)shape.contents)[3] = 1;
    
    // stride
    if (format == MNN_DATA_FORMAT_NHWC) {
        ((int *)shape.contents)[4] = s * c;
        ((int *)shape.contents)[5] = 1;
        ((int *)shape.contents)[6] = c;
        ((int *)shape.contents)[7] = 1;
    } else {
        ((int *)shape.contents)[4] = s * c;
        ((int *)shape.contents)[5] = s;
        ((int *)shape.contents)[6] = 1;
        ((int *)shape.contents)[7] = 1;
    }
    // threads
    MTLSize threads = {(NSUInteger)s * b * z, 1, 1};
    return threads;
}
static const char* gTranspose = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct tensor_shape {
    uint4 size; // n, c, plane, 1
    uint4 stride;
};
kernel void main0(const device IType* in [[buffer(0)]], device OType* out [[buffer(1)]], constant tensor_shape &uConstant [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    int channel = uConstant.size.y;
    if (gid < channel * uConstant.size.x * uConstant.size.z) {
        int tmp = gid % (channel * uConstant.size.x);
        int x = gid / (channel * uConstant.size.x);
        int b = tmp / channel;
        int c = tmp % channel;
        int outPos = b * uConstant.size.y * uConstant.size.z + c * uConstant.size.z + x;
        int inPos = b * uConstant.size.y * uConstant.size.z + c + x * uConstant.size.y;
        out[outPos] = (OType)(in[inPos]);
    }
})metal";

static const char* gNC4HW4Convert = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct tensor_shape {
    uint4 size; // n, c, plane, 1
    uint4 stride;
};
kernel void main0(const device IType* in [[buffer(0)]], device OType* out [[buffer(1)]], constant tensor_shape &uConstant [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    int channelC4 = (uConstant.size.y + 3) / 4;
    if (gid < channelC4 * uConstant.size.x * uConstant.size.z)
    {
        int3 pos;
        pos.z = gid % (channelC4 * uConstant.size.x);
        pos.y = gid / (channelC4 * uConstant.size.x);
        pos.x = 0;
        int batchIndex = pos.z / channelC4;
        int zDiv4 = pos.z % channelC4;

        int lastZ = uConstant.size.y / 4;
        int cIndex = uConstant.size.y % 4;

        int z = zDiv4*4;
        int basicOffset = 0
            + batchIndex*uConstant.stride.x
            + z * uConstant.stride.y
            + pos.y * uConstant.stride.z
            ;
#ifdef MNN_OUTPUT_C4
        OType color = OType(0);
        if(zDiv4 == lastZ)
        {
            if(cIndex == 1)
            {
                color.r = in[basicOffset+0];
                color.g = 0.0;
                color.b = 0.0;
                color.a = 0.0;
            }
            else if(cIndex == 2)
            {
                color.r = in[basicOffset+0];
                color.g = in[basicOffset+1*uConstant.stride.y];
                color.b = 0.0;
                color.a = 0.0;
            }
            else
            {
                color.r = in[basicOffset+0];
                color.g = in[basicOffset+1*uConstant.stride.y];
                color.b = in[basicOffset+2*uConstant.stride.y];
                color.a = 0.0;
            }
        }
        else
        {
            color.r = in[basicOffset+0];
            color.g = in[basicOffset+1*uConstant.stride.y];
            color.b = in[basicOffset+2*uConstant.stride.y];
            color.a = in[basicOffset+3*uConstant.stride.y];
        }

        out[0
            + pos.y
            + uConstant.size.x * uConstant.size.z*zDiv4
            + batchIndex*uConstant.size.z
            ] = color;
#else
        IType color = in[0
            + pos.y
            + uConstant.size.x * uConstant.size.z*zDiv4
            + batchIndex*uConstant.size.z
            ];
        if(zDiv4 == lastZ)
        {
            if(cIndex == 1)
            {
                out[basicOffset+0*uConstant.stride.y] = color.r;
            }
            else if(cIndex == 2)
            {
                out[basicOffset+0*uConstant.stride.y] = color.r;
                out[basicOffset+1*uConstant.stride.y] = color.g;
            }
            else
            {
                out[basicOffset+0*uConstant.stride.y] = color.r;
                out[basicOffset+1*uConstant.stride.y] = color.g;
                out[basicOffset+2*uConstant.stride.y] = color.b;
            }
        }
        else
        {
            out[basicOffset+0*uConstant.stride.y] = color.r;
            out[basicOffset+1*uConstant.stride.y] = color.g;
            out[basicOffset+2*uConstant.stride.y] = color.b;
            out[basicOffset+3*uConstant.stride.y] = color.a;
        }
#endif
    }
}
)metal";

static const char* gCopy = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
kernel void main0(const device IType *in [[buffer(0)]], device OType *out [[buffer(1)]], constant uint4& limit [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < limit.x) {
        out[int(gid)] = (OType)in[int(gid)];
    }
})metal";

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

static std::string _getType(const halide_type_t& type, MNN_DATA_FORMAT format, bool useFp16AsFp32) {
    std::string res;
    if (type.code == halide_type_float) {
        if (useFp16AsFp32) {
            res = "half";
        } else {
            res = "float";
        }
    } else {
        switch (type.bytes()) {
            case 1:
                res = "char";
                break;
            case 2:
                res = "short";
                break;
            case 4:
                res = "int";
                break;
            default:
                MNN_ASSERT(false);
                break;
        }
    }
    if (format == MNN_DATA_FORMAT_NC4HW4) {
        return res + "4";
    }
    return res;
}
MetalBackend::CopyPipeline MetalBackend::_makeCopyInfo(const Tensor *src, const Tensor *dst, id<MTLBuffer> shape, int castType) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    MetalBackend::CopyPipeline res;
    auto sfmt = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt = TensorUtils::getDescribe(dst)->dimensionFormat;
    if (shape == nil) {
        shape = getConstBuffer(8 * sizeof(int));
    }
    res.shape = shape;
    if (sfmt == dfmt || src->dimensions() <= 1) {
        auto srcType = _getType(src->getType(), MNN_DATA_FORMAT_NC4HW4, mUseFloatAsFp16 && castType != 1);
        auto dstType = _getType(dst->getType(), MNN_DATA_FORMAT_NC4HW4, mUseFloatAsFp16 && castType != 2);
        auto size      = dst->elementSize();
        size = UP_DIV(size, 4);
        std::vector<std::string> keys = {
            "copyC4",
            srcType,
            dstType
        };
        ((uint32_t*)[shape contents])[0] = size;
        id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[1].c_str()) forKey:@"IType"];
            [dic setValue:@(keys[2].c_str()) forKey:@"OType"];
            option.preprocessorMacros = dic;
            pipeline = makeComputePipelineWithSourceOption(gCopy, "main0", option);
            mRuntime->insertPipeline(keys, pipeline);
        }
        res.groupSize = MTLSizeMake(UP_DIV(size, 256), 1, 1);
        res.localSize = MTLSizeMake(256, 1, 1);
        res.pipeline = pipeline;
        return res;
    }
    auto srcType = _getType(src->getType(), sfmt, mUseFloatAsFp16 && castType != 1);
    auto dstType = _getType(dst->getType(), dfmt, mUseFloatAsFp16 && castType != 2);
    if (sfmt == MNN_DATA_FORMAT_NC4HW4 || dfmt == MNN_DATA_FORMAT_NC4HW4) {
        auto normalTensor = dst;
        if (dfmt == MNN_DATA_FORMAT_NC4HW4) {
            normalTensor = src;
        }
        // convert C4 / NCHW
        std::vector<std::string> keys = {
            "c4convert",
            srcType,
            dstType
        };
        if (dfmt == MNN_DATA_FORMAT_NC4HW4) {
            keys.emplace_back("outputc4");
        }
        id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[1].c_str()) forKey:@"IType"];
            [dic setValue:@(keys[2].c_str()) forKey:@"OType"];
            if (dfmt == MNN_DATA_FORMAT_NC4HW4) {
                [dic setValue:@"1" forKey:@"MNN_OUTPUT_C4"];
            }
            option.preprocessorMacros = dic;
            pipeline = makeComputePipelineWithSourceOption(gNC4HW4Convert, "main0", option);
            mRuntime->insertPipeline(keys, pipeline);
        }
        res.pipeline = pipeline;
        auto size = getTensorShape(shape, normalTensor);
        auto gl = [ctx computeBestGroupAndLocal:pipeline threads:size];
        res.groupSize = gl.first;
        res.localSize = gl.second;
        return res;
    }
    // NCHW <-> NHWC
    std::vector<std::string> keys = {
        "transpose",
        srcType,
        dstType
    };
    id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
    if (nil == pipeline) {
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(keys[1].c_str()) forKey:@"IType"];
        [dic setValue:@(keys[2].c_str()) forKey:@"OType"];
        option.preprocessorMacros = dic;
        pipeline = makeComputePipelineWithSourceOption(gTranspose, "main0", option);
        mRuntime->insertPipeline(keys, pipeline);
    }
    res.pipeline = pipeline;
    int n, c, plane;
    _getNCPlane(dst, plane, c, n);
    auto shapePtr = (uint32_t*)shape.contents;
    shapePtr[0] = n;
    shapePtr[3] = 1;
    if (MNN_DATA_FORMAT_NHWC == dfmt) {
        shapePtr[1] = plane;
        shapePtr[2] = c;
    } else {
        shapePtr[1] = c;
        shapePtr[2] = plane;
    }
    auto size = plane * n * c;
    res.localSize = MTLSizeMake(256, 1, 1);
    res.groupSize = MTLSizeMake(UP_DIV(size, 256), 1, 1);
    return res;
}

static void _execute(id<MTLComputeCommandEncoder> encoder, const MetalBackend::CopyPipeline& info, std::pair<id<MTLBuffer>, int> src, std::pair<id<MTLBuffer>, int> dst) {
    [encoder setComputePipelineState:info.pipeline];
    [encoder setBuffer:src.first offset:src.second atIndex:0];
    [encoder setBuffer:dst.first offset:dst.second atIndex:1];
    [encoder setBuffer:info.shape offset:0 atIndex:2];
    [encoder dispatchThreadgroups:info.groupSize threadsPerThreadgroup:info.localSize];
}
void MetalBackend::onCopyDeviceToDevice(const Tensor *src, const Tensor *dst,
                                        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape, int castType) const {
    auto ctx    = (__bridge MNNMetalContext *)context();
    auto info = _makeCopyInfo(src, dst, shape, castType);
    auto standalone = encoder == nil;
    encoder = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
    _execute(encoder, info, MetalBackend::getBuffer(src), MetalBackend::getBuffer(dst));
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
        return;
    }
    auto sfmt = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt = TensorUtils::getDescribe(dst)->dimensionFormat;
    bool formatDiff = sfmt != dfmt && src->dimensions() > 1;
    auto floats  = src->getType().code == halide_type_float;
    bool dataTypeDiff = floats && mUseFloatAsFp16;
    bool needConvert = formatDiff || dataTypeDiff;

    if (!src->buffer().host && dst->buffer().host) {
        auto device = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId())->getBuffer();
        auto devicePtr = (uint8_t*)device.contents + TensorUtils::getDescribe(src)->extra.offset;
        if (needConvert) {
            auto tDst = const_cast<Tensor*>(dst);
            auto tmpBuffer = getHostBuffer(dst->usize());
            auto info = _makeCopyInfo(src, dst, shape, 2);
            auto standalone = encoder == nil;
            encoder = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
            _execute(encoder, info, MetalBackend::getBuffer(src), std::make_pair(tmpBuffer, 0));
            if (standalone) {
                [encoder endEncoding];
            }
            commit();
            devicePtr = (uint8_t*)tmpBuffer.contents;
        }
        wait();
        ::memcpy(dst->host<void>(), devicePtr, dst->usize());
        return;
    }
    if (src->buffer().host && !dst->buffer().host) {
        // For command queue from user, need user to make sure last frame's gpu work is ready
        bool needWait = !mRuntime->userSync();
        if (needWait) {
            wait();
        }
        auto srcSize = src->usize();
        if (needConvert) {
            auto tmpBuffer = getHostBuffer(srcSize);
            ::memcpy(tmpBuffer.contents, src->host<void>(), srcSize);
            auto info = _makeCopyInfo(src, dst, shape, 1);
            auto standalone = encoder == nil;
            encoder = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
            _execute(encoder, info, std::make_pair(tmpBuffer, 0), MetalBackend::getBuffer(dst));
            if (standalone) {
                [encoder endEncoding];
            }
            commit();
        } else {
            auto device = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)dst->deviceId())->getBuffer();
            auto devicePtr = (uint8_t*)device.contents + TensorUtils::getDescribe(dst)->extra.offset;
            ::memcpy(devicePtr, src->host<void>(), srcSize);
        }
        return;
    }
    MNN_ASSERT(false); // should not be handled here
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
std::pair<id<MTLBuffer>, int> MetalBackend::getBuffer(const MNN::Tensor* tensor) {
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
