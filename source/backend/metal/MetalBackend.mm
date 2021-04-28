//
//  MetalBackend.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalBackend.hpp"
#import <mutex>
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/TensorUtils.hpp"


namespace MNN {
#if MNN_METAL_ENABLED

void registerMetalOps();
MetalRuntime::BufferAllocator::BufferAllocator(void* context) {
    mContext = context;
}
MetalRuntime::BufferAllocator::~ BufferAllocator() {
    // Do nothing
}
float MetalRuntime::BufferAllocator::computeSizeInMB() const {
    float total = 0.0f;
    for (auto& iter : mAllocated) {
        total += iter.second / 1024.0f / 1024.0f;
    }
    for (auto& iter : mReusableBuffers) {
        total += iter.first / 1024.0f / 1024.0f;
    }
    return total;
}

float MetalRuntime::onGetMemoryInMB() {
    return mStatic->computeSizeInMB() + mDynamic->computeSizeInMB();
}

id<MTLBuffer> MetalRuntime::BufferAllocator::alloc(size_t size, bool seperate) {
    if (!seperate) {
        auto iter = mReusableBuffers.lower_bound(size);
        if (iter != mReusableBuffers.end()) {
            auto buffer = iter->second;
            mAllocated.insert(std::make_pair(buffer, iter->first));
            mReusableBuffers.erase(iter);
            return buffer;
        }
    }
    auto context = (__bridge MNNMetalContext *)mContext;
    auto buffer = [context newDeviceBuffer:size access:CPUWriteOnly];
    mAllocated.insert(std::make_pair(buffer, size));
    return buffer;
}

void MetalRuntime::BufferAllocator::release(id<MTLBuffer> buffer) {
    auto iter = mAllocated.find(buffer);
    MNN_ASSERT(iter != mAllocated.end());
    mReusableBuffers.insert(std::make_pair(iter->second, buffer));
    mAllocated.erase(iter);
}
void MetalRuntime::BufferAllocator::clear() {
    mReusableBuffers.clear();
}
id<MTLBuffer> MetalRuntime::getHostBuffer(size_t size) const {
    // reuse
    if (nullptr != mHostBuffer && mHostBuffer.length >= size) {
        return mHostBuffer;
    }

    // create larger
    auto context = (__bridge MNNMetalContext *)mContext;
    mHostBuffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return mHostBuffer;
}

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

MetalBackend::MetalBackend(const MetalRuntime* runtime) : Backend(MNN_FORWARD_METAL), mShapeH2D(runtime), mShapeD2H(runtime) {
    mRuntime = runtime;
}
MetalBackend::~MetalBackend() {
    // Do nothing
}
void *MetalBackend::context() const {
    return mRuntime->context();
}

bool MetalBackend::onAcquireBuffer(const Tensor *_tensor, StorageType storageType) {
    auto tensor  = const_cast<Tensor *>(_tensor);
    auto size    = tensor->elementSize();
    if (0 == size) {
        return false;
    }
    size = UP_DIV(size, 4) * 4;

    // use metal_float when meets float
    if (halide_type_float == tensor->buffer().type.code && tensor->buffer().type.bits == 32) {
        size*= sizeof(metal_float);
    } else {
        size *= tensor->getType().bytes();
    }

    // reuse if possible
    id<MTLBuffer> buffer;
    switch (storageType) {
        case Backend::STATIC: {
            buffer = mRuntime->mStatic->alloc(size);
        } break;
        case Backend::DYNAMIC: {
            buffer = mRuntime->mDynamic->alloc(size);
            mHoldBuffers.emplace_back(buffer);
        } break;
        case Backend::DYNAMIC_SEPERATE: {
            buffer = mRuntime->mDynamic->alloc(size, true);
            mHoldBuffers.emplace_back(buffer);
        } break;
    }
    tensor->buffer().device = (uint64_t)buffer;
    return true;
}
bool MetalBackend::onReleaseBuffer(const Tensor *tensor, StorageType storageType) {
    auto buffer = (__bridge id<MTLBuffer>)(void *)(tensor->buffer().device);
    if (buffer) {
        switch (storageType) {
            case Backend::STATIC: {
                mRuntime->mStatic->release(buffer);
            } break;
            case Backend::DYNAMIC: {
                mRuntime->mDynamic->release(buffer);
            } break;
            case Backend::DYNAMIC_SEPERATE: {
                // do nothing
            } break;
        }
    }
    return true;
}
bool MetalBackend::onClearBuffer() {
    mHoldBuffers.clear();
    return true;
}
std::pair<float, bool> MetalBackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                              const MNN::Op* op) {
    auto map  = getCreatorMap();
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        return std::make_pair(0.0f, false);
    }
    return std::make_pair(0.05f, true);
}

Execution *MetalBackend::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                  const Op *op) {
    auto map  = getCreatorMap();
    auto iter = map->find(op->type());
    
    if (iter == map->end()) {
        mOpFullSupport = false;
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type [%s], %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type [%s]\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, op, this);
    if (NULL == exe) {
        mOpFullSupport = false;
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

void MetalBackend::setOpEncoder() const {
    mOpEncoderSet = true;
}

void MetalBackend::onExecuteBegin() const {
    //
}
void MetalBackend::onExecuteEnd() const {
    flushEncoder();
    auto ctx = (__bridge MNNMetalContext *)context();
    [ctx commit_net];

    if(mFrameEncodeCache) {
        for(auto opEncoder : mOpEncoders) {
            opEncoder();
        }
        setOpEncoder();
    }
}

bool MetalBackend::isCommandEncoderSet() const{
    return mOpEncoderSet;
}

void MetalBackend::addOpEncoder(std::function<void(void)> opEncoder) {
    mOpEncoders.push_back(opEncoder);
}


id<MTLBuffer> MetalBackend::getHostBuffer(size_t size) const {
    return mRuntime->getHostBuffer(size);
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
    ((int *)shape.contents)[2] = z;
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
                {nil, nil, @"upcast_f_NHWC_to_NC4HW4", nil, nil},
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
                {nil, nil, @"downcast_f_NCHW_to_NC4HW4", nil, nil},
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
    mShapeH2D.reset(4 * sizeof(int));
    mShapeD2H.reset(4 * sizeof(int));
    mOpFullSupport = true;
    mFrameEncodeCache = false;
    mOpEncoderSet = false;
}

void MetalBackend::onResizeEnd() {
    auto ctx = (__bridge MNNMetalContext *)context();
    mFrameEncodeCache = (!ctx.isCommitEachShader && mOpFullSupport);
}

void MetalBackend::onCopyHostToDevice(const Tensor *src, const Tensor *dst) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (__bridge id<MTLBuffer>)(void *)dst->deviceId();
    auto floats  = src->getType().code == halide_type_float;

    // cast
    if (sfmt == dfmt || src->dimensions() <= 1) {
        if (floats) {
            NSUInteger size = src->elementSize();
            auto sizeC4 = UP_DIV(size, 4);
            auto host = this->getHostBuffer(sizeC4 * 4 * sizeof(float));
            [ctx wait];// make sure previous gpu task finished. for reuse mHostBuffer and mShapeH2D
            memcpy(host.contents, src->host<float>(), src->size());
            unsigned int limits[] = {
                (unsigned int)sizeC4,
                1,
                1,
                1
            };
            ::memcpy(mShapeH2D.buffer().contents, limits, sizeof(limits));
            auto encoder    = [ctx encoder];
            auto bandwidth  = [ctx load: @"downcast_float4" encoder:encoder];
            
            [encoder setBuffer:host offset:0 atIndex:0];
            [encoder setBuffer:device offset:0 atIndex:1];
            [encoder setBuffer:mShapeH2D.buffer() offset:0 atIndex:2];
            //[ctx dispatchEncoder:encoder threads:{sizeC4, 1, 1} bandwidth:bandwidth];
            std::pair<MTLSize, MTLSize> threads;
            threads.first = {sizeC4, 1, 1};
            threads.second = {bandwidth.maxThreadsPerThreadgroup, 1, 1};
            threads.second.width = threads.second.width <= threads.first.width ? threads.second.width : threads.first.width;
            threads.first.width = UP_DIV(threads.first.width, threads.second.width);
            [encoder dispatchThreadgroups:threads.first threadsPerThreadgroup:threads.second];
            [encoder endEncoding];
            [ctx commit];
            //[ctx wait];
        } else {
            [ctx wait];
            memcpy(device.contents, src->host<uint8_t>(), src->size());
            [ctx commit];
            //[ctx wait];
        }
    }
    // convert
    else {

        auto buffer = getHostBuffer(src->elementSize() * sizeof(float));
        [ctx wait];// make sure previous gpu task finished. for reuse mHostBuffer and mShapeH2D
        auto size = getTensorShape(mShapeH2D.buffer(), src);
        memcpy(buffer.contents, src->host<float>(), src->size());
        auto encoder = [ctx encoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Down);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto bandwidth = [ctx load:kernel encoder:encoder];
        
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBuffer:device offset:0 atIndex:1];
        [encoder setBuffer:mShapeH2D.buffer() offset:0 atIndex:2];
        [ctx dispatchEncoder:encoder threads:size bandwidth:bandwidth];
        [encoder endEncoding];
        [ctx commit];
        //[ctx wait];
    }
}

void MetalBackend::onCopyDeviceToHost(const Tensor *src, const Tensor *dst) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (__bridge id<MTLBuffer>)(void *)src->deviceId();
    auto floats  = src->getType().code == halide_type_float;

    // cast
    if (sfmt == dfmt || src->dimensions() <= 1) {
        if (floats) {
            auto eleSize = dst->elementSize();
            eleSize = UP_DIV(eleSize, 4) * 4;
            auto buffer = getHostBuffer(eleSize * dst->getType().bytes());

            NSUInteger size = src->elementSize();
            auto encoder    = [ctx encoder];
            auto bandwidth  = [ctx load: @"upcast_float4" encoder:encoder];
            [encoder setBuffer:device offset:0 atIndex:0];
            [encoder setBuffer:buffer offset:0 atIndex:1];
            auto sizeC4 = UP_DIV(size, 4);
            unsigned int limits[] = {
                (unsigned int)sizeC4,
                1,
                1,
                1
            };
            ::memcpy(mShapeD2H.buffer().contents, limits, sizeof(limits));
            [encoder setBuffer:mShapeD2H.buffer() offset:0 atIndex:2];
            //[ctx dispatchEncoder:encoder threads:{sizeC4, 1, 1} bandwidth:bandwidth];
            std::pair<MTLSize, MTLSize> threads;
            threads.first = {sizeC4, 1, 1};
            threads.second = {bandwidth.maxThreadsPerThreadgroup, 1, 1};
            threads.second.width = threads.second.width <= threads.first.width ? threads.second.width : threads.first.width;
            threads.first.width = UP_DIV(threads.first.width, threads.second.width);
            [encoder dispatchThreadgroups:threads.first threadsPerThreadgroup:threads.second];
            
            [encoder endEncoding];
            [ctx commit];
            [ctx wait];

            memcpy(dst->host<float>(), buffer.contents, dst->size());
        } else {
            [ctx commit];
            [ctx wait];
            memcpy(dst->host<uint8_t>(), device.contents, dst->size());
        }
    }
    // convert
    else {
        auto size = getTensorShape(mShapeD2H.buffer(), src);
        auto buffer  = getHostBuffer(dst->size());
        auto encoder = [ctx encoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Up);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto bandwidth = [ctx load:kernel encoder:encoder];
        [encoder setBuffer:device offset:0 atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:1];
        [encoder setBuffer:mShapeD2H.buffer() offset:0 atIndex:2];
        [ctx dispatchEncoder:encoder threads:size bandwidth:bandwidth];
        [encoder endEncoding];
        [ctx commit];
        [ctx wait];
        memcpy(dst->host<float>(), buffer.contents, dst->size());
    }
}

MetalBackend::AutoBuffer::~AutoBuffer() {
    if (nil != mBuffer) {
        mRuntime->mStatic->release(mBuffer);
    }
    mBuffer = nil;
}
void MetalBackend::AutoBuffer::reset(size_t length) {
    if (nil != mBuffer) {
        mRuntime->mStatic->release(mBuffer);
        mBuffer = nil;
    }
    mBuffer = mRuntime->mStatic->alloc(length);
}


void MetalBackend::onCopyDeviceToDevice(const Tensor *src, const Tensor *dst,
                                        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const {
    auto ctx    = (__bridge MNNMetalContext *)context();
    auto standalone = encoder == nil;
    encoder         = encoder ?: [ctx encoder];
    auto sfmt       = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt       = TensorUtils::getDescribe(dst)->dimensionFormat;

    // copy
    if (sfmt == dfmt || src->dimensions() <= 1) {
        auto flt       = dst->getType().code == halide_type_float;
        auto size      = flt ? dst->elementSize() : dst->size();
        auto bandwidth = [ctx load:flt ? @"copy_float" : @"copy_byte" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)src->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)dst->deviceId() offset:0 atIndex:1];
        [ctx dispatchEncoder:encoder threads:{(NSUInteger)size, 1, 1} bandwidth:bandwidth];
    }
    // convert
    else {
        auto kernel = kernelForConvert(src->getType(), sfmt, dfmt, None);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt
        bool needRelease = false;
        if (shape == nil) {
            shape = mRuntime->mStatic->alloc(4 * sizeof(int));
            needRelease = true;
        }

        auto size     = getTensorShape(shape, src);
        auto bandwidth = [ctx load:kernel encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(src->buffer().device) offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(dst->buffer().device) offset:0 atIndex:1];
        [encoder setBuffer:shape offset:0 atIndex:2];
        [ctx dispatchEncoder:encoder threads:size bandwidth:bandwidth];
        if (needRelease) {
            mRuntime->mStatic->release(shape);
        }
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
        [ctx commit_net];
    }

    //printf("%d!!!!\n", mFrameEncodeCache);
    onCopyBuffer(src, dst, nil, nil);
}

id<MTLComputeCommandEncoder> MetalBackend::encoder() const {
    if (nil == mComputeEncoder) {
        auto ctx = (__bridge MNNMetalContext *)context();
        mComputeEncoder = [ctx encoder_net];//TO DO :: use which cmdBuffer
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
MetalRuntime::MetalRuntime() {
    mContext = (__bridge_retained void *)[[MNNMetalContext alloc] init];
    mStatic.reset(new BufferAllocator(mContext));
    mDynamic.reset(new BufferAllocator(mContext));
}
MetalRuntime::~ MetalRuntime() {
    CFRelease(mContext);
}
Backend* MetalRuntime::onCreate(const BackendConfig* config) const {
    return new MetalBackend(this);
}
void MetalRuntime::onGabageCollect(int level) {
    if (level > 50) {
        mDynamic->clear();
    }
    mStatic->clear();
}

class MetalRuntimeCreator : public RuntimeCreator {
    virtual Runtime *onCreate(const Backend::Info &info) const {
        static std::once_flag s_flag;
        std::call_once(s_flag, [&]() { registerMetalOps(); });
        auto rt = new MetalRuntime;
        if (nullptr == rt->context()) {
            return nullptr;
        }
        auto ctx = (__bridge MNNMetalContext *)rt->context();
        // according to
        // https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/HardwareGPUInformation/HardwareGPUInformation.html
        // not all device with iOS 8+ supports metal.
        if (nullptr == ctx.device) {
            return nullptr;
        }
        return rt;
    }
};

void registerMetalRuntimeCreator() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_METAL, new MetalRuntimeCreator, true);
}

#ifndef MNN_CODEGEN_REGISTER
static const auto __metal_global_initializer = []() {
    registerMetalRuntimeCreator();
    return true;
}();
#endif
#else
void registerMetalRuntimeCreator() {
    // Do nothing
}
#endif
} // namespace MNN
