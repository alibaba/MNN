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

MetalBackend::MetalBackend(std::shared_ptr<BufferAllocator> staticMem, const MetalRuntime* runtime) : Backend(MNN_FORWARD_METAL) {
    mRuntime = runtime;
    mBufferPool.reset(new BufferAllocator(BufferAllocator::Allocator::createRecurse(staticMem.get()), 1024));
    mStaticBufferPool = staticMem;
}
MetalBackend::~MetalBackend() {
    // Do nothing
}
void *MetalBackend::context() const {
    return mRuntime->context();
}

class MetalMemRelease : public Backend::MemObj {
public:
    MetalMemRelease(std::pair<void*, int> buffer, BufferAllocator* allocator) {
        mBuffer = buffer;
        mAllocator = allocator;
    }
    virtual ~ MetalMemRelease() {
        mAllocator->free(mBuffer);
    }
private:
    std::pair<void*, int> mBuffer;
    BufferAllocator* mAllocator;
};
Backend::MemObj* MetalBackend::onAcquire(const Tensor *_tensor, StorageType storageType) {
    auto tensor  = const_cast<Tensor *>(_tensor);
    auto width    = tensor->width();
    auto height   = tensor->height();
    auto batch    = tensor->batch();
    auto channel  = tensor->channel();

    auto size = batch * ROUND_UP(channel, 16) * ROUND_UP(height, 4) * ROUND_UP(width, 4);
    if (0 == size || tensor->dimensions() > 4) {
        size = ROUND_UP(tensor->elementSize(), 4);
    }
    if (0 == size) {
        return nullptr;
    }
    
    // use metal_float when meets float
    if (halide_type_float == tensor->buffer().type.code && tensor->buffer().type.bits == 32) {
        size*= sizeof(metal_float);
    } else {
        size *= tensor->getType().bytes();
    }

    // reuse if possible
    std::pair<void*, int> buffer;
    BufferAllocator* allocator = nullptr;
    switch (storageType) {
        case Backend::STATIC: {
            buffer = mStaticBufferPool->alloc(size, false);
            allocator = mStaticBufferPool.get();
        } break;
        case Backend::DYNAMIC: {
            buffer = mBufferPool->alloc(size, false);
            allocator = mBufferPool.get();
        } break;
        case Backend::DYNAMIC_SEPERATE: {
            buffer = mBufferPool->alloc(size, true);
            allocator = mBufferPool.get();
        } break;
    }
    
    if(nullptr == buffer.first) {
        MNN_ERROR("onAcquireBuffer error!\n");
        return nullptr;
    };
    
    // ptr of MetalBufferAlloc
    auto host = buffer.first;
    ((Tensor*)_tensor)->buffer().device = (uint64_t)host;
    auto des = TensorUtils::getDescribe(_tensor);
    des->extra.offset = buffer.second;
    
    return new MetalMemRelease(buffer, allocator);
}

bool MetalBackend::onClearBuffer() {
    mBufferPool->release(true);
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
    mEncoderCount = 0;
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
    auto context = (__bridge MNNMetalContext *)this->context();
    auto buffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return buffer;
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
    mShapeH2D = getConstBuffer(4 * sizeof(int));
    mShapeD2H = getConstBuffer(4 * sizeof(int));
    mOpFullSupport = true;
    mFrameEncodeCache = false;
    mOpEncoderSet = false;
    mOpEncoders.clear();
    
    // Finish last inference task if needed
    flushEncoder();
    auto ctx = (__bridge MNNMetalContext *)context();
    [ctx commit_net];
    [ctx wait];
}

void MetalBackend::onResizeEnd() {
    auto ctx = (__bridge MNNMetalContext *)context();
    mFrameEncodeCache = (!ctx.isCommitEachShader && mOpFullSupport);
}

void MetalBackend::onCopyHostToDevice(const Tensor *src, const Tensor *dst) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *) (dst->deviceId()))->getBuffer();
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
            ::memcpy(mShapeH2D.contents, limits, sizeof(limits));
            auto encoder    = [ctx encoder];
            auto bandwidth  = [ctx load: @"downcast_float4" encoder:encoder];
            
            [encoder setBuffer:host offset:0 atIndex:0];
            [encoder setBuffer:device offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
            [encoder setBuffer:mShapeH2D offset:0 atIndex:2];
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
        auto size = getTensorShape(mShapeH2D, src);
        memcpy(buffer.contents, src->host<float>(), src->size());
        auto encoder = [ctx encoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Down);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto bandwidth = [ctx load:kernel encoder:encoder];
        
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBuffer:device offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
        [encoder setBuffer:mShapeH2D offset:0 atIndex:2];
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
    auto device  = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId())->getBuffer();
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
        auto size = getTensorShape(mShapeD2H, src);
        auto buffer  = getHostBuffer(dst->size());
        auto encoder = [ctx encoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Up);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto bandwidth = [ctx load:kernel encoder:encoder];
        [encoder setBuffer:device offset:TensorUtils::getDescribe(src)->extra.offset atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:1];
        [encoder setBuffer:mShapeD2H offset:0 atIndex:2];
        [ctx dispatchEncoder:encoder threads:size bandwidth:bandwidth];
        [encoder endEncoding];
        [ctx commit];
        [ctx wait];
        memcpy(dst->host<float>(), buffer.contents, dst->size());
    }
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
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId())->getBuffer() offset:TensorUtils::getDescribe(src)->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)dst->deviceId())->getBuffer() offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
        [ctx dispatchEncoder:encoder threads:{(NSUInteger)size, 1, 1} bandwidth:bandwidth];
    }
    // convert
    else {
        auto kernel = kernelForConvert(src->getType(), sfmt, dfmt, None);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt
        if (shape == nil) {
            shape = getConstBuffer(4 * sizeof(int));
        }

        auto size     = getTensorShape(shape, src);
        auto bandwidth = [ctx load:kernel encoder:encoder];
        [encoder setBuffer:( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(src->buffer().device))->getBuffer() offset:TensorUtils::getDescribe(src)->extra.offset atIndex:0];
        [encoder setBuffer:( id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)(dst->buffer().device))->getBuffer() offset:TensorUtils::getDescribe(dst)->extra.offset atIndex:1];
        [encoder setBuffer:shape offset:0 atIndex:2];
        [ctx dispatchEncoder:encoder threads:size bandwidth:bandwidth];
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

MetalRuntime::MetalRuntime(const Backend::Info info) {
    mContext = (__bridge_retained void *)[[MNNMetalContext alloc] init];
    if(nullptr == mContext) {
        mIsCreateError = true;
    } else {
        auto ctx = (__bridge MNNMetalContext *)mContext;
        // according to
        // https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/HardwareGPUInformation/HardwareGPUInformation.html
        // not all device with iOS 8+ supports metal.
        if (nullptr == ctx.device) {
            mIsCreateError = true;
        }
    }
    setGpuMode(info.gpuMode);
}

MetalRuntime::~ MetalRuntime() {
    if(mContext) {
        CFRelease(mContext);
    }
}


bool MetalRuntime::setCache(std::pair<const void*, size_t> cache) {//Get Cache
    auto buffer = cache.first;
    auto size   = cache.second;
    if (nullptr == buffer) {
        mCacheOutside = nullptr;
        mCacheOutsideSize = 0;
        mBuffer.clear();
        return true;//actually get nothing
    }
    mCacheOutsideSize = size;
    mCacheOutside = buffer;
    auto cacheBuffer = GetCache(buffer);

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

std::pair<const void*, size_t> MetalRuntime::makeCache() {//make Cache
    if (nullptr != mCacheOutside) {
        return std::make_pair(mCacheOutside, mCacheOutsideSize);
    }
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

    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Cache::Pack(builder, cache.get());
    builder.Finish(lastOffset);
    mBuffer.resize(builder.GetSize());
    ::memcpy(mBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
    return std::make_pair(mBuffer.data(), mBuffer.size());
}

MetalRuntimeWrapper::MetalRuntimeWrapper(const Backend::Info info) {
    mMetalRuntime.reset(new MetalRuntime(info));

    if (mMetalRuntime.get()) {
        if (mMetalRuntime->isCreateError() == true) {
            mIsCreateError = true;
            return;
        }
        
        std::shared_ptr<BufferAllocator::Allocator> allocator(new MetalRuntimeAllocator(mMetalRuntime.get()));
        mBufferPool.reset(new BufferAllocator(allocator));
    }
}
MetalRuntimeWrapper::~MetalRuntimeWrapper() {
    // Do nothing
}

float MetalRuntimeWrapper::onGetMemoryInMB() {
    auto staticMemoryInMB = mBufferPool->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

Backend* MetalRuntimeWrapper::onCreate(const BackendConfig* config) const {
    return new MetalBackend(mBufferPool, mMetalRuntime.get());
}

void MetalRuntimeWrapper::onGabageCollect(int level) {
    mBufferPool->release(false);
}

std::pair<const void*, size_t> MetalRuntimeWrapper::onGetCache() {//make Cache
    return mMetalRuntime->makeCache();
}

bool MetalRuntimeWrapper::onSetCache(const void* buffer, size_t size) {//set Cache
    return mMetalRuntime->setCache(std::make_pair(buffer, size));
}

std::pair<void*, int> MetalRuntimeAllocator::onAlloc(int size, int align) {
    auto context = (__bridge MNNMetalContext *)mMetalRuntime->context();
    mBuffer = [context newDeviceBuffer:size access:CPUWriteOnly];
    auto mMetalBufferAlloc = (new MetalBufferAlloc(mBuffer));
    return std::make_pair((void *)mMetalBufferAlloc, 0);
}
void MetalRuntimeAllocator::onRelease(std::pair<void*, int> ptr) {
    delete (MetalBufferAlloc *)ptr.first;
}


class MetalRuntimeCreator : public RuntimeCreator {
    virtual Runtime *onCreate(const Backend::Info &info) const {
        static std::once_flag s_flag;
        std::call_once(s_flag, [&]() { registerMetalOps(); });
        
        auto rt = new MetalRuntimeWrapper(info);
        if (rt != nullptr) {
            if (!rt->isCreateError()) {
                return rt;
            } else {
                delete rt;
            }
        }
        return nullptr;
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
