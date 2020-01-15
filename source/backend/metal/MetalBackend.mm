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

#if MNN_METAL_ENABLED

namespace MNN {

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

MetalBackend::MetalBackend() : Backend(MNN_FORWARD_METAL) {
    mContext = (__bridge_retained void *)[[MNNMetalContext alloc] init];
}
MetalBackend::~MetalBackend() {
    for (auto t : mStaticBuffers) {
        CFRelease(t.first);
    }
    mStaticBuffers.clear();
    onClearBuffer();
    CFRelease(mContext);
}
void *MetalBackend::context() {
    return mContext;
}

bool MetalBackend::onAcquireBuffer(const Tensor *_tensor, StorageType storageType) {
    auto context = (__bridge MNNMetalContext *)mContext;
    auto tensor  = const_cast<Tensor *>(_tensor);
    auto size    = tensor->size();
    if (0 == size) {
        return false;
    }

    // use metal_float when meets float
    if (halide_type_float == tensor->buffer().type.code && tensor->buffer().type.bits == 32) {
        size /= sizeof(float) / sizeof(metal_float);
    }

    // reuse if possible
    switch (storageType) {
        case Backend::STATIC: {
            // do not reuse
        } break;
        case Backend::DYNAMIC: {
            auto iter = mReusableBuffers.lower_bound(size);
            if (iter != mReusableBuffers.end()) {
                tensor->buffer().device = iter->second;
                mDynamicBuffers.insert(std::make_pair((void*)iter->second, iter->first));
                mReusableBuffers.erase(iter);
                return true;
            }
        } break;
        case Backend::DYNAMIC_SEPERATE: {
            // do not reuse
        } break;
    }

    // create new
    auto buffer = (__bridge_retained void *)[context newDeviceBuffer:size access:CPUWriteOnly];
    switch (storageType) {
            case Backend::STATIC: {
                mStaticBuffers.insert(std::make_pair(buffer, size));
            } break;
            case Backend::DYNAMIC: {
                mDynamicBuffers.insert(std::make_pair(buffer, size));
            } break;
            case Backend::DYNAMIC_SEPERATE: {
                mSeparatedBuffers.insert(std::make_pair(buffer, size));
            } break;
    }
    tensor->buffer().device = (uint64_t)buffer;
    return true;
}
bool MetalBackend::onReleaseBuffer(const Tensor *tensor, StorageType storageType) {
    auto buffer = tensor->buffer().device;
    if (buffer) {
        switch (storageType) {
            case Backend::STATIC: {
                auto iter = mStaticBuffers.find((void *)buffer);
                if (iter != mStaticBuffers.end()) {
                    mStaticBuffers.erase(iter);
                    CFRelease((void *)buffer);
                }
            } break;
            case Backend::DYNAMIC: {
                auto iter = mDynamicBuffers.find((void *)buffer);
                if (iter != mDynamicBuffers.end()) {
                    mReusableBuffers.insert(std::make_pair(iter->second, buffer));
                    mDynamicBuffers.erase(iter);
                }
            } break;
            case Backend::DYNAMIC_SEPERATE: {
                // do nothing
            } break;
        }
    }
    return true;
}
bool MetalBackend::onAllocateBuffer() {
    return true;
}
bool MetalBackend::onClearBuffer() {
    for (auto t : mDynamicBuffers) {
        CFRelease(t.first);
    }
    mDynamicBuffers.clear();
    for (auto t : mSeparatedBuffers) {
        CFRelease(t.first);
    }
    mSeparatedBuffers.clear();
    mReusableBuffers.clear();
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
    // according to
    // https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/HardwareGPUInformation/HardwareGPUInformation.html
    // not all device with iOS 8+ supports metal.
    auto context = (__bridge MNNMetalContext *)mContext;
    if (!context.device) {
        MNN_PRINT("Metal is not supported on this device.");
        return NULL;
    }

    auto map  = getCreatorMap();
    auto iter = map->find(op->type());
    if (iter == map->end()) {
        return NULL;
    }
    auto exe = iter->second->onCreate(inputs, op, this);
    if (NULL == exe) {
        MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name() ? op->name()->c_str() : "");
        return NULL;
    }
    return exe;
}
void MetalBackend::onExecuteBegin() const {
    // do nothing
}
void MetalBackend::onExecuteEnd() const {
    auto context = (__bridge MNNMetalContext *)mContext;
    [context commit];
}
bool MetalBackend::onWaitFinish() {
    auto context = (__bridge MNNMetalContext *)mContext;
    [context commit];
    [context wait];
    return true;
}

id<MTLBuffer> MetalBackend::getHostBuffer(size_t size) const {
    // reuse
    if (mHostBuffer.length >= size)
        return mHostBuffer;

    // create larger
    auto context = (__bridge MNNMetalContext *)mContext;
    mHostBuffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return mHostBuffer;
}

std::tuple<id<MTLBuffer>, MTLSize> getTensorShape(MNNMetalContext *context, const Tensor *tensor) {
    int s = 0, c = 0, b = 0;
    if (tensor->dimensions() == 4) {
        s = tensor->width() * tensor->height();
        c = tensor->channel();
        b = tensor->batch();
    } else if (tensor->dimensions() == 3) {
        s = tensor->length(2);
        c = tensor->length(1);
        b = tensor->length(0);
    } else if (tensor->dimensions() == 2) {
        s = 1;
        c = tensor->length(1);
        b = tensor->length(0);
    }
    int z = UP_DIV(c, 4);

    // shape
    auto shape                 = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = s;
    ((int *)shape.contents)[1] = c;
    ((int *)shape.contents)[2] = z;
    ((int *)shape.contents)[3] = b * z;

    // threads
    MTLSize threads = {(NSUInteger)s, (NSUInteger)b * z, 1};
    return std::make_tuple(shape, threads);
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

void MetalBackend::onCopyHostToDevice(const Tensor *src, const Tensor *dst) const {
    auto context = (__bridge MNNMetalContext *)mContext;
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (__bridge id<MTLBuffer>)(void *)dst->deviceId();
    auto floats  = src->getType().code == halide_type_float;

    // cast
    if (sfmt == dfmt || src->dimensions() <= 1) {
        if (floats) {
            auto host = this->getHostBuffer(src->elementSize() * sizeof(float));
            memcpy(host.contents, src->host<float>(), src->size());

            NSUInteger size = src->elementSize();
            auto simd       = size % 4 == 0;
            auto encoder    = [context encoder];
            auto bandwidth  = [context load:simd ? @"downcast_float4" : @"downcast_float" encoder:encoder];
            [encoder setBuffer:host offset:0 atIndex:0];
            [encoder setBuffer:device offset:0 atIndex:1];
            [context dispatchEncoder:encoder threads:{simd ? size / 4 : size, 1, 1} bandwidth:bandwidth];
            [encoder endEncoding];
            [context commit];
            [context wait];
        } else {
            [context commit];
            [context wait];
            memcpy(device.contents, src->host<uint8_t>(), src->size());
        }
    }
    // convert
    else {
        auto shape  = getTensorShape(context, src);
        auto buffer = getHostBuffer(src->elementSize() * sizeof(float));
        memcpy(buffer.contents, src->host<float>(), src->size());
        auto encoder = [context encoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Down);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto bandwidth = [context load:kernel encoder:encoder];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBuffer:device offset:0 atIndex:1];
        [encoder setBuffer:std::get<0>(shape) offset:0 atIndex:2];
        [context dispatchEncoder:encoder threads:std::get<1>(shape) bandwidth:bandwidth];
        [encoder endEncoding];
        [context commit];
        [context wait];
    }
}

void MetalBackend::onCopyDeviceToHost(const Tensor *src, const Tensor *dst) const {
    auto context = (__bridge MNNMetalContext *)mContext;
    auto sfmt    = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(dst)->dimensionFormat;
    auto device  = (__bridge id<MTLBuffer>)(void *)src->deviceId();
    auto floats  = src->getType().code == halide_type_float;

    // cast
    if (sfmt == dfmt || src->dimensions() <= 1) {
        if (floats) {
            auto buffer = getHostBuffer(dst->size());

            NSUInteger size = src->elementSize();
            auto simd       = size % 4 == 0;
            auto encoder    = [context encoder];
            auto bandwidth  = [context load:simd ? @"upcast_float4" : @"upcast_float" encoder:encoder];
            [encoder setBuffer:device offset:0 atIndex:0];
            [encoder setBuffer:buffer offset:0 atIndex:1];
            [context dispatchEncoder:encoder threads:{simd ? size / 4 : size, 1, 1} bandwidth:bandwidth];
            [encoder endEncoding];
            [context commit];
            [context wait];

            memcpy(dst->host<float>(), buffer.contents, dst->size());
        } else {
            [context commit];
            [context wait];
            memcpy(dst->host<uint8_t>(), device.contents, dst->size());
        }
    }
    // convert
    else {
        auto shape   = getTensorShape(context, src);
        auto buffer  = getHostBuffer(dst->size());
        auto encoder = [context encoder];
        auto kernel  = kernelForConvert(src->getType(), sfmt, dfmt, Up);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto bandwidth = [context load:kernel encoder:encoder];
        [encoder setBuffer:device offset:0 atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:1];
        [encoder setBuffer:std::get<0>(shape) offset:0 atIndex:2];
        [context dispatchEncoder:encoder threads:std::get<1>(shape) bandwidth:bandwidth];
        [encoder endEncoding];
        [context commit];
        [context wait];
        memcpy(dst->host<float>(), buffer.contents, dst->size());
    }
}

void MetalBackend::onCopyDeviceToDevice(const Tensor *src, const Tensor *dst,
                                        id<MTLComputeCommandEncoder> encoder) const {
    auto context    = (__bridge MNNMetalContext *)mContext;
    auto standalone = encoder == nil;
    encoder         = encoder ?: [context encoder];
    auto sfmt       = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt       = TensorUtils::getDescribe(dst)->dimensionFormat;

    // copy
    if (sfmt == dfmt || src->dimensions() <= 1) {
        auto flt       = dst->getType().code == halide_type_float;
        auto size      = flt ? dst->elementSize() : dst->size();
        auto bandwidth = [context load:flt ? @"copy_float" : @"copy_byte" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)src->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)dst->deviceId() offset:0 atIndex:1];
        [context dispatchEncoder:encoder threads:{(NSUInteger)size, 1, 1} bandwidth:bandwidth];
    }
    // convert
    else {
        auto kernel = kernelForConvert(src->getType(), sfmt, dfmt, None);
        MNN_ASSERT(kernel != nil); // unsupported sfmt to dfmt

        auto shape     = getTensorShape(context, src);
        auto bandwidth = [context load:kernel encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(src->buffer().device) offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(dst->buffer().device) offset:0 atIndex:1];
        [encoder setBuffer:std::get<0>(shape) offset:0 atIndex:2];
        [context dispatchEncoder:encoder threads:std::get<1>(shape) bandwidth:bandwidth];
    }

    if (standalone) {
        [encoder endEncoding];
        MNN_PRINT_ENCODER(context, encoder);
    }
}

void MetalBackend::onCopyBuffer(const Tensor *src, const Tensor *dst) const {
    onCopyBuffer(src, dst, nil);
}

void MetalBackend::onCopyBuffer(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder) const {
    MNN_ASSERT(src->buffer().dimensions == dst->buffer().dimensions);

    if (!src->buffer().host && !dst->buffer().host) {
        onCopyDeviceToDevice(src, dst, encoder);
    } else if (!src->buffer().host && dst->buffer().host) {
        onCopyDeviceToHost(src, dst);
    } else if (src->buffer().host && !dst->buffer().host) {
        onCopyHostToDevice(src, dst);
    } else {
        MNN_ASSERT(false); // should not be handled here
    }
}


class MetalBackendCreator : public BackendCreator {
    virtual Backend *onCreate(const Backend::Info &info) const {
        static std::once_flag s_flag;
        std::call_once(s_flag, [&]() { registerMetalOps(); });
        auto bn = new MetalBackend;
        if (nullptr == bn->context()) {
            return nullptr;
        }
        return bn;
    }
};

void registerMetalBackendCreator() {
    MNNInsertExtraBackendCreator(MNN_FORWARD_METAL, new MetalBackendCreator, true);
}
} // namespace MNN
#else
namespace MNN {
void registerMetalBackendCreator() {
}
}
#endif /* MNN_METAL_ENABLED */
