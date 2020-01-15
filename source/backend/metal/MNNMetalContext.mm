//
//  MNNMetalContext.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"

#if MNN_METAL_ENABLED

using namespace MNN;

@interface MNNMetalContext ()
// public
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (assign, nonatomic) NSUInteger maxThreadgroupMemoryLength;
// private
@property (strong, nonatomic) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *caches;
@property (strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>> *waitings;
@property (strong, nonatomic) id<MTLLibrary> library;
@property (strong, nonatomic) id<MTLHeap> sharedHeap NS_AVAILABLE_IOS(10.0);
@property (strong, nonatomic) id<MTLHeap> privateHeap NS_AVAILABLE_IOS(10.0);
@end

@implementation MNNMetalContext

+ (id<MTLDevice>)device {
    static id<MTLDevice> device = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        device = MTLCreateSystemDefaultDevice();
    });
    return device;
}

+ (id<MTLLibrary>)library {
    static id<MTLLibrary> library = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
#if TARGET_OS_IOS
        NSString *path = [[NSBundle bundleForClass:[MNNMetalContext class]] pathForResource:@"mnn" ofType:@"metallib"];
#else
        NSString *path = @"mnn.metallib";
#endif
        library = path ? [self.device newLibraryWithFile:path error:NULL] : [self.device newDefaultLibrary];
        if (nil == library) {
            MNN_ERROR("Can't load mnn.metallib\n");
        }
    });
    return library;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        // public
        _device        = self.class.device;
        _commandQueue  = [_device newCommandQueue];
        _commandBuffer = [_commandQueue commandBuffer];
        if (@available(iOS 11.0, *)) {
            _maxThreadgroupMemoryLength = _device.maxThreadgroupMemoryLength;
        } else {
            _maxThreadgroupMemoryLength = 16352; // 16352(16k - 32b) on iOS 11- according to feature set doc
        }

        // private
        _caches   = [NSMutableDictionary dictionary];
        _waitings = [NSMutableArray array];
        _library  = self.class.library;
        if (nil == _library) {
            return nil;
        }

        if (@available(iOS 10.0, *)) {
            MTLHeapDescriptor *shared = [[MTLHeapDescriptor alloc] init];
            shared.storageMode        = MTLStorageModeShared;
            shared.size               = 0x1000; // initial size
            _sharedHeap               = [_device newHeapWithDescriptor:shared];

            MTLHeapDescriptor *priv = [[MTLHeapDescriptor alloc] init];
            priv.storageMode        = MTLStorageModePrivate;
            priv.size               = 0x0800; // initial size
            _privateHeap            = [_device newHeapWithDescriptor:priv];
        }
    }
    return self;
}

#pragma mark device
- (MTLResourceOptions)optionForAccess:(MNN::MetalAccess)access {
    if (access == CPUWriteOnly) {
        return MTLResourceOptionCPUCacheModeWriteCombined;
    } else if (access == CPUTransparent) {
        if (@available(iOS 9.0, *)) {
            return MTLResourceStorageModePrivate;
        } else {
            return MTLResourceOptionCPUCacheModeDefault;
        }
    } else { // access == CPUReadWrite
        return MTLResourceOptionCPUCacheModeDefault;
    }
}

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size access:(MNN::MetalAccess)access {
    return [_device newBufferWithLength:size options:[self optionForAccess:access]];
}

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size bytes:(const void *)bytes access:(MNN::MetalAccess)access {
    return [_device newBufferWithBytes:bytes length:size options:[self optionForAccess:access]];
}

#pragma mark heap
- (id<MTLBuffer>)newHeapBuffer:(NSUInteger)size access:(MNN::MetalAccess)access {
    MTLResourceOptions options = [self optionForAccess:access];
    if (@available(iOS 10.0, *)) {
        id<MTLHeap> heap = access == CPUTransparent ? _privateHeap : _sharedHeap;
        if (size <= [heap maxAvailableSizeWithAlignment:1]) {
            id<MTLBuffer> buffer = [heap newBufferWithLength:size options:options];
            if (buffer)
                return buffer;
        }
    }
    return [_device newBufferWithLength:size options:options];
}

- (id<MTLBuffer>)newHeapBuffer:(NSUInteger)size bytes:(const void *)bytes access:(MNN::MetalAccess)access {
    MNN_ASSERT(access != CPUReadWrite);
    MTLResourceOptions options = [self optionForAccess:access];
    if (@available(iOS 10.0, *)) {
        id<MTLHeap> heap = access == CPUTransparent ? _privateHeap : _sharedHeap;
        if (size <= [heap maxAvailableSizeWithAlignment:1]) {
            id<MTLBuffer> buffer = [heap newBufferWithLength:size options:options];
            if (buffer) {
                memcpy(buffer.contents, bytes, size);
                return buffer;
            }
        }
    }
    return [_device newBufferWithBytes:bytes length:size options:options];
}

- (void)releaseHeapBuffer:(id<MTLBuffer>)buffer {
    if (@available(iOS 10.0, *)) {
        if (buffer.heap)
            [buffer makeAliasable];
    }
}

#pragma mark enqueue
- (id<MTLFunction>)functionWithName:(NSString *)name {
    if (!name)
        return nil;
    id<MTLFunction> result = [_library newFunctionWithName:name];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    if (@available(iOS 10.0, *))
        result.label = name;
#endif
    return result;
}

- (id<MTLComputePipelineState>)pipelineWithName:(NSString *)name {
    id<MTLComputePipelineState> result = _caches[name];
    if (result)
        return result;

    id<MTLFunction> function = [self functionWithName:name];
    if (!function)
        return nil;

    NSError *error = nil;
    result         = [_device newComputePipelineStateWithFunction:function error:&error];
#if MNN_METAL_DEBUG
    if (error)
        printf("[METAL] create pipeline error: %s\n", error.localizedDescription.UTF8String);
#endif
    if (result)
        _caches[name] = result;
    return result;
}

- (id<MTLComputeCommandEncoder>)encoder {
    id<MTLComputeCommandEncoder> result = [_commandBuffer computeCommandEncoder];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    result.label = nil;
#endif
    return result;
}

- (MetalBandwidth)load:(NSString *)name encoder:(id<MTLComputeCommandEncoder>)encoder {
    id<MTLComputePipelineState> pipeline = [self pipelineWithName:name];
    [encoder setComputePipelineState:pipeline];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    if (!name) {
    } else if (!encoder.label) {
        encoder.label = name;
    } else {
        NSArray *components = [encoder.label componentsSeparatedByString:@","];
        if (![components containsObject:name]) {
            components = [components arrayByAddingObject:name];
        }
        encoder.label = [components componentsJoinedByString:@","];
    }
#endif
    return {pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup, NO};
}

#pragma mark dispatch
- (void)commit {
    if (_commandBuffer.status < MTLCommandBufferStatusCommitted) {
        [_commandBuffer commit];
        [_waitings addObject:_commandBuffer];
        _commandBuffer = [_commandQueue commandBuffer]; // create a new command buffer
    }
}

- (void)wait {
    NSArray *buffers = _waitings.copy;
    [_waitings removeAllObjects];

    for (id<MTLCommandBuffer> buffer in buffers) {
        if (buffer.status >= MTLCommandBufferStatusCompleted)
            continue;

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
}

static NSUInteger smallest_log2(NSUInteger integer) {
    if (integer == 0)
        return 0;
    NSUInteger power = 0;
    while ((integer & 0b1) == 0) {
        integer = integer >> 1;
        power++;
    }
    return power;
}

- (MTLSize)threadsPerGroupWithThreads:(MTLSize)t bandwidth:(MetalBandwidth)bw {
    auto pwarp = smallest_log2(bw.threadExecutionWidth);
    auto px = smallest_log2(t.width), sx = (NSUInteger)ceil(log2(t.width));
    auto py = smallest_log2(t.height), sy = (NSUInteger)ceil(log2(t.height));

    // accurately match on x
    if (px >= pwarp) {
        return {bw.threadExecutionWidth, 1, 1};
    }
    // accurately match on xy
    else if (px + py >= pwarp && sx < pwarp / 2) {
        NSUInteger x = pow(2, px);
        return {x, bw.threadExecutionWidth / x, 1};
    }
    // similarly match on x
    else if (sx >= pwarp) {
        return {bw.threadExecutionWidth, 1, 1};
    }
    // similarly match on xy
    else if (sx + sy >= pwarp) {
        NSUInteger x = pow(2, sx);
        return {x, bw.threadExecutionWidth / x, 1};
    }

    // on xyz (for most shaders do not protect gid.z, z axis must be accurately match)
    auto pz = smallest_log2(t.depth);
    auto sz = bw.zAxisProtected ? ceil(log2(t.depth)) : pz;
    if (px + py + pz >= pwarp) {
        NSUInteger x = pow(2, px), y = pow(2, py);
        return {x, y, bw.threadExecutionWidth / x / y};
    } else if (sx + sy + sz >= pwarp) {
        NSUInteger x = pow(2, sx), z = pow(2, MIN(sz, pwarp - sx));
        return {x, bw.threadExecutionWidth / x / z, z};
    } else {
        NSUInteger z = pow(2, sz);
        return {t.width, t.height, z};
    }
}

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                threads:(MTLSize)threads
              bandwidth:(MetalBandwidth)bandwidth {
    [self dispatchEncoder:encoder
                  threads:threads
          threadsPerGroup:[self threadsPerGroupWithThreads:threads bandwidth:bandwidth]
                bandwidth:bandwidth];
}

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                threads:(MTLSize)threads
        threadsPerGroup:(MTLSize)threadsPerGroup
              bandwidth:(MetalBandwidth)bandwidth {
#if MNN_METAL_DEBUG
    if (threads.width == 0 || threads.height == 0 || threads.depth == 0 || threadsPerGroup.width == 0 ||
        threadsPerGroup.height == 0 || threadsPerGroup.depth == 0) {
        printf("[METAL] dispatch error %td %td %td / %td %td %td\n", threads.width, threads.height, threads.depth,
               threadsPerGroup.width, threadsPerGroup.height, threadsPerGroup.depth);
    }
#endif

    //    NSLog(@"dispatch {%td %td %td} with {%td %td %td}",
    //          threads.width, threads.height, threads.depth,
    //          threadsPerGroup.width, threadsPerGroup.height, threadsPerGroup.depth);
    threadsPerGroup.width  = MIN(threadsPerGroup.width, bandwidth.maxThreadsPerThreadgroup);
    threadsPerGroup.height = MIN(threadsPerGroup.height, bandwidth.maxThreadsPerThreadgroup);
    threadsPerGroup.depth  = MIN(threadsPerGroup.depth, bandwidth.maxThreadsPerThreadgroup);
#ifdef MNN_BUILD_FOR_IOS
    if (@available(iOS 11.0, *)) {
        if ([_device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1]) {
            [encoder dispatchThreads:threads threadsPerThreadgroup:threadsPerGroup];
            return;
        }
    }
#endif
    MTLSize groups = {
        UP_DIV(threads.width, threadsPerGroup.width), UP_DIV(threads.height, threadsPerGroup.height),
        UP_DIV(threads.depth, threadsPerGroup.depth),
    };
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}

#if MNN_METAL_DEBUG
#pragma mark debug
- (void)printTensor:(const Tensor *)tensor {
    tensor->print();
}

template <typename T>
void printBuffer(const void *content, unsigned long bytes, const char *fmt) {
    const T *data = (const T *)content;
    for (int i = 0; i < bytes / sizeof(T); i++) {
        if (i % 4 == 0)
            printf("%3d > ", i / 4);
        printf(fmt, data[i]);
        printf((i + 1) % 4 == 0 ? ",\n" : " ");
    }
}

- (void)printBuffer:(halide_buffer_t)buffer {
    if (buffer.host) {
        [self printBytes:buffer.host
                  length:buffer.dim[0].stride * buffer.dim[0].extent * buffer.type.bytes()
                    type:buffer.type.code
                    bits:buffer.type.bits];
    } else if (buffer.type.code == halide_type_float) {
        [self printBuffer:(__bridge id<MTLBuffer>)(void *)buffer.device type:buffer.type.code bits:16];
    } else {
        [self printBuffer:(__bridge id<MTLBuffer>)(void *)buffer.device type:buffer.type.code bits:buffer.type.bits];
    }
}

- (void)printBuffer:(id<MTLBuffer>)buffer type:(halide_type_code_t)type bits:(int)bits {
    [self printBytes:buffer.contents length:buffer.length type:type bits:bits];
}

- (void)printBytes:(const void *)bytes length:(NSUInteger)length type:(halide_type_code_t)type bits:(int)bits {
    if (type == halide_type_int) {
        if (bits == 8) { // int8
            printBuffer<int8_t>(bytes, length, "%3d");
        } else if (bits == 16) { // int16
            printBuffer<int16_t>(bytes, length, "%d");
        } else if (bits == 32) { // int32
            printBuffer<int32_t>(bytes, length, "%d");
        }
    } else if (type == halide_type_uint) {
        if (bits == 8) { // uint8
            printBuffer<uint8_t>(bytes, length, "%3d");
        } else if (bits == 16) { // uint16
            printBuffer<uint16_t>(bytes, length, "%d");
        } else if (bits == 32) { // uint32
            printBuffer<uint32_t>(bytes, length, "%d");
        }
    } else if (type == halide_type_float) {
        if (bits == 16) { // half
            printBuffer<metal_float>(bytes, length, "%.4f");
        } else { // float
            printBuffer<float>(bytes, length, "%.4f");
        }
    }
}

- (void)printEncoder:(id<MTLCommandEncoder>)encoder {
    printf("[METAL] %s encoded.\n", encoder.label.UTF8String);
}
#endif

@end
#endif /* MNN_METAL_ENABLED */
