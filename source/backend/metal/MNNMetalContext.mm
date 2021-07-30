//
//  MNNMetalContext.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import <sys/utsname.h>

#if MNN_METAL_ENABLED

using namespace MNN;

@interface MNNMetalContext ()
// public
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer_net;
@property (assign, nonatomic) NSUInteger maxThreadgroupMemoryLength;
// private
@property (strong, nonatomic) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *caches;
@property (strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>> *waitings;
@property (strong, nonatomic) id<MTLLibrary> library;
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
        NSString *remotePath = [self getMetalLibFromRuntimeCore];
        NSString *path = remotePath ? remotePath : [NSBundle.mainBundle pathForResource:@"mnn" ofType:@"metallib"];
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

+ (NSString *)getMetalLibFromRuntimeCore {
    id MRTFileSystemClass = NSClassFromString(@"MRTFileSystem");
    NSString *resourcePath = [MRTFileSystemClass performSelector:@selector(resourceContainerWithName:) withObject:@"metallib_transfer"];
    NSString *metallibPath = [resourcePath stringByAppendingPathComponent:@"mnn.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
        return metallibPath;
    } else {
        return nil;
    }
}

+ (BOOL)commit_frequent{
    struct utsname systemInfo;
    uname(&systemInfo);

    NSString *deviceString = [NSString stringWithCString:systemInfo.machine encoding:NSASCIIStringEncoding];

    if ([deviceString isEqualToString:@"iPhone11,2"]) return YES; //@"iPhone XS";
    if ([deviceString isEqualToString:@"iPhone11,4"]) return YES; //@"iPhone XS Max";
    if ([deviceString isEqualToString:@"iPhone11,6"]) return YES; //@"iPhone XS Max";
    if ([deviceString isEqualToString:@"iPhone11,8"]) return YES; //@"iPhone XR";
    if ([deviceString isEqualToString:@"iPhone12,1"]) return YES; //@"iPhone 11";
    if ([deviceString isEqualToString:@"iPhone12,3"]) return YES; //@"iPhone 11 Pro";
    if ([deviceString isEqualToString:@"iPhone12,5"]) return YES; //@"iPhone 11 Pro Max";
    if ([deviceString isEqualToString:@"iPhone12,8"]) return YES; //@"iPhone SE 2";
    if ([deviceString isEqualToString:@"iPhone13,1"]) return YES; //@"iPhone 12 mini";
    if ([deviceString isEqualToString:@"iPhone13,2"]) return YES; //@"iPhone 12";
    if ([deviceString isEqualToString:@"iPhone13,3"]) return YES; //@"iPhone 12 Pro";
    if ([deviceString isEqualToString:@"iPhone13,4"]) return YES; //@"iPhone 12 Pro Max";
    return NO;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        // public
        _device        = self.class.device;
        _commandQueue  = [_device newCommandQueue];
        _commandBuffer = [_commandQueue commandBuffer];
        _commandBuffer_net = [_commandQueue commandBuffer];
        if (@available(iOS 11.0, *)) {
            _maxThreadgroupMemoryLength = _device.maxThreadgroupMemoryLength;
        } else {
            _maxThreadgroupMemoryLength = 16352; // 16352(16k - 32b) on iOS 11- according to feature set doc
        }
        
        _isCommitEachShader = self.class.commit_frequent;
        
        // private
        _caches   = [NSMutableDictionary dictionary];
        _waitings = [NSMutableArray array];
        _library  = self.class.library;
        if (nil == _library) {
            return nil;
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
- (id<MTLBlitCommandEncoder>)encoderBlit {
    id<MTLBlitCommandEncoder> result = [_commandBuffer blitCommandEncoder];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    result.label = nil;
#endif
    return result;
}

- (id<MTLComputeCommandEncoder>)encoder_net {
    id<MTLComputeCommandEncoder> result = [_commandBuffer_net computeCommandEncoder];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    result.label = nil;
#endif
    return result;
}
- (id<MTLBlitCommandEncoder>)encoderBlit_net {
    id<MTLBlitCommandEncoder> result = [_commandBuffer_net blitCommandEncoder];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    result.label = nil;
#endif
    return result;
}

- (MetalBandwidth)load:(NSString *)name encoder:(id<MTLComputeCommandEncoder>)encoder {
    id<MTLComputePipelineState> pipeline = [self pipelineWithName:name];
    MNN_ASSERT(nil != pipeline);
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

- (id<MTLCommandBuffer>) newCmdBuffer {
    id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer]; // create a new command buffer
    return cmdBuffer;
}

- (NSUInteger)timeUsed:(id<MTLCommandBuffer>)buffer {
    [buffer commit];
    [buffer waitUntilCompleted];
    NSUInteger time = (NSUInteger)((buffer.GPUEndTime - buffer.GPUStartTime)* 1000000.f);//us
    return time;
}


- (std::pair<MTLSize, MTLSize>) getGridAndThreadgroup: (id<MTLComputePipelineState>)pipeline gid:(MTLSize)threads loop:(NSUInteger)count buffer:(NSArray *)buffers {
    NSUInteger gid_x = threads.width;
    NSUInteger gid_y = threads.height;
    NSUInteger gid_z = threads.depth;
    
    std::pair<MTLSize, MTLSize> thread;//Grid and ThreadGroup
    thread.second = MTLSizeMake(2, 1, 1);
    thread.first = {UP_DIV(gid_x, thread.second.width), UP_DIV(gid_y, thread.second.height), UP_DIV(gid_z, thread.second.depth)};

#ifdef MNN_METAL_TUNE
    NSUInteger min_time = UINT_MAX;
    for(NSUInteger z = 1; z < gid_z*2; z *= 2) {
        for(NSUInteger y = 1; y < gid_y*2; y *= 2) {
            for(NSUInteger x = 2; x < gid_x*2; x *= 2) {
                if(x * y * z <= pipeline.maxTotalThreadsPerThreadgroup) {
                    id<MTLCommandBuffer> commamd_buffer = [self newCmdBuffer];
                    id<MTLComputeCommandEncoder> encoder = [commamd_buffer computeCommandEncoder];
                    MTLSize local = {x, y, z};
                    MTLSize global = {UP_DIV(gid_x, x), UP_DIV(gid_y, y), UP_DIV(gid_z, z)};
                    int loop = count;
                    while(loop--) {
                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:[buffers objectAtIndex:0] offset:0 atIndex:0];
                        [encoder setBuffer:[buffers objectAtIndex:1] offset:0 atIndex:1];
                        [encoder setBuffer:[buffers objectAtIndex:2] offset:0 atIndex:2];
                        [encoder setBuffer:[buffers objectAtIndex:3] offset:0 atIndex:3];
                        [encoder setBuffer:[buffers objectAtIndex:4] offset:0 atIndex:4];
                                            
                        [encoder dispatchThreadgroups:global threadsPerThreadgroup:local];
                    }
                    [encoder endEncoding];
                    auto time = [self timeUsed :commamd_buffer];
                    if(time < min_time) {
                        min_time = time;
                        thread.first = global;
                        thread.second = local;
                    }
                }
            }
        }
    }
    //printf("prit: %d   us, %d %d %d, %d %d %d\n", min_time, threads.width, threads.height, threads.depth, thread.second.width, thread.second.height, thread.second.depth);
#else
    thread = [self computeBestGroupAndLocal:pipeline threads:threads];
    //printf("prit:%d %d %d, %d %d %d, \n", thread.first.width, thread.first.height, thread.first.depth, thread.second.width, thread.second.height, thread.second.depth);
#endif
    return thread;
}


#pragma mark dispatch
- (void)commit {
    if (_commandBuffer.status < MTLCommandBufferStatusCommitted) {
        [_commandBuffer commit];
        [_waitings addObject:_commandBuffer];
        _commandBuffer = [_commandQueue commandBuffer]; // create a new command buffer
    }
}

- (void)commit_net {
    if (_commandBuffer_net.status < MTLCommandBufferStatusCommitted) {
        [_commandBuffer_net commit];
        [_waitings addObject:_commandBuffer_net];
        _commandBuffer_net = [_commandQueue commandBuffer]; // create a new command buffer
    }
}

- (void)wait {
    for (id<MTLCommandBuffer> buffer in _waitings) {
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
    [_waitings removeAllObjects];
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

- (std::pair<MTLSize, MTLSize>)computeBestGroupAndLocal:(id<MTLComputePipelineState>) bw threads:(MTLSize)t {
    auto local = [self computeBestGroup:bw threads:t];
    auto globalSize = MTLSizeMake(UP_DIV(t.width, local.width), UP_DIV(t.height, local.height), UP_DIV(t.depth, local.depth));
    return std::make_pair(globalSize, local);
}

- (MTLSize)computeBestGroup:(id<MTLComputePipelineState>) bw threads:(MTLSize)t {
    if (bw.maxTotalThreadsPerThreadgroup > 64) {
        auto res = MTLSizeMake(8, 8, 8);
        int reduceNumber = 0;
        if (t.depth < 4) {
            res.depth = 1;
            reduceNumber++;
        }
        if (t.width < 4) {
            res.width = 1;
            reduceNumber++;
        }
        if (t.height < 4) {
            res.height = 1;
            reduceNumber++;
        }
        if (reduceNumber == 0) {
            return MTLSizeMake(4, 4, 4);
        }
        if (reduceNumber == 2) {
            if (res.width > 1) {
                res.width = 64;
            }
            if (res.height > 1) {
                res.height = 64;
            }
            if (res.depth > 1) {
                res.depth = 64;
            }
        }
        return res;
    }
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
    auto sz = pz;
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
