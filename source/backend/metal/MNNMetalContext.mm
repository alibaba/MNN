//
//  MNNMetalContext.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MNNMetalContext.h"
#import "backend/metal/MetalBackend.hpp"
#import "core/Macro.h"
#import <sys/utsname.h>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#if MNN_METAL_ENABLED
#include "ShaderMap.hpp"
#include <sstream>

using namespace MNN;

@interface MNNMetalContext ()
// public
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer_net;
// private
@property (strong, nonatomic) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *caches;
@property (strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>> *waitings;
@property (strong, nonatomic) NSMutableDictionary<NSString *, id<MTLLibrary>>* library;
@end

@implementation MNNMetalContext

static void createLibrary(id<MTLDevice> device, NSMutableDictionary<NSString *, id<MTLLibrary>>* libraryMap) {
    AUTOTIME;
    ShaderMap shader;
    auto first = shader.search("shader_MetalDefine_metal");
    auto second = shader.search("shader_MetalConvolutionActivation_metal");
    auto& raw = shader.getRaw();
    for (auto& iter : raw) {
        std::ostringstream total;
        if (iter.first == "shader_MetalDefine_metal") {
            continue;
        }
        if (iter.first == "shader_MetalConvolutionActivation_metal") {
            continue;
        }
        total << first << "\n" << second << "\n" << iter.second;
        auto totalString = total.str();
        auto totalNSString = [[NSString alloc] initWithUTF8String:totalString.c_str()];
        NSError *err = nil;
        auto library = [device newLibraryWithSource:totalNSString options:nil error:&err];
        if (nil == library) {
            if (err) {
                printf("Error Key = %s\n", iter.first.c_str());
                NSLog(@"Warning: Metallib Library error: %@", err);
            }
            [libraryMap removeAllObjects];
            libraryMap = nil;
            return;
        }
        auto functionNames = [library functionNames];
        for(int i=0; i<functionNames.count ; i++) {
            libraryMap[functionNames[i]] = library;
        }
    }
}

+ (BOOL)commit_frequent{
    struct utsname systemInfo;
    uname(&systemInfo);

    NSString *deviceString = [NSString stringWithCString:systemInfo.machine encoding:NSASCIIStringEncoding];

    if ([deviceString isEqualToString:@"iPhone10,1"]) return YES; //@"iPhone 8 Global";
    if ([deviceString isEqualToString:@"iPhone10,2"]) return YES; //@"iPhone 8 Plus Global";
    if ([deviceString isEqualToString:@"iPhone10,4"]) return YES; //@"iPhone 8 GSM";
    if ([deviceString isEqualToString:@"iPhone10,5"]) return YES; //@"iPhone 8 Plus GSM";
    if ([deviceString isEqualToString:@"iPhone10,3"]) return YES; //@"A1865/A1902 iPhone X";
    if ([deviceString isEqualToString:@"iPhone10,6"]) return YES; //@"Global/A1901 iPhone X";
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

- (BOOL) initWithSharedContext:(const MNNMetalSharedContext*)context dev:(id<MTLDevice>)device {
    MNN_ASSERT(nullptr != context);
    _device = context->device;
    _library = [NSMutableDictionary dictionary];
    createLibrary(_device, _library);
    _commandQueue  = context->queue;
    _commandBuffer = [_commandQueue commandBuffer];
    _commandBuffer_net = [_commandQueue commandBuffer];
    _caches   = [NSMutableDictionary dictionary];
    _waitings = [NSMutableArray array];
    _isCommitEachShader = self.class.commit_frequent;
    
    return (0 != [_library count]);
}

- (instancetype)init {
    self = [super init];
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
    auto lib = _library[name];
    id<MTLFunction> result = [lib newFunctionWithName:name];
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

- (id<MTLCommandBuffer>) newCmdBuffer:(MTLSize) localIndex {
    id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer]; // create a new command buffer
    std::string label = std::to_string((int)localIndex.width) + "_" + std::to_string((int)localIndex.height) + "_" + std::to_string((int)localIndex.depth);
    cmdBuffer.label = [NSString stringWithCString:label.c_str() encoding:[NSString defaultCStringEncoding]];
    return cmdBuffer;
}

- (NSUInteger)timeUsed:(id<MTLCommandBuffer>)buffer {
    [buffer commit];
    [buffer waitUntilCompleted];
    NSUInteger time = (NSUInteger)((buffer.GPUEndTime - buffer.GPUStartTime)* 1000000.f);//us
    return time;
}


- (std::tuple<MTLSize, MTLSize, NSUInteger>) getGridAndThreadgroup: (id<MTLComputePipelineState>)pipeline gid:(MTLSize)threads loop:(NSUInteger)count buffer:(NSArray *)buffers runtime:(MetalRuntime *) rt shaderName:(std::string) kernelName {
    NSUInteger gid_x = threads.width;
    NSUInteger gid_y = threads.height;
    NSUInteger gid_z = threads.depth;

    auto& tunedThreadGroup = rt->getTunedThreadGroup();
    std::vector<uint32_t> gws = {(uint32_t)gid_x, (uint32_t)gid_y, (uint32_t)gid_z};
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedThreadGroup.find(info) != tunedThreadGroup.end()) {
        //printf("conv2d1x1LocalWSOpt Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        auto groupNum = std::get<0>(tunedThreadGroup[info]);
        auto groupSize = std::get<1>(tunedThreadGroup[info]);
        auto timeCost = std::get<2>(tunedThreadGroup[info]);

        MTLSize _groupNum = {(NSUInteger)groupNum[0], (NSUInteger)groupNum[1], (NSUInteger)groupNum[2]};
        MTLSize _groupSize = {(NSUInteger)groupSize[0], (NSUInteger)groupSize[1], (NSUInteger)groupSize[2]};

        std::tuple<MTLSize, MTLSize, NSUInteger> result(_groupNum, _groupSize, (NSUInteger)timeCost);
        return result;
    }
    std::pair<MTLSize, MTLSize> thread;//Grid and ThreadGroup
    // set trick by computing
    thread = [self computeBestGroupAndLocal:pipeline threads:threads];
    
    if(rt->getTuneLevel() == Heavy) {
        count = 50;
    }
    NSUInteger min_time = UINT_MAX;
    if(rt->getTuneLevel() != Never)
    {
        //get original trick time
        {
            id<MTLCommandBuffer> commamd_buffer = [self newCmdBuffer:thread.second];
            id<MTLComputeCommandEncoder> encoder = [commamd_buffer computeCommandEncoder];
            
            int loop = count;
            while(loop--) {
                [encoder setComputePipelineState:pipeline];
                for(NSUInteger idx = 0; idx < buffers.count; idx++) {
                    [encoder setBuffer:[buffers objectAtIndex:idx] offset:0 atIndex:idx];
                }
                MNN_ASSERT(thread.second.width >= 1);
                MNN_ASSERT(thread.second.height >= 1);
                MNN_ASSERT(thread.second.depth >= 1);

                [encoder dispatchThreadgroups:thread.first threadsPerThreadgroup:thread.second];
            }
            [encoder endEncoding];
            min_time = [self timeUsed :commamd_buffer];
            //MNN_PRINT("orig prit: %d   us, %d %d %d\n", min_time, thread.second.width, thread.second.height, thread.second.depth);
        }
        
        bool isMuchTime = (min_time > 8000) ? true : false;
        NSUInteger magic_l = 1;
        NSUInteger magic_z = 16;
        NSUInteger magic_y = 4;
        NSUInteger magic_x = 4;
        
        if(rt->getTuneLevel() == Heavy) {
            magic_l = 2;
            magic_z = UINT_MAX;
            magic_y = UINT_MAX;
            magic_x = UINT_MAX;
        } else if(rt->getTuneLevel() == Wide) {
            bool isMuchTime = (min_time > 5000) ? true : false;
            magic_z = 16;
            magic_y = (isMuchTime ? 4 : 16);
            magic_x = (isMuchTime ? 4 : 16);
        } else if(rt->getTuneLevel() == Normal) {
            magic_z = 16;
            magic_y = 4;
            magic_x = 4;
        } else if(rt->getTuneLevel() == Fast) {
            magic_z = 4;
            magic_y = 4;
            magic_x = 4;
        }
        
        for(NSUInteger z = 1; z < gid_z * magic_l && z <= magic_z; z *= 4) {
            for(NSUInteger y = 1; y < gid_y * magic_l && y <= magic_y; y *= 4) {
                for(NSUInteger x = 1; x < gid_x * magic_l && x <= magic_x; x *= 4) {
                    if(x * y * z <= pipeline.maxTotalThreadsPerThreadgroup) {
                        if(x==1 && y==1 && z==1) {
                            continue;
                        }
                        MTLSize local = {x, y, z};
                        MTLSize global = {UP_DIV(gid_x, x), UP_DIV(gid_y, y), UP_DIV(gid_z, z)};
                        id<MTLCommandBuffer> commamd_buffer = [self newCmdBuffer:local];
                        id<MTLComputeCommandEncoder> encoder = [commamd_buffer computeCommandEncoder];

                        int loop = count;
                        while(loop--) {
                            [encoder setComputePipelineState:pipeline];
                            for(NSUInteger idx = 0; idx < buffers.count; idx++) {
                                [encoder setBuffer:[buffers objectAtIndex:idx] offset:0 atIndex:idx];
                            }
                                                
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
    }
    //MNN_PRINT("tune prit: %d   us, %d %d %d\n", min_time, thread.second.width, thread.second.height, thread.second.depth);

    if (tunedThreadGroup.find(info) == tunedThreadGroup.end()) {
        //MNN_PRINT("2dLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        std::vector<uint32_t> groupNum(3 ,0);
        groupNum[0] = thread.first.width;
        groupNum[1] = thread.first.height;
        groupNum[2] = thread.first.depth;
        
        std::vector<uint32_t> groupSize(3 ,0);
        groupSize[0] = thread.second.width;
        groupSize[1] = thread.second.height;
        groupSize[2] = thread.second.depth;

        tunedThreadGroup.insert(std::make_pair(info, std::make_tuple(groupNum, groupSize, (uint32_t)min_time)));
    }

    return std::make_tuple(thread.first, thread.second, min_time);
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
    local.width = ALIMAX(local.width, 1);
    local.height = ALIMAX(local.height, 1);
    local.depth = ALIMAX(local.depth, 1);
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
    MNN_ASSERT(threadsPerGroup.width >= 1);
    MNN_ASSERT(threadsPerGroup.height >= 1);
    MNN_ASSERT(threadsPerGroup.depth >= 1);

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
        [self printBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)buffer.device)->getBuffer() type:buffer.type.code bits:16];
    } else {
        [self printBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)buffer.device)->getBuffer() type:buffer.type.code bits:buffer.type.bits];
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
