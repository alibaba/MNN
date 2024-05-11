//
//  MNNMetalContext.h
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNMETALCONTEXT_H
#define MNNMETALCONTEXT_H

#import "MetalDefine.h"
#import <MNN/Tensor.hpp>
#import "MetalBackend.hpp"
#if MNN_METAL_ENABLED
#define MNN_PRINT_ENCODER(context, encoder) ((void)0)
#define MNN_METAL
#import <MNN/MNNSharedContext.h>

namespace MNN {
typedef enum {
    /** read write in CPU */
    CPUReadWrite = 0,
    /** write in CPU but never read */
    CPUWriteOnly,
    /** neither read nor write in CPU */
    CPUTransparent
} MetalAccess;

}

@interface MNNMetalContext : NSObject
/** metal device */
@property (strong, nonatomic, readonly) id<MTLDevice> device;
/** max memory length cound be used in threadgroup */
@property (assign, nonatomic, readonly) BOOL isCommitEachShader;
@property (assign, nonatomic, readonly) BOOL isIphone;

/**
 * @brief alloc temp buffer on device
 * @param size      buffer size
 * @param access    metal access type
 * @return created device buffer
 */
- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size access:(MNN::MetalAccess)access;

/**
 * @brief alloc temp buffer on device
 * @param size      buffer size
 * @param bytes     buffer data
 * @param access    metal access type
 * @return created device buffer
 */
- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size bytes:(const void *)bytes access:(MNN::MetalAccess)access;


/**
 * @brief load encoder with function name. returns maxTotalThreadsPerThreadgroup of pipeline.
 * @param name      pipline name
 * @param encoder   command encoder
 * @return bandwidth info for function
 */
- (id<MTLCommandBuffer>) newCmdBuffer:(MTLSize) localIndex queue:(id<MTLCommandQueue>) cmdqueue;

- (NSUInteger)timeUsed:(id<MTLCommandBuffer>) buffer;

- (std::tuple<MTLSize, MTLSize, NSUInteger>) getGridAndThreadgroup: (id<MTLComputePipelineState>)pipeline gid:(MTLSize)threads loop:(NSUInteger)count buffer:(NSArray *)buffers runtime:(MNN::MetalRuntime *) rt shaderName:(std::string) kernelName offsets:(int *) offset_arr queue:(id<MTLCommandQueue>) cmdqueue;
- (NSUInteger)PipelinetimeUsed: (id<MTLComputePipelineState>)pipeline global:(MTLSize)globals local:(MTLSize)locals loop:(NSUInteger)count buffer:(NSArray *)buffers queue:(id<MTLCommandQueue>) cmdqueue;


- (BOOL) initWithSharedContext:(const MNNMetalSharedContext*)context dev:(id<MTLDevice>)device;

/**
 * @brief dispatch encoder with designated threads per threadgroup
 * @param encoder           command encoder
 * @param threads           threads size
 * @param threadsPerGroup   thread size per group
 * @param bandwidth         bandwidth
 */
- (id<MTLComputePipelineState>)pipelineWithName:(NSString *)name fp16:(BOOL)fp16;
- (id<MTLComputePipelineState>)pipelineWithSourceOption:(NSString *)source name:(NSString *)name options:(MTLCompileOptions *)options;
- (MTLSize)computeBestGroup:(id<MTLComputePipelineState>) pipeline threads:(MTLSize)threads;

- (std::pair<MTLSize, MTLSize>)computeBestGroupAndLocal:(id<MTLComputePipelineState>) bw threads:(MTLSize)t;

#if MNN_METAL_DEBUG
/**
 * @brief print tensor contents
 */
- (void)printTensor:(const MNN::Tensor *)tensor;

/**
 * @brief print halide buffer
 */
- (void)printBuffer:(halide_buffer_t)buffer;

/**
 * @brief print buffer contents
 */
- (void)printBuffer:(id<MTLBuffer>)buffer type:(halide_type_code_t)type bits:(int)bits;

/**
 * @brief print bytes
 */
- (void)printBytes:(const void *)bytes length:(NSUInteger)length type:(halide_type_code_t)type bits:(int)bits;

/**
 * @brief print encoder
 */
- (void)printEncoder:(id<MTLCommandEncoder>)encoder;


#endif
@end

#endif /* MNN_METAL_ENABLED */
#endif
