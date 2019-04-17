//
//  MetalDefine.h
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalDefine_h
#define MetalDefine_h

#if !defined(__APPLE__)
#define MNN_METAL_ENABLED 0
#else
#import <Metal/Metal.h>
#import <float.h>
#endif

#ifndef MNN_METAL_ENABLED
#define MNN_METAL_ENABLED (!(TARGET_OS_IPHONE && TARGET_OS_SIMULATOR))
#endif

#ifndef MNN_METAL_DEBUG
#if DEBUG
#define MNN_METAL_DEBUG 1
#else
#define MNN_METAL_DEBUG 0
#endif
#endif

#define MNN_METAL_BENCHMARK 0
#define MNN_METAL_FULL_PRECISION 0 // should edit in metal too

#if MNN_METAL_FULL_PRECISION || !defined(__FLT16_EPSILON__)
typedef float metal_float;
#define MNNMetalPixelFormatRGBAFloat MTLPixelFormatRGBA32Float
#else
typedef __fp16 metal_float;
#define MNNMetalPixelFormatRGBAFloat MTLPixelFormatRGBA16Float
#endif

#endif /* MetalDefine_h */
