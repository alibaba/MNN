//
//  MetalDefine.h
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalDefine_h
#define MetalDefine_h


#ifdef MNN_METAL_ENABLED
#if !defined(__APPLE__)
#undef MNN_METAL_ENABLED
#define MNN_METAL_ENABLED 0
#else
#import <Metal/Metal.h>
#import <float.h>
#endif

#endif
#ifndef MNN_METAL_DEBUG
#if DEBUG
#define MNN_METAL_DEBUG 1
#else
#define MNN_METAL_DEBUG 0
#endif
#endif

#define MNN_METAL_BENCHMARK 0


#endif /* MetalDefine_h */
