//
//  CoreMLDefine.h
//  MNN
//
//  Created by MNN on 2021/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CoreMLDefine_h
#define CoreMLDefine_h


#ifdef MNN_COREML_ENABLED
#if !defined(__APPLE__)
#undef MNN_COREML_ENABLED
#define MNN_COREML_ENABLED 0
#else
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#if (TARGET_OS_IPHONE && TARGET_OS_SIMULATOR)
#undef MNN_COREML_ENABLED
#define MNN_COREML_ENABLED 0
#endif

#endif

#endif /* CoreMLDefine_h */
