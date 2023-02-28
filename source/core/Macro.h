//
//  Macro.h
//  MNN
//
//  Created by MNN on 2018/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef macro_h
#define macro_h
#include <MNN/MNNDefine.h>

#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)

// fraction length difference is 16bit. calculate the real value, it's about 0.00781
#define F32_BF16_MAX_LOSS ((0xffff * 1.0f ) / ( 1 << 23 ))

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


#ifndef MNN_USE_NEON
#if (__arm__ || __aarch64__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#define MNN_USE_NEON
#endif
#endif

#if defined(ENABLE_ARMV82)
#if defined(MNN_BUILD_FOR_ANDROID) || defined(__aarch64__)
#define MNN_USE_ARMV82
#endif

#if defined(__APPLE__)
#if TARGET_OS_SIMULATOR
#ifdef MNN_USE_ARMV82
#undef MNN_USE_ARMV82
#endif
#endif
#endif

#endif

#endif /* macro_h */
