//
//  Macro.h
//  MNN
//
//  Created by MNN on 2018/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef macro_h
#define macro_h
#include "MNNDefine.h"

#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)
#if (__arm__ || __aarch64__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#define MNN_USE_NEON
#endif

#endif /* macro_h */
