//
//  AutoTime.cpp
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "AutoTime.hpp"
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "Macro.h"

namespace MNN {
AutoTime::AutoTime(int line, const char* func) {
    mName = ::strdup(func);
    mLine = line;
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    mCurrentTime = Current.tv_sec * 1000000 + Current.tv_usec;
}
AutoTime::~AutoTime() {
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    auto lastTime = Current.tv_sec * 1000000 + Current.tv_usec;

    MNN_PRINT("%s, %d, cost time: %f ms\n", mName, mLine, (float)(lastTime - mCurrentTime) / 1000.0f);
    free(mName);
}
} // namespace MNN
