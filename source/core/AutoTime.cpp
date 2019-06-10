//
//  AutoTime.cpp
//  MNN
//
//  Created by MNN on 2018/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <Windows.h>
#else
#include <sys/time.h>
#endif
#include "AutoTime.hpp"
#include "Macro.h"

namespace MNN {
AutoTime::AutoTime(int line, const char* func) {
    mName = ::strdup(func);
    mLine = line;
#if defined(_MSC_VER)
    LARGE_INTEGER time, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time);
    uint64_t sec = time.QuadPart / freq.QuadPart;
    uint64_t usec = (time.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    mCurrentTime = sec * 1000000 + usec;
#else
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    mCurrentTime = Current.tv_sec * 1000000 + Current.tv_usec;
#endif
}
AutoTime::~AutoTime() {
#if defined(_MSC_VER)
    LARGE_INTEGER time, freq;
    QueryPerformanceCounter(&time);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = time.QuadPart / freq.QuadPart;
    uint64_t usec = (time.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    auto lastTime = sec * 1000000 + usec;
#else
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    auto lastTime = Current.tv_sec * 1000000 + Current.tv_usec;
#endif

    MNN_PRINT("%s, %d, cost time: %f ms\n", mName, mLine, (float)(lastTime - mCurrentTime) / 1000.0f);
    free(mName);
}
} // namespace MNN
