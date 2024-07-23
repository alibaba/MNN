//
//  Concurrency.h
//  MNN
//
//  Created by MNN on 2018/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef concurrency_h
#define concurrency_h

#ifdef MNN_FORBIT_MULTI_THREADS
#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }

#elif defined(MNN_USE_THREAD_POOL)
#include "backend/cpu/ThreadPool.hpp"

#define MNN_STRINGIFY(a) #a
#define MNN_CONCURRENCY_BEGIN(__iter__, __num__)       \
    {                                                  \
        std::pair<std::function<void(int)>, int> task; \
        task.second = __num__;                         \
        task.first  = [&](int __iter__) {
#define MNN_CONCURRENCY_END()                                      \
    }                                                              \
    ;                                                              \
    auto cpuBn = (CPUBackend*)backend();                           \
    MNN::ThreadPool::enqueue(std::move(task), cpuBn->taskIndex(), cpuBn->threadOpen() ? cpuBn->threadNumber() : 1); \
    }

#else
// iOS / OSX
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <stddef.h>

#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
dispatch_apply(__num__, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t __iter__) {
#define MNN_CONCURRENCY_END() \
    (void)(backend()); \
    });

// Windows
#elif defined(_MSC_VER)
#include <omp.h>

#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
    __pragma(omp parallel for) for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }
#define MNN_CONCURRENCY_BEGIN_CONDITION(__iter__, __num__, __condition__) \
    int __iter__ = 0;                                                     \
    __pragma(omp parallel for if(__condition__))                          \
    for (; __iter__ < __num__; __iter__++) {
// Android
#else
#include <omp.h>

#define MNN_STRINGIFY(a) #a
#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
    _Pragma("omp parallel for") for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }

#endif
#endif
#endif /* concurrency_h */
