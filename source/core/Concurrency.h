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
#define MNN_CONCURRENCY_BEGIN_CONDITION(__iter__, __num__, __condition__) \
    int __iter__ = 0;                                                     \
    for (; __iter__ < __num__; __iter__++) {
// iOS / OSX
#elif defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <stddef.h>

#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
dispatch_apply(__num__, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t __iter__) {
#define MNN_CONCURRENCY_END() \
    });
#define MNN_CONCURRENCY_BEGIN_CONDITION(__iter__, __num__, __condition__) \
dispatch_apply(__num__, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t __iter__) {
// Windows
#elif defined(_MSC_VER)
#include <omp.h>

#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
    int __iter__ = 0;                            \
    _Pragma("omp parallel for") for (; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }
#define MNN_CONCURRENCY_BEGIN_CONDITION(__iter__, __num__, __condition__) \
    int __iter__ = 0;                                                     \
    for (; __iter__ < __num__; __iter__++) {
// Android
#else
#include <omp.h>

#define MNN_STRINGIFY(a) #a
#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
    _Pragma("omp parallel for") for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }
#define MNN_CONCURRENCY_BEGIN_CONDITION(__iter__, __num__, __condition__) \
    _Pragma(MNN_STRINGIFY(omp parallel for if(__condition__))) \
    for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#endif
#endif /* concurrency_h */
