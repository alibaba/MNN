//
//  MNNRvvFastPathUtils.hpp
//  MNN
//
//  Created by MNN on 2026/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_RVV_FAST_PATH_UTILS_HPP
#define MNN_RVV_FAST_PATH_UTILS_HPP

#include <functional>
#include <utility>

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

template <typename Function>
static inline void MNNRvvFastPathParallelFor(Backend* backend, int threadNumber, Function&& function) {
    if (threadNumber <= 1) {
        function(0);
        return;
    }
#if defined(MNN_FORBIT_MULTI_THREADS)
    for (int threadId = 0; threadId < threadNumber; ++threadId) {
        function(threadId);
    }
#elif defined(MNN_USE_THREAD_POOL)
    std::pair<std::function<void(int)>, int> task;
    task.first = std::forward<Function>(function);
    task.second = threadNumber;
    static_cast<CPUBackend*>(backend)->enqueue(task);
#else
#pragma omp parallel for
    for (int threadId = 0; threadId < threadNumber; ++threadId) {
        function(threadId);
    }
#endif
}

} // namespace MNN

#endif // MNN_RVV_FAST_PATH_UTILS_HPP
