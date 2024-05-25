//
//  WrapExecution.hpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WrapExecution_hpp
#define WrapExecution_hpp

#include <stdio.h>
#include <memory>
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"

namespace MNN {

/** execution wrapper. hiding cross-backend tensor converting. */
class MNN_PUBLIC WrapExecution {
public:
    static void copyReplaceTensor(const Tensor* input, Tensor* output);
    static bool needWrap(const Tensor* input, Backend* current);
    static bool allocAndCopy(Backend* curBackend, const Tensor* input, Tensor* output);
    static Tensor* copyConstCache(Tensor* tensor, Backend* curBackend, std::map<Tensor*, std::shared_ptr<Tensor>>& cache, bool forbidReplace);
    static std::shared_ptr<Tensor> makeCopyTensor(Tensor* tensor, Backend* targetBackend);
    static Execution* makeCopyExecution(Backend* backend, Backend* backupBackend);
};


} // namespace MNN

#endif /* WrapExecution_hpp */
