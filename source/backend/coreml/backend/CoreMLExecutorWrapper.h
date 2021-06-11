//
//  CoreMLExecutorWrapper.h
//  MNN
//
//  Created by MNN on 2021/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLEXECUTORWRAPPER_H
#define MNN_COREMLEXECUTORWRAPPER_H

#include <vector>
#include <MNN/Tensor.hpp>
#include "Model.pb-c.h"

// this class is wrapper of CoreMLExecutor
// seprate objectc source file and cpp source file
namespace MNN {
    class CoreMLExecutorWrapper {
    public:
        CoreMLExecutorWrapper();
        ~CoreMLExecutorWrapper();
        bool compileModel(CoreML__Specification__Model* model);
        void invokModel(const std::vector<std::pair<const MNN::Tensor*, std::string>>& inputs,
                        const std::vector<std::pair<const MNN::Tensor*, std::string>>& outputs);
    private:
        void* mCoreMLExecutorPtr = nullptr;
    };
};

#endif //MNN_COREMLEXECUTORWRAPPER_H
