//
//  CoreMLExecutorWrapper.mm
//  MNN
//
//  Created by MNN on 2021/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLDefine.h"
#import "CoreMLExecutorWrapper.h"
#import "CoreMLExecutor.h"

namespace MNN {
// cast c-ptr to objectc-ptr and transfers ownership
static inline CoreMLExecutor* getCoreMLExecutoreOwn(void* ptr) {
    return (__bridge_transfer CoreMLExecutor*)ptr;
}
// cast c-ptr to objectc-ptr with no transfer of ownership
static inline CoreMLExecutor* getCoreMLExecutoreRef(void* ptr) {
    return (__bridge CoreMLExecutor*)ptr;
}

CoreMLExecutorWrapper::CoreMLExecutorWrapper() {
    if (mCoreMLExecutorPtr == nullptr)  {
        mCoreMLExecutorPtr = (__bridge_retained void*)[[CoreMLExecutor alloc] init];
    }
}

CoreMLExecutorWrapper::~CoreMLExecutorWrapper() {
    auto executor = getCoreMLExecutoreOwn(mCoreMLExecutorPtr);
    (void)executor;
    mCoreMLExecutorPtr = nullptr;
}

bool CoreMLExecutorWrapper::compileModel(CoreML__Specification__Model* model) {
    if (@available(iOS 11, *)) {
        auto executor = getCoreMLExecutoreRef(mCoreMLExecutorPtr);
        NSURL* model_url = [executor saveModel:model];
        if (![executor build:model_url]) {
            printf("Failed to Compile and save Model.");
            return false;
        }
        [executor cleanup];
        return true;
    } else {
        printf("Failed to Compile and save Model.");
        return false;
    }
}

void CoreMLExecutorWrapper::invokModel(const std::vector<std::pair<const MNN::Tensor*, std::string>>& inputs,
                                       const std::vector<std::pair<const MNN::Tensor*, std::string>>& outputs) {
    if (@available(iOS 11, *)) {
        auto executor = getCoreMLExecutoreRef(mCoreMLExecutorPtr);
        if (![executor invokeWithInputs:inputs outputs:outputs]) {
            printf("Failed to Invok the Model.\n");
            return;
        }
    } else {
        printf("Failed to Invok the Model.\n");
        return;
    }
}
};
