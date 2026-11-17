//
//  CoreMLExecutorWrapper.mm
//  MNN
//
//  Created by MNN on 2021/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "CoreMLExecutorWrapper.h"
#import "CoreMLExecutor.h"
namespace MNN {
static CoreMLExecutor* executorRef(void* ptr) { return (__bridge CoreMLExecutor*)ptr; }
CoreMLExecutorWrapper::CoreMLExecutorWrapper(int precision) {
    auto executor = [[CoreMLExecutor alloc] init];
    executor.precision = precision;
    mCoreMLExecutorPtr = (__bridge_retained void*)executor;
}
CoreMLExecutorWrapper::~CoreMLExecutorWrapper() {
    @autoreleasepool {
        auto executor = (__bridge_transfer CoreMLExecutor*)mCoreMLExecutorPtr;
        (void)executor;
    }
}
bool CoreMLExecutorWrapper::compiledFromSource() const { return executorRef(mCoreMLExecutorPtr).compiledFromSource; }
bool CoreMLExecutorWrapper::compileModel(CoreML__Specification__Model* model, const std::string& weightsPath,
                                         const std::shared_ptr<std::vector<uint8_t>>& weights, bool forceFileCompilation) {
    @autoreleasepool {
        return [executorRef(mCoreMLExecutorPtr) buildModel:model weightsPath:weightsPath weights:weights
                                      forceFileCompilation:forceFileCompilation];
    }
}
std::shared_ptr<CoreMLExecutorWrapper> CoreMLExecutorWrapper::createSession(const CoreMLExecutorWrapper* previous) const {
    @autoreleasepool {
        auto executor = [executorRef(mCoreMLExecutorPtr) sessionWithPrevious:
            previous ? executorRef(previous->mCoreMLExecutorPtr) : nil];
        if (!executor) {
            return nullptr;
        }
        std::shared_ptr<CoreMLExecutorWrapper> session(new CoreMLExecutorWrapper(executor.precision));
        auto unused = (__bridge_transfer CoreMLExecutor*)session->mCoreMLExecutorPtr;
        (void)unused;
        session->mCoreMLExecutorPtr = (__bridge_retained void*)executor;
        session->mPredictionMutex = mPredictionMutex;
        return session;
    }
}
bool CoreMLExecutorWrapper::invokModel(const std::vector<std::pair<const Tensor*, std::string>>& inputs,
                                      const std::vector<std::pair<const Tensor*, std::string>>& outputs, bool flatIO) {
    std::lock_guard<std::mutex> lock(*mPredictionMutex);
    @autoreleasepool { return [executorRef(mCoreMLExecutorPtr) invokeWithInputs:inputs outputs:outputs flatIO:flatIO]; }
}
} // namespace MNN
