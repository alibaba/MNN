//
//  CoreMLExecutorWrapper.h
//  MNN
//
//  Created by MNN on 2021/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLEXECUTORWRAPPER_H
#define MNN_COREMLEXECUTORWRAPPER_H
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <MNN/Tensor.hpp>
#include "Model.pb-c.h"
namespace MNN {
std::shared_ptr<void> createCoreMLTemporaryDirectory(std::string& path);
class CoreMLExecutorWrapper {
public:
    explicit CoreMLExecutorWrapper(int precision);
    ~CoreMLExecutorWrapper();
    bool compileModel(CoreML__Specification__Model* model, const std::string& weightsPath = "",
                      const std::shared_ptr<std::vector<uint8_t>>& weights = nullptr,
                      bool forceFileCompilation = false);
    bool compiledFromSource() const;
    std::shared_ptr<CoreMLExecutorWrapper> createSession(const CoreMLExecutorWrapper* previous = nullptr) const;
    bool invokModel(const std::vector<std::pair<const Tensor*, std::string>>& inputs,
                    const std::vector<std::pair<const Tensor*, std::string>>& outputs, bool flatIO = false);

private:
    void* mCoreMLExecutorPtr = nullptr;
    std::shared_ptr<std::mutex> mPredictionMutex = std::make_shared<std::mutex>();
};
} // namespace MNN
#endif
