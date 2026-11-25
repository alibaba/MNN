//
//  CoreMLExecutorWrapper.h
//  MNN
//
//  Created by MNN on 2021/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLEXECUTORWRAPPER_H
#define MNN_COREMLEXECUTORWRAPPER_H
#include <CoreVideo/CVPixelBuffer.h>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>
#include <MNN/Tensor.hpp>
#include "Model.pb-c.h"
namespace MNN {
std::shared_ptr<void> createCoreMLTemporaryDirectory(std::string& path);
using CoreMLWeightImage = std::shared_ptr<std::remove_pointer<CVPixelBufferRef>::type>;
using CoreMLImageInputs = std::vector<std::pair<std::string, CVPixelBufferRef>>;
CoreMLWeightImage createCoreMLWeightImage(int width, int height);

class CoreMLExecutorWrapper {
public:
    explicit CoreMLExecutorWrapper(int precision);
    ~CoreMLExecutorWrapper();
#ifdef MNN_COREML_PREFILL_BENCH
    bool loadBenchmarkModel(const std::string& directory);
#endif
    bool compileModel(CoreML__Specification__Model* model, const std::string& weightsPath = "",
                      const std::shared_ptr<std::vector<uint8_t>>& weights = nullptr,
                      bool forceFileCompilation = false);
    bool compiledFromSource() const;
    std::shared_ptr<CoreMLExecutorWrapper> createSession(const CoreMLExecutorWrapper* previous = nullptr) const;
    bool invokModel(const std::vector<std::pair<const Tensor*, std::string>>& inputs,
                    const std::vector<std::pair<const Tensor*, std::string>>& outputs, bool flatIO = false);
    bool invokModel(const std::vector<std::pair<const Tensor*, std::string>>& inputs,
                    const std::vector<std::pair<const Tensor*, std::string>>& outputs, bool flatIO,
                    const CoreMLImageInputs& images);

private:
    void* mCoreMLExecutorPtr = nullptr;
    std::shared_ptr<std::mutex> mPredictionMutex = std::make_shared<std::mutex>();
};
} // namespace MNN
#endif
