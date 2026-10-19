//
// HiAI HCL V600 runtime bridge.
//

#ifndef MNN_HIAI_HCL_V600_RUNTIME_HPP
#define MNN_HIAI_HCL_V600_RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace MNN {

class HiaiHclV600Runtime {
public:
    static std::unique_ptr<HiaiHclV600Runtime>
    CreateIfSupported(bool enableAutoTuning = false, const std::string& tuningCacheDirectory = std::string());
    static bool IsSupported(std::string* version = nullptr);

    ~HiaiHclV600Runtime();

    HiaiHclV600Runtime(const HiaiHclV600Runtime&) = delete;
    HiaiHclV600Runtime& operator=(const HiaiHclV600Runtime&) = delete;

    bool buildAndLoad(const void* irData, size_t irSize, const std::string& modelName, bool preferFp16,
                      const std::string& cacheFile);
    bool resetModel();
    bool release();
    int run();

    size_t inputCount() const;
    size_t outputCount() const;
    size_t inputSize(size_t index) const;
    size_t outputSize(size_t index) const;
    void* inputData(size_t index) const;
    void* outputData(size_t index) const;

    bool bindNativeHandleIo(size_t inputIndex, int inputFd, size_t inputBytes, size_t outputIndex, int outputFd,
                            size_t outputBytes);
    void clearNativeHandleIo();

    const std::string& version() const;
    const std::string& lastError() const;

private:
    struct Impl;
    explicit HiaiHclV600Runtime(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> mImpl;
};

} // namespace MNN

#endif // MNN_HIAI_HCL_V600_RUNTIME_HPP
