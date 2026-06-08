#ifndef MNN_RKNNBACKEND_HPP
#define MNN_RKNNBACKEND_HPP

#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {
namespace RKNN {

class RKNNRuntime;

class RKNNBackend : public Backend {
public:
    explicit RKNNBackend(const RKNNRuntime* runtime);
    ~RKNNBackend() override = default;

    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                        const MNN::Op* op) override;
    void onResizeBegin() override;
    ErrorCode onResizeEnd() override;
    void onExecuteBegin() const override;
    void onExecuteEnd() const override;
    MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    bool onClearBuffer() override;
    void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
    const Runtime* getRuntime() override;

private:
    const RKNNRuntime* mRuntime;
};

class RKNNRuntime : public Runtime {
public:
    explicit RKNNRuntime(const Backend::Info& info);
    ~RKNNRuntime() override = default;

    Backend* onCreate(const BackendConfig* config = nullptr, Backend* origin = nullptr) const override;
    void onGabageCollect(int level) override;
    CompilerType onGetCompilerType() const override;

private:
    Backend::Info mInfo;
};

class RKNNRuntimeCreator : public RuntimeCreator {
public:
    Runtime* onCreate(const Backend::Info& info) const override;
    bool onValid(Backend::Info& info) const override;
};

} // namespace RKNN

void registerRKNNRuntimeCreator();
} // namespace MNN

#endif
