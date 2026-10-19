#ifndef MNN_HIAI_BACKEND_CONFIG_HPP
#define MNN_HIAI_BACKEND_CONFIG_HPP

#include "../include/MNNHiAIBackend.h"
#include "../include/MNNHiAIIO.h"
#include <MNN/ErrorCode.hpp>
#include <cstdio>
#include <memory>
#include <string>
#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {

inline void hiaiDiagnostic(MNNHiAIDiagnosticsV1* out, MNNHiAIStage stage, ErrorCode code,
                           const char* message, int64_t nativeCode = 0) {
    if (out == nullptr || out->structSize < sizeof(*out) || out->version != MNN_HIAI_CONFIG_VERSION) return;
    out->stage = stage;
    out->code = code;
    out->nativeCode = nativeCode;
    std::snprintf(out->message, sizeof(out->message), "%s", message == nullptr ? "" : message);
}

struct HiAIBackendOptions {
    BackendConfig config;
    MNNHiAIBackendConfigV1 view = {};
    bool explicitConfig = false;
    std::string runtimeLibraryDirectory;
    std::string acceleratorCacheDirectory;
    bool autoTuning = false;
    MNNHiAINativeHandleIoContext* nativeIoContext = nullptr;
    MNNHiAIDiagnosticsV1* diagnostics = nullptr;

    void report(MNNHiAIStage stage, ErrorCode code, const char* message, int64_t nativeCode = 0) const {
        hiaiDiagnostic(diagnostics, stage, code, message, nativeCode);
    }
    void ready() const {
        if (diagnostics != nullptr && diagnostics->structSize >= sizeof(*diagnostics) &&
            diagnostics->version == MNN_HIAI_CONFIG_VERSION) ++diagnostics->readyCount;
        report(MNN_HIAI_STAGE_RESIZE, NO_ERROR, "HiAI graph ready");
    }
};

class HiAICopyDiagnostic {
public:
    HiAICopyDiagnostic(const HiAIBackendOptions* options, const ErrorCode& code) : mOptions(options), mCode(code) {}
    ~HiAICopyDiagnostic() {
        if (mOptions != nullptr && mCode != NO_ERROR)
            mOptions->report(MNN_HIAI_STAGE_COPY, mCode, "HiAI host I/O copy failed");
    }
private:
    const HiAIBackendOptions* mOptions;
    const ErrorCode& mCode;
};

inline MNNHiAIStatus validateHiAIConfig(const MNNHiAIBackendConfigV1* config) {
    if (config == nullptr || config->structSize < sizeof(*config) || config->version != MNN_HIAI_CONFIG_VERSION)
        return MNN_HIAI_STATUS_INVALID_ARGUMENT;
    if (config->autoTuning > 1) {
        hiaiDiagnostic(config->diagnostics, MNN_HIAI_STAGE_CONFIG, NOT_SUPPORT, "Unsupported HiAI option");
        return MNN_HIAI_STATUS_UNSUPPORTED_OPTION;
    }
    if (config->diagnostics != nullptr &&
        (config->diagnostics->structSize < sizeof(MNNHiAIDiagnosticsV1) ||
         config->diagnostics->version != MNN_HIAI_CONFIG_VERSION))
        return MNN_HIAI_STATUS_INVALID_ARGUMENT;
    return MNN_HIAI_STATUS_SUCCESS;
}

inline std::shared_ptr<HiAIBackendOptions> copyHiAIOptions(const Backend::Info& info) {
    std::shared_ptr<HiAIBackendOptions> out(new HiAIBackendOptions);
    if (info.user != nullptr) out->config = *info.user;
    if (info.user == nullptr || info.user->sharedContext == nullptr) return out;
    const auto* config = static_cast<const MNNHiAIBackendConfigV1*>(info.user->sharedContext);
    if (validateHiAIConfig(config) != MNN_HIAI_STATUS_SUCCESS) return nullptr;
    const auto copy = [](const char* value) { return value == nullptr ? std::string() : std::string(value); };
    out->explicitConfig = true;
    out->runtimeLibraryDirectory = copy(config->runtimeLibraryDirectory);
    out->acceleratorCacheDirectory = copy(config->acceleratorCacheDirectory);
    out->autoTuning = config->autoTuning != 0;
    out->nativeIoContext = config->nativeIoContext;
    out->diagnostics = config->diagnostics;
    out->view = *config;
    out->view.runtimeLibraryDirectory = out->runtimeLibraryDirectory.c_str();
    out->view.acceleratorCacheDirectory = out->acceleratorCacheDirectory.c_str();
    out->config.sharedContext = &out->view;
    return out;
}

class HiAIRejectedExecution : public Execution {
public:
    HiAIRejectedExecution(Backend* backend, ErrorCode code) : Execution(backend), mCode(code) { mNeedAllocIO = false; }
    ErrorCode onResize(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override { return mCode; }
    ErrorCode onExecute(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override { return mCode; }
private:
    ErrorCode mCode;
};

} // namespace MNN
#endif
