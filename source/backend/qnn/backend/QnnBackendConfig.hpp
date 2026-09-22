#ifndef MNN_QNN_BACKEND_CONFIG_HPP
#define MNN_QNN_BACKEND_CONFIG_HPP

#include "../include/MNNQnnBackend.h"
#include <MNN/ErrorCode.hpp>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {

inline void qnnDiagnostic(MNNQnnDiagnosticsV1* out, MNNQnnStage stage, ErrorCode code,
                          const char* message, int64_t nativeCode = 0) {
    if (out == nullptr || out->structSize < sizeof(*out) || out->version != MNN_QNN_CONFIG_VERSION) return;
    out->stage = stage;
    out->code = code;
    out->nativeCode = nativeCode;
    std::snprintf(out->message, sizeof(out->message), "%s", message == nullptr ? "" : message);
}

struct QnnBackendOptions {
    BackendConfig config;
    MNNQnnBackendConfigV1 view = {};
    bool explicitConfig = false;
    uint32_t flags = 0;
    MNNQnnRuntimeKind runtime = MNN_QNN_RUNTIME_AUTO;
    bool directInt8NhwcIo = false;
    std::string runtimeLibraryDirectory;
    std::string runtimeManifestPath;
    std::string acceleratorResourceDirectory;
    std::string modelDirectory;
    std::string opPackageInterfaceProvider;
    std::string opPackageName;
    MNNQnnDiagnosticsV1* diagnostics = nullptr;

    void report(MNNQnnStage stage, ErrorCode code, const char* message, int64_t nativeCode = 0) const {
        qnnDiagnostic(diagnostics, stage, code, message, nativeCode);
    }
    void ready() const {
        if (diagnostics != nullptr && diagnostics->structSize >= sizeof(*diagnostics) &&
            diagnostics->version == MNN_QNN_CONFIG_VERSION) ++diagnostics->readyCount;
        report(MNN_QNN_STAGE_RESIZE, NO_ERROR, "QNN graph ready");
    }
};

class QnnCopyDiagnostic {
public:
    QnnCopyDiagnostic(const QnnBackendOptions* options, const ErrorCode& code) : mOptions(options), mCode(code) {}
    ~QnnCopyDiagnostic() {
        if (mOptions != nullptr && mCode != NO_ERROR)
            mOptions->report(MNN_QNN_STAGE_COPY, mCode, "QNN host I/O copy failed");
    }
private:
    const QnnBackendOptions* mOptions;
    const ErrorCode& mCode;
};

inline MNNQnnStatus validateQnnConfig(const MNNQnnBackendConfigV1* config) {
    if (config == nullptr || config->structSize < sizeof(*config) || config->version != MNN_QNN_CONFIG_VERSION)
        return MNN_QNN_STATUS_INVALID_ARGUMENT;
    const uint32_t allowed = MNN_QNN_CONFIG_OFFLINE_CONTEXT | MNN_QNN_CONFIG_DUMP_OUTPUTS;
    if ((config->flags & ~allowed) != 0 || config->runtime < MNN_QNN_RUNTIME_AUTO ||
        config->runtime > MNN_QNN_RUNTIME_DSP || config->directInt8NhwcIo > 1) {
        qnnDiagnostic(config->diagnostics, MNN_QNN_STAGE_CONFIG, NOT_SUPPORT, "Unsupported QNN option");
        return MNN_QNN_STATUS_UNSUPPORTED_OPTION;
    }
    if (config->diagnostics != nullptr &&
        (config->diagnostics->structSize < sizeof(MNNQnnDiagnosticsV1) ||
         config->diagnostics->version != MNN_QNN_CONFIG_VERSION))
        return MNN_QNN_STATUS_INVALID_ARGUMENT;
    return MNN_QNN_STATUS_SUCCESS;
}

inline std::shared_ptr<QnnBackendOptions> copyQnnOptions(const Backend::Info& info) {
    std::shared_ptr<QnnBackendOptions> out(new QnnBackendOptions);
    if (info.user != nullptr) out->config = *info.user;
    if (info.type != MNN_FORWARD_QNN) {
        // Legacy callers use the flags member of the union, never the
        // backend-specific sharedContext structure introduced for TBLive.
        if (info.user != nullptr) out->flags = info.user->flags & MNN_QNN_CONFIG_DUMP_OUTPUTS;
        out->config.sharedContext = nullptr;
        return out;
    }
    if (info.user == nullptr || info.user->sharedContext == nullptr) return out;
    // Historical converter flag stored directly in sharedContext; do not dereference it.
    if (reinterpret_cast<uintptr_t>(info.user->sharedContext) == (1u << 16)) {
        out->flags = MNN_QNN_CONFIG_DUMP_OUTPUTS;
        out->config.sharedContext = nullptr;
        return out;
    }
    const auto* config = static_cast<const MNNQnnBackendConfigV1*>(info.user->sharedContext);
    if (validateQnnConfig(config) != MNN_QNN_STATUS_SUCCESS) return nullptr;
    const auto copy = [](const char* value) { return value == nullptr ? std::string() : std::string(value); };
    out->explicitConfig = true;
    out->flags = config->flags;
    out->runtime = config->runtime;
    out->directInt8NhwcIo = config->directInt8NhwcIo != 0;
    out->runtimeLibraryDirectory = copy(config->runtimeLibraryDirectory);
    out->runtimeManifestPath = copy(config->runtimeManifestPath);
    out->acceleratorResourceDirectory = copy(config->acceleratorResourceDirectory);
    out->modelDirectory = copy(config->modelDirectory);
    out->opPackageInterfaceProvider = copy(config->opPackageInterfaceProvider);
    out->opPackageName = copy(config->opPackageName);
    out->diagnostics = config->diagnostics;
    out->view = *config;
    out->view.runtimeLibraryDirectory = out->runtimeLibraryDirectory.c_str();
    out->view.runtimeManifestPath = out->runtimeManifestPath.c_str();
    out->view.acceleratorResourceDirectory = out->acceleratorResourceDirectory.c_str();
    out->view.modelDirectory = out->modelDirectory.c_str();
    out->view.opPackageInterfaceProvider = out->opPackageInterfaceProvider.c_str();
    out->view.opPackageName = out->opPackageName.c_str();
    out->config.sharedContext = &out->view;
    return out;
}

class QnnRejectedExecution : public Execution {
public:
    QnnRejectedExecution(Backend* backend, ErrorCode code) : Execution(backend), mCode(code) { mNeedAllocIO = false; }
    ErrorCode onResize(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override { return mCode; }
    ErrorCode onExecute(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override { return mCode; }
private:
    ErrorCode mCode;
};

} // namespace MNN
#endif
