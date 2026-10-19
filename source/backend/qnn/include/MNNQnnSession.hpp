#ifndef MNN_QNN_SESSION_HPP
#define MNN_QNN_SESSION_HPP

#include "MNNQnnBackend.h"
#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <string>
#include <vector>

namespace MNN {
struct QnnSessionResult {
    Session* session = nullptr;
    ErrorCode code = NO_ERROR;
    std::string message;
};

/** Create one strict QNN-only Session; the config and diagnostics outlive it. */
inline QnnSessionResult createQnnSession(Interpreter* interpreter,
                                         const ScheduleConfig& schedule,
                                         const MNNQnnBackendApiV1* api) {
    QnnSessionResult result;
    const auto fail = [&result](ErrorCode code, const std::string& message) {
        result.code = code;
        result.message = message;
        return result;
    };
    if (interpreter == nullptr || api == nullptr || api->structSize < sizeof(*api) ||
        api->abiVersion != MNN_QNN_PLUGIN_ABI_VERSION || api->forwardType != MNN_FORWARD_QNN ||
        api->registerRuntime == nullptr || api->validateModel == nullptr ||
        schedule.type != MNN_FORWARD_QNN || schedule.backupType != schedule.type ||
        schedule.backendConfig == nullptr || schedule.backendConfig->sharedContext == nullptr) {
        return fail(INVALID_VALUE, "QNN requires a matching plugin/config and disabled fallback");
    }
    const auto* config = static_cast<const MNNQnnBackendConfigV1*>(schedule.backendConfig->sharedContext);
    auto* diagnostics = config->diagnostics;
    if (config->structSize < sizeof(*config) || config->version != MNN_QNN_CONFIG_VERSION ||
        diagnostics == nullptr || diagnostics->structSize < sizeof(*diagnostics) ||
        diagnostics->version != MNN_QNN_CONFIG_VERSION) {
        return fail(INVALID_VALUE, "Invalid QNN configuration/diagnostics header");
    }
    diagnostics->code = NO_ERROR;
    diagnostics->readyCount = 0;
    diagnostics->message[0] = '\0';
    const auto model = interpreter->getModelBuffer();
    if (api->validateModel(config, model.first, model.second) != MNN_QNN_STATUS_SUCCESS) {
        return fail(INVALID_VALUE, diagnostics->message[0] ? diagnostics->message : "QNN model validation failed");
    }
    if (api->registerRuntime() != MNN_QNN_STATUS_SUCCESS) {
        return fail(NO_EXECUTION, "QNN Runtime registration failed");
    }
    std::vector<ScheduleConfig> configs(1, schedule);
    const auto runtimes = Interpreter::createRuntime(configs);
    const auto found = runtimes.first.find(schedule.type);
    if (found == runtimes.first.end() || !found->second) {
        return fail(NO_EXECUTION, diagnostics->message[0] ? diagnostics->message : "QNN Runtime could not be created");
    }
    Session* session = interpreter->createMultiPathSession(configs, runtimes);
    if (session == nullptr) return fail(NO_EXECUTION, "QNN session scheduling failed");
    int resizeStatus = -1;
    const bool ready = interpreter->getSessionInfo(session, Interpreter::RESIZE_STATUS, &resizeStatus) &&
                       resizeStatus == 0 && diagnostics->code == NO_ERROR && diagnostics->readyCount > 0;
    if (!ready) {
        const ErrorCode code = diagnostics->code != NO_ERROR ? static_cast<ErrorCode>(diagnostics->code) : COMPUTE_SIZE_ERROR;
        const std::string message = diagnostics->message[0] ? diagnostics->message : "QNN initial resize failed";
        interpreter->releaseSession(session);
        return fail(code, message);
    }
    result.session = session;
    return result;
}
} // namespace MNN
#endif
