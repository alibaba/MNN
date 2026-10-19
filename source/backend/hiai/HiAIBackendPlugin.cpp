#include "backend/HiAIBackendConfig.hpp"
#include "include/MNNHiAIBackend.h"
#include "MNN_generated.h"
#include <limits>

namespace MNN { bool registerHiAIRuntimeCreator(); }

extern "C" MNNHiAIStatus MNNRegisterHiAIRuntime(void) {
    return MNN::registerHiAIRuntimeCreator()
               ? MNN_HIAI_STATUS_SUCCESS : MNN_HIAI_STATUS_UNAVAILABLE;
}

namespace {
MNNHiAIStatus validateModel(const MNNHiAIBackendConfigV1* config, const void* data, size_t size) {
    const auto status = MNN::validateHiAIConfig(config);
    if (status != MNN_HIAI_STATUS_SUCCESS) return status;
    const auto fail = [config](const char* message) {
        MNN::hiaiDiagnostic(config->diagnostics, MNN_HIAI_STAGE_CONFIG, MNN::INVALID_VALUE, message);
        return MNN_HIAI_STATUS_INVALID_ARGUMENT;
    };
    if (data == nullptr || size < sizeof(uint32_t) ||
        size >= static_cast<size_t>(std::numeric_limits<flatbuffers::soffset_t>::max()))
        return fail("Invalid HiAI model buffer size");
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data), size);
    if (!MNN::VerifyNetBuffer(verifier)) return fail("Invalid HiAI model FlatBuffer");
    const auto* net = MNN::GetNet(data);
    return net->oplists() != nullptr && net->oplists()->size() != 0
               ? MNN_HIAI_STATUS_SUCCESS : fail("HiAI model has no operators");
}
} // namespace

extern "C" const MNNHiAIBackendApiV1* MNNGetHiAIBackendApiV1(void) {
    static const MNNHiAIBackendApiV1 api = {
        sizeof(MNNHiAIBackendApiV1), MNN_HIAI_PLUGIN_ABI_VERSION, MNN_FORWARD_USER_0,
        MNNRegisterHiAIRuntime, validateModel,
    };
    return &api;
}
