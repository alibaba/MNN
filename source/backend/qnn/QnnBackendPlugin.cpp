#include "backend/QnnBackendConfig.hpp"
#include "include/MNNQnnBackend.h"
#include "MNN_generated.h"
#include <limits>

namespace MNN { bool registerQNNRuntimeCreator(bool, bool, MNNForwardType*); }

extern "C" MNNQnnStatus MNNRegisterQNNRuntime(void) {
    MNNForwardType type = MNN_FORWARD_QNN;
    return MNN::registerQNNRuntimeCreator(false, true, &type)
               ? MNN_QNN_STATUS_SUCCESS : MNN_QNN_STATUS_UNAVAILABLE;
}

namespace {
MNNQnnStatus validateModel(const MNNQnnBackendConfigV1* config, const void* data, size_t size) {
    const auto status = MNN::validateQnnConfig(config);
    if (status != MNN_QNN_STATUS_SUCCESS) return status;
    const auto fail = [config](const char* message) {
        MNN::qnnDiagnostic(config->diagnostics, MNN_QNN_STAGE_CONFIG, MNN::INVALID_VALUE, message);
        return MNN_QNN_STATUS_INVALID_ARGUMENT;
    };
    if (data == nullptr || size < sizeof(uint32_t) ||
        size >= static_cast<size_t>(std::numeric_limits<flatbuffers::soffset_t>::max()))
        return fail("Invalid QNN model buffer size");
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data), size);
    if (!MNN::VerifyNetBuffer(verifier)) return fail("Invalid QNN model FlatBuffer");
    const auto* net = MNN::GetNet(data);
    if (net->oplists() == nullptr || net->oplists()->size() == 0) return fail("QNN model has no operators");
    if ((config->flags & MNN_QNN_CONFIG_OFFLINE_CONTEXT) == 0) return MNN_QNN_STATUS_SUCCESS;
    size_t plugins = 0;
    for (const auto* op : *net->oplists()) {
        if (op->type() == MNN::OpType_Input) continue;
        if (op->type() != MNN::OpType_Plugin || op->main_as_Plugin() == nullptr)
            return fail("QNN offline mode requires a Plugin(QNN) wrapper, not an online graph");
        const auto* plugin = op->main_as_Plugin();
        if (plugin->type() == nullptr || plugin->type()->str() != "QNN" || plugin->attr() == nullptr)
            return fail("Invalid QNN offline Plugin descriptor");
        bool path = false, inputs = false, outputs = false, graphs = false;
        for (const auto* attr : *plugin->attr()) {
            if (attr->key() == nullptr) return fail("QNN offline attribute has no name");
            const auto key = attr->key()->str();
            if (key == "path") path = attr->s() != nullptr && attr->s()->size() != 0;
            if (key == "inputs") inputs = attr->list() != nullptr && attr->list()->s() != nullptr;
            if (key == "outputs") outputs = attr->list() != nullptr && attr->list()->s() != nullptr;
            if (key == "allGraphName")
                graphs = attr->list() != nullptr && attr->list()->s() != nullptr && attr->list()->s()->size() != 0;
        }
        if (!path || !inputs || !outputs || !graphs)
            return fail("QNN offline wrapper is missing path, inputs, outputs or allGraphName");
        ++plugins;
    }
    return plugins == 1 ? MNN_QNN_STATUS_SUCCESS
                        : fail("QNN offline mode requires exactly one Plugin(QNN)");
}
} // namespace

extern "C" const MNNQnnBackendApiV1* MNNGetQnnBackendApiV1(void) {
    static const MNNQnnBackendApiV1 api = {
        sizeof(MNNQnnBackendApiV1), MNN_QNN_PLUGIN_ABI_VERSION, MNN_FORWARD_QNN,
        MNNRegisterQNNRuntime, validateModel,
    };
    return &api;
}
