#ifndef MNN_QNN_PLUGIN_HPP
#define MNN_QNN_PLUGIN_HPP

#include "MNNQnnBackend.h"
#include <mutex>
#include <string>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace MNN {
/** Load the QNN plugin once. A successful handle remains loaded for MNN's lifetime. */
inline const MNNQnnBackendApiV1* loadQnnBackendPlugin(std::string* error = nullptr) {
#if defined(_WIN32)
    if (error) *error = "Use the QNN plugin entry point directly on Windows";
    return nullptr;
#else
    struct State {
        std::mutex mutex;
        void* handle = nullptr;
        const MNNQnnBackendApiV1* api = nullptr;
    };
    static State* state = new State;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->api != nullptr) return state->api;
    const char* library = "libMNN_Backend_QNN.so";
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* reason = dlerror();
        if (error) *error = std::string("Could not load ") + library + ": " + (reason ? reason : "unknown error");
        return nullptr;
    }
    const auto getApi = reinterpret_cast<MNNGetQnnBackendApiV1Fn>(dlsym(handle, "MNNGetQnnBackendApiV1"));
    const auto* api = getApi ? getApi() : nullptr;
    if (api == nullptr || api->structSize < sizeof(*api) ||
        api->abiVersion != MNN_QNN_PLUGIN_ABI_VERSION ||
        api->forwardType != MNN_FORWARD_QNN || api->registerRuntime == nullptr ||
        api->validateModel == nullptr) {
        if (error) *error = std::string("Invalid QNN plugin ABI: ") + library;
        dlclose(handle);
        return nullptr;
    }
    state->handle = handle;
    state->api = api;
    return api;
#endif
}
} // namespace MNN
#endif
