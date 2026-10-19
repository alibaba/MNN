#ifndef MNN_HIAI_PLUGIN_HPP
#define MNN_HIAI_PLUGIN_HPP

#include "MNNHiAIBackend.h"
#include <mutex>
#include <string>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace MNN {
/** Load the HiAI plugin once. A successful handle remains loaded for MNN's lifetime. */
inline const MNNHiAIBackendApiV1* loadHiAIBackendPlugin(std::string* error = nullptr) {
#if defined(_WIN32)
    if (error) *error = "Use the HiAI plugin entry point directly on Windows";
    return nullptr;
#else
    struct State {
        std::mutex mutex;
        void* handle = nullptr;
        const MNNHiAIBackendApiV1* api = nullptr;
    };
    static State* state = new State;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->api != nullptr) return state->api;
    const char* library = "libMNN_Backend_HiAI.so";
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* reason = dlerror();
        if (error) *error = std::string("Could not load ") + library + ": " + (reason ? reason : "unknown error");
        return nullptr;
    }
    const auto getApi = reinterpret_cast<MNNGetHiAIBackendApiV1Fn>(dlsym(handle, "MNNGetHiAIBackendApiV1"));
    const auto* api = getApi ? getApi() : nullptr;
    if (api == nullptr || api->structSize < sizeof(*api) ||
        api->abiVersion != MNN_HIAI_PLUGIN_ABI_VERSION ||
        api->forwardType != MNN_FORWARD_USER_0 || api->registerRuntime == nullptr ||
        api->validateModel == nullptr) {
        if (error) *error = std::string("Invalid HiAI plugin ABI: ") + library;
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
