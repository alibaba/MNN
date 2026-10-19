#ifndef MNN_HIAI_DYNAMIC_LOADER_HPP
#define MNN_HIAI_DYNAMIC_LOADER_HPP

#include <string>

namespace MNN {

enum class HiAIDynamicLoadScope {
    GraphOnly,
    Full,
};

// Loads either the GE graph ABI used by HCL or the complete legacy HiAI
// graph/runtime ABI. Successful handles stay alive for the process lifetime
// because GE objects retain vendor vtable/function pointers. The first
// successful ABI provider is process-wide and is reused by later sessions; all
// delivered clients therefore need the same C++ ABI. An empty runtime
// directory preserves the legacy soname lookup, while an explicit vendor
// session may provide its verified resource directory for deterministic local
// loading when it is the first HiAI user in the process.
bool loadHiAIDynamicSymbols(HiAIDynamicLoadScope scope,
                            bool requireExtended,
                            const std::string& runtimeLibraryDirectory,
                            std::string* error = nullptr);

} // namespace MNN

#endif // MNN_HIAI_DYNAMIC_LOADER_HPP
