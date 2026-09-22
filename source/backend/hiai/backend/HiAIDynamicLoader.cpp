#include "HiAIDynamicLoader.hpp"

#include <core/Macro.h>

#include <dlfcn.h>
#include <pthread.h>

#include <cstring>
#include <string>

#define MNN_HIAI_STRINGIFY_IMPL(value) #value
#define MNN_HIAI_STRINGIFY(value) MNN_HIAI_STRINGIFY_IMPL(value)

namespace {

enum {
    kHiAIFunctionCount = 0
#define MNN_HIAI_FUNCTION(symbol) +1
#define MNN_HIAI_OBJECT(symbol, size64, size32)
#include "HiAIDynamicSymbols.inc"
#undef MNN_HIAI_OBJECT
#undef MNN_HIAI_FUNCTION
};

} // namespace

extern "C" {

__attribute__((visibility("hidden"))) void* mnn_hiai_dynamic_function_table[kHiAIFunctionCount] = {};

} // extern "C"

#define MNN_HIAI_FUNCTION(symbol)
#define MNN_HIAI_OBJECT(symbol, size64, size32)                                                            \
    extern "C" __attribute__((visibility("hidden"))) unsigned char mnn_hiai_dynamic_object_##symbol[] asm( \
        MNN_HIAI_STRINGIFY(symbol));
#include "HiAIDynamicSymbols.inc"
#undef MNN_HIAI_OBJECT
#undef MNN_HIAI_FUNCTION

namespace MNN {
namespace {

struct DynamicObject {
    const char* symbol;
    void* destination;
    size_t size;
};

constexpr const char* kHiAIGraphLibraries[] = {
    "libhiai_ir.so",
};

constexpr const char* kHiAIFullLibraries[] = {
    "libhiai_ir.so",
    "libhiai_ir_build.so",
    "libhiai.so",
};

const char* const kHiAIFunctions[] = {
#define MNN_HIAI_FUNCTION(symbol) MNN_HIAI_STRINGIFY(symbol),
#define MNN_HIAI_OBJECT(symbol, size64, size32)
#include "HiAIDynamicSymbols.inc"
#undef MNN_HIAI_OBJECT
#undef MNN_HIAI_FUNCTION
};
static_assert(sizeof(kHiAIFunctions) / sizeof(kHiAIFunctions[0]) == kHiAIFunctionCount,
              "HiAI function table and symbol list must stay aligned");

DynamicObject kHiAIObjects[] = {
#define MNN_HIAI_FUNCTION(symbol)
#if __SIZEOF_POINTER__ == 8
#define MNN_HIAI_OBJECT(symbol, size64, size32) {MNN_HIAI_STRINGIFY(symbol), mnn_hiai_dynamic_object_##symbol, size64},
#else
#define MNN_HIAI_OBJECT(symbol, size64, size32) {MNN_HIAI_STRINGIFY(symbol), mnn_hiai_dynamic_object_##symbol, size32},
#endif
#include "HiAIDynamicSymbols.inc"
#undef MNN_HIAI_OBJECT
#undef MNN_HIAI_FUNCTION
};

pthread_mutex_t gHiAILoaderMutex = PTHREAD_MUTEX_INITIALIZER;
bool gHiAIGraphLoaded = false;
bool gHiAIFullLoaded = false;
bool gHiAIExtendedLoaded = false;
bool gHiAIProviderSelected = false;

std::string& hiAILibraryDirectory() {
    static auto* directory = new std::string;
    return *directory;
}

bool isGraphFunction(const char* symbol) {
    return std::strncmp(symbol, "_ZN2ge", 6) == 0 ||
           std::strncmp(symbol, "_ZNK2ge", 7) == 0;
}

bool isExtendedSymbol(const char* symbol) {
    return std::strcmp(
               symbol,
               "_ZN4hiai18AiModelMngerClient16GetModelAippParaERKNSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEjRNS1_6vectorINS1_10shared_ptrINS_8AippParaEEENS5_ISD_EEEE") == 0 ||
           std::strcmp(
               symbol,
               "_ZN4hiai8AiTensor4InitERKNS_12NativeHandleEPKNS_15TensorDimensionENS_13HIAI_DataTypeE") == 0;
}

void* resolveFromHandles(void* const* handles, size_t handleCount, const char* symbol) {
    for (size_t i = 0; i < handleCount; ++i) {
        (void)dlerror();
        void* address = dlsym(handles[i], symbol);
        const char* error = dlerror();
        if (address != nullptr && error == nullptr) {
            return address;
        }
    }
    return nullptr;
}

std::string libraryPath(const std::string& directory, const char* library) {
    if (directory.empty()) {
        return library;
    }
    if (directory.back() == '/') {
        return directory + library;
    }
    return directory + "/" + library;
}

void closeHandles(void** handles, size_t handleCount) {
    while (handleCount > 0) {
        --handleCount;
        if (handles[handleCount] != nullptr) {
            dlclose(handles[handleCount]);
            handles[handleCount] = nullptr;
        }
    }
}

} // namespace

bool loadHiAIDynamicSymbols(HiAIDynamicLoadScope scope,
                            bool requireExtended,
                            const std::string& runtimeLibraryDirectory,
                            std::string* error) {
    pthread_mutex_lock(&gHiAILoaderMutex);
    if (error != nullptr) {
        error->clear();
    }
    const bool loadFull = scope == HiAIDynamicLoadScope::Full;
    if ((!loadFull && gHiAIGraphLoaded) ||
        (loadFull && gHiAIFullLoaded &&
         (!requireExtended || gHiAIExtendedLoaded))) {
        pthread_mutex_unlock(&gHiAILoaderMutex);
        return true;
    }

    const char* const* libraries = loadFull ? kHiAIFullLibraries
                                            : kHiAIGraphLibraries;
    const size_t libraryCount = loadFull
        ? sizeof(kHiAIFullLibraries) / sizeof(kHiAIFullLibraries[0])
        : sizeof(kHiAIGraphLibraries) / sizeof(kHiAIGraphLibraries[0]);
    const std::string& selectedDirectory = gHiAIProviderSelected
        ? hiAILibraryDirectory()
        : runtimeLibraryDirectory;
    void* handles[sizeof(kHiAIFullLibraries) /
                  sizeof(kHiAIFullLibraries[0])] = {};
    for (size_t step = 0; step < libraryCount; ++step) {
        // Packaged V320 libraries depend on both the graph and client DSOs.
        // Open those dependencies before libhiai_ir_build.so. The empty-dir
        // path keeps the original soname order used by legacy callers.
        size_t index = step;
        if (loadFull && !selectedDirectory.empty()) {
            constexpr size_t kPackagedFullOpenOrder[] = {0, 2, 1};
            index = kPackagedFullOpenOrder[step];
        }
        const std::string path =
            libraryPath(selectedDirectory, libraries[index]);
        handles[index] = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handles[index] == nullptr) {
            const char* dynamicError = dlerror();
            if (error != nullptr) {
                *error = std::string("failed to open ") + path;
                if (dynamicError != nullptr) {
                    *error += std::string(": ") + dynamicError;
                }
            }
            closeHandles(handles, libraryCount);
            pthread_mutex_unlock(&gHiAILoaderMutex);
            return false;
        }
    }

    void* functions[kHiAIFunctionCount] = {};
    bool allExtendedSymbolsAvailable = true;
    size_t resolvedFunctionCount = 0;
    for (size_t i = 0; i < kHiAIFunctionCount; ++i) {
        if (!loadFull && !isGraphFunction(kHiAIFunctions[i])) {
            continue;
        }
        functions[i] = resolveFromHandles(handles, libraryCount, kHiAIFunctions[i]);
        if (functions[i] == nullptr) {
            if (isExtendedSymbol(kHiAIFunctions[i])) {
                allExtendedSymbolsAvailable = false;
                if (!requireExtended) {
                    continue;
                }
            }
            if (error != nullptr) {
                *error = std::string("missing HiAI symbol: ") + kHiAIFunctions[i];
            }
            closeHandles(handles, libraryCount);
            pthread_mutex_unlock(&gHiAILoaderMutex);
            return false;
        }
        ++resolvedFunctionCount;
    }

    constexpr size_t objectCount = sizeof(kHiAIObjects) / sizeof(kHiAIObjects[0]);
    void* objects[objectCount] = {};
    if (loadFull) {
        for (size_t i = 0; i < objectCount; ++i) {
            objects[i] = resolveFromHandles(handles, libraryCount, kHiAIObjects[i].symbol);
            if (objects[i] == nullptr) {
                if (error != nullptr) {
                    *error = std::string("missing HiAI object: ") +
                             kHiAIObjects[i].symbol;
                }
                closeHandles(handles, libraryCount);
                pthread_mutex_unlock(&gHiAILoaderMutex);
                return false;
            }
        }
    }

    // A graph-only load can be upgraded later for a V320 session. Never
    // replace an address that may already be used by a live GE/runtime object;
    // a different address means the process contains incompatible client DSOs.
    for (size_t i = 0; i < kHiAIFunctionCount; ++i) {
        if (mnn_hiai_dynamic_function_table[i] != nullptr &&
            functions[i] != nullptr &&
            mnn_hiai_dynamic_function_table[i] != functions[i]) {
            if (error != nullptr) {
                *error = std::string("incompatible HiAI symbol provider: ") +
                         kHiAIFunctions[i];
            }
            closeHandles(handles, libraryCount);
            pthread_mutex_unlock(&gHiAILoaderMutex);
            return false;
        }
    }
    for (size_t i = 0; i < kHiAIFunctionCount; ++i) {
        if (mnn_hiai_dynamic_function_table[i] == nullptr) {
            mnn_hiai_dynamic_function_table[i] = functions[i];
        }
    }
    if (loadFull) {
        for (size_t i = 0; i < objectCount; ++i) {
            std::memcpy(kHiAIObjects[i].destination, objects[i],
                        kHiAIObjects[i].size);
        }
    }
    // Do not dlclose successful handles. Function pointers and copied vtables
    // remain live until process exit.
    gHiAIGraphLoaded = true;
    if (loadFull) {
        gHiAIFullLoaded = true;
        gHiAIExtendedLoaded = allExtendedSymbolsAvailable;
    }
    if (!gHiAIProviderSelected) {
        hiAILibraryDirectory() = runtimeLibraryDirectory;
        gHiAIProviderSelected = true;
    }
    const bool extendedLoaded = gHiAIExtendedLoaded;
    pthread_mutex_unlock(&gHiAILoaderMutex);

    MNN_PRINT("[NPU] HiAI dynamic ABI loaded: scope=%s functions=%lu objects=%lu extended=%d\n",
              loadFull ? "full" : "graph",
              static_cast<unsigned long>(resolvedFunctionCount),
              static_cast<unsigned long>(loadFull ? objectCount : 0),
              extendedLoaded ? 1 : 0);
    return true;
}

} // namespace MNN
