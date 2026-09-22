//
// Bridge to the public HiAI HCL V600 C ABI exposed by recent
// Kirin devices. The ABI names and signatures are from Huawei's Apache-2.0
// hiai_ddk. Everything is resolved dynamically so the platform-wide HiAI
// resource package remains usable on older Kirin phones.
//

#include "HiaiHclV600Runtime.hpp"

#include <dlfcn.h>

#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <unistd.h>

#include <core/Macro.h>

namespace MNN {
namespace {

constexpr int32_t kHiaiSuccess = 0;
constexpr const char* kHclLibrary = "libai_fmk_hcl_model_runtime.so";
constexpr const char* kHclPluginLibrary = "libhiai_hcl_model_runtime.so";
constexpr const char* kTensorLibrary = "libai_fmk_tensor.so";
constexpr const char* kMinimumHclVersion = "100.600.000.000";

std::string UniqueCacheTemporaryFile(const std::string& cacheFile) {
    static std::atomic<std::uint64_t> sequence{0};
    return cacheFile + ".tmp-" + std::to_string(getpid()) + "-" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

struct HclModelBuildOptions;
struct HclModelDeviceConfig;
struct HclTuningConfig;
struct HclBuiltModel;
struct HclModelManager;
struct HclModelInitOptions;
struct HclTensorDesc;
struct HclTensorBuffer;
struct HclNativeHandle;

template <typename T>
bool Resolve(void* library, const char* name, T* result, std::string* error) {
    *result = reinterpret_cast<T>(dlsym(library, name));
    if (*result != nullptr) {
        return true;
    }
    if (error != nullptr) {
        const char* detail = dlerror();
        *error = std::string("missing ") + name;
        if (detail != nullptr) {
            *error += std::string(": ") + detail;
        }
    }
    return false;
}

bool VersionAtLeast(const std::string& actual, const char* minimum) {
    return actual.size() >= std::strlen(minimum) && actual.compare(minimum) >= 0;
}

void* OpenFromEnhancePlugin(const char* library) {
    // Newer Huawei DDKs expose Android-compatible compute-capability plugins
    // through libhiai_enhance and the com.huawei.hiai service. The JNI owner
    // enables this channel with its application Context before MNN creates
    // the backend.
    void* enhance = dlopen("libhiai_enhance.so", RTLD_NOW | RTLD_LOCAL);
    if (enhance == nullptr) {
        return nullptr;
    }
    using GetPluginHandleFn = long (*)(const char*);
    auto getPluginHandle = reinterpret_cast<GetPluginHandleFn>(dlsym(enhance, "GetPluginSoHandleDefault"));
    void* result = getPluginHandle == nullptr ? nullptr : reinterpret_cast<void*>(getPluginHandle(library));
    dlclose(enhance);
    return result;
}

} // namespace

struct HiaiHclV600Runtime::Impl {
    using GetVersionFn = const char* (*)();
    using ModelBuildOptionsCreateFn = HclModelBuildOptions* (*)();
    using ModelBuildOptionsDestroyFn = void (*)(HclModelBuildOptions**);
    using ModelBuildOptionsSetPrecisionFn = int32_t (*)(HclModelBuildOptions*, int32_t);
    using TuningConfigCreateFn = HclTuningConfig* (*)();
    using TuningConfigSetOptionFn = int32_t (*)(HclTuningConfig*, int32_t);
    using TuningConfigSetCacheDirFn = int32_t (*)(HclTuningConfig*, const char*);
    using TuningConfigDestroyFn = void (*)(HclTuningConfig**);
    using ModelBuildOptionsSetTuningConfigFn = int32_t (*)(HclModelBuildOptions*, HclTuningConfig*);
    using ModelDeviceConfigCreateFn = HclModelDeviceConfig* (*)();
    using ModelDeviceConfigSetModeFn = void (*)(HclModelDeviceConfig*, int32_t);
    using ModelDeviceConfigSetFallbackFn = void (*)(HclModelDeviceConfig*, int32_t);
    using ModelDeviceConfigSetOrderFn = void (*)(HclModelDeviceConfig*, size_t, int32_t*);
    using ModelBuildOptionsSetDeviceConfigFn = void (*)(HclModelBuildOptions*, HclModelDeviceConfig*);
    using ModelBuilderBuildFn = int32_t (*)(const HclModelBuildOptions*, const char*, const void*, size_t,
                                            HclBuiltModel**);
    using BuiltModelDestroyFn = void (*)(HclBuiltModel**);
    using BuiltModelSaveToFileFn = int32_t (*)(const HclBuiltModel*, const char*);
    using BuiltModelRestoreFromFileFn = HclBuiltModel* (*)(const char*);
    using BuiltModelCheckCompatibilityFn = int32_t (*)(const HclBuiltModel*, int32_t*);
    using BuiltModelGetTensorNumFn = int32_t (*)(const HclBuiltModel*);
    using BuiltModelGetTensorDescFn = HclTensorDesc* (*)(const HclBuiltModel*, size_t);
    using ModelManagerCreateFn = HclModelManager* (*)();
    using ModelManagerDestroyFn = void (*)(HclModelManager**);
    using ModelInitOptionsCreateFn = HclModelInitOptions* (*)();
    using ModelInitOptionsDestroyFn = void (*)(HclModelInitOptions**);
    using ModelInitOptionsSetModeFn = void (*)(HclModelInitOptions*, int32_t);
    using ModelManagerInitV2Fn = int32_t (*)(HclModelManager*, const HclModelInitOptions*, const HclBuiltModel*,
                                             const void*);
    using ModelManagerSetPriorityFn = int32_t (*)(HclModelManager*, int32_t);
    using ModelManagerDeinitFn = int32_t (*)(HclModelManager*);
    using ModelManagerRunV3Fn = int32_t (*)(HclModelManager*, HclTensorBuffer*[], int32_t, HclTensorBuffer*[], int32_t);
    using TensorBufferCreateFn = HclTensorBuffer* (*)(const HclTensorDesc*);
    using TensorBufferCreateNativeFn = HclTensorBuffer* (*)(const HclTensorDesc*, const HclNativeHandle*);
    using TensorBufferDestroyFn = void (*)(HclTensorBuffer**);
    using TensorBufferGetSizeFn = size_t (*)(const HclTensorBuffer*);
    using TensorBufferGetDataFn = void* (*)(const HclTensorBuffer*);
    using TensorDescDestroyFn = void (*)(HclTensorDesc**);
    using NativeHandleCreateFn = HclNativeHandle* (*)(int, int, int);
    using NativeHandleDestroyFn = void (*)(HclNativeHandle**);

    void* hclLibrary = nullptr;
    void* tensorLibrary = nullptr;
    bool ownsHclLibrary = false;
    bool ownsTensorLibrary = false;
    std::string runtimeVersion;
    std::string error;
    bool enableAutoTuning = false;
    std::string tuningCacheDirectory;

    GetVersionFn getVersion = nullptr;
    ModelBuildOptionsCreateFn createBuildOptions = nullptr;
    ModelBuildOptionsDestroyFn destroyBuildOptions = nullptr;
    ModelBuildOptionsSetPrecisionFn setPrecision = nullptr;
    TuningConfigCreateFn createTuningConfig = nullptr;
    TuningConfigSetOptionFn setTuningMode = nullptr;
    TuningConfigSetOptionFn setTuningObjective = nullptr;
    TuningConfigSetCacheDirFn setTuningCacheDir = nullptr;
    TuningConfigSetOptionFn setTuningMemoryReusePlan = nullptr;
    TuningConfigDestroyFn destroyTuningConfig = nullptr;
    ModelBuildOptionsSetTuningConfigFn setTuningConfig = nullptr;
    ModelDeviceConfigCreateFn createDeviceConfig = nullptr;
    ModelDeviceConfigSetModeFn setDeviceConfigMode = nullptr;
    ModelDeviceConfigSetFallbackFn setFallbackMode = nullptr;
    ModelDeviceConfigSetOrderFn setModelDeviceOrder = nullptr;
    ModelBuildOptionsSetDeviceConfigFn setModelDeviceConfig = nullptr;
    ModelBuilderBuildFn buildV2 = nullptr;
    BuiltModelDestroyFn destroyBuiltModel = nullptr;
    BuiltModelSaveToFileFn saveBuiltModelToFile = nullptr;
    BuiltModelRestoreFromFileFn restoreBuiltModelFromFile = nullptr;
    BuiltModelCheckCompatibilityFn checkBuiltModelCompatibility = nullptr;
    BuiltModelGetTensorNumFn getInputCount = nullptr;
    BuiltModelGetTensorNumFn getOutputCount = nullptr;
    BuiltModelGetTensorDescFn getInputDesc = nullptr;
    BuiltModelGetTensorDescFn getOutputDesc = nullptr;
    ModelManagerCreateFn createManager = nullptr;
    ModelManagerDestroyFn destroyManager = nullptr;
    ModelInitOptionsCreateFn createInitOptions = nullptr;
    ModelInitOptionsDestroyFn destroyInitOptions = nullptr;
    ModelInitOptionsSetModeFn setPerfMode = nullptr;
    ModelManagerInitV2Fn initV2 = nullptr;
    ModelManagerSetPriorityFn setPriority = nullptr;
    ModelManagerDeinitFn deinit = nullptr;
    ModelManagerRunV3Fn runV3 = nullptr;
    TensorBufferCreateFn createTensorBuffer = nullptr;
    TensorBufferCreateNativeFn createNativeTensorBuffer = nullptr;
    TensorBufferDestroyFn destroyTensorBuffer = nullptr;
    TensorBufferGetSizeFn getTensorBufferSize = nullptr;
    TensorBufferGetDataFn getTensorBufferData = nullptr;
    TensorDescDestroyFn destroyTensorDesc = nullptr;
    NativeHandleCreateFn createNativeHandle = nullptr;
    NativeHandleDestroyFn destroyNativeHandle = nullptr;

    HclBuiltModel* builtModel = nullptr;
    HclModelManager* manager = nullptr;
    bool managerInitialized = false;
    bool loggedRunV3Success = false;
    std::vector<HclTensorDesc*> inputDescs;
    std::vector<HclTensorDesc*> outputDescs;
    std::vector<HclTensorBuffer*> hostInputs;
    std::vector<HclTensorBuffer*> hostOutputs;
    std::vector<HclTensorBuffer*> nativeInputs;
    std::vector<HclTensorBuffer*> nativeOutputs;
    std::vector<HclNativeHandle*> inputNativeHandles;
    std::vector<HclNativeHandle*> outputNativeHandles;

    ~Impl() {
        (void)release();
    }

    bool release() {
        bool released = resetModel();
        if (ownsTensorLibrary && tensorLibrary != nullptr) {
            if (dlclose(tensorLibrary) != 0) released = false;
            tensorLibrary = nullptr;
            ownsTensorLibrary = false;
        }
        if (ownsHclLibrary && hclLibrary != nullptr) {
            if (dlclose(hclLibrary) != 0) released = false;
            hclLibrary = nullptr;
            ownsHclLibrary = false;
        }
        return released;
    }

    void destroyBuffers(std::vector<HclTensorBuffer*>* buffers) {
        if (destroyTensorBuffer == nullptr)
            return;
        for (auto& buffer : *buffers) {
            if (buffer != nullptr) {
                destroyTensorBuffer(&buffer);
            }
        }
        buffers->clear();
    }

    void destroyHandles(std::vector<HclNativeHandle*>* handles) {
        if (destroyNativeHandle == nullptr)
            return;
        for (auto& handle : *handles) {
            if (handle != nullptr) {
                destroyNativeHandle(&handle);
            }
        }
        handles->clear();
    }

    void destroyDescs(std::vector<HclTensorDesc*>* descs) {
        if (destroyTensorDesc == nullptr)
            return;
        for (auto& desc : *descs) {
            if (desc != nullptr) {
                destroyTensorDesc(&desc);
            }
        }
        descs->clear();
    }

    void clearNativeHandleIo() {
        destroyBuffers(&nativeInputs);
        destroyBuffers(&nativeOutputs);
        destroyHandles(&inputNativeHandles);
        destroyHandles(&outputNativeHandles);
        nativeInputs.resize(hostInputs.size(), nullptr);
        nativeOutputs.resize(hostOutputs.size(), nullptr);
        inputNativeHandles.resize(hostInputs.size(), nullptr);
        outputNativeHandles.resize(hostOutputs.size(), nullptr);
    }

    bool resetModel() {
        bool released = true;
        loggedRunV3Success = false;
        clearNativeHandleIo();
        destroyBuffers(&hostInputs);
        destroyBuffers(&hostOutputs);
        destroyDescs(&inputDescs);
        destroyDescs(&outputDescs);
        if (manager != nullptr) {
            if (managerInitialized) {
                if (deinit == nullptr || deinit(manager) != kHiaiSuccess) {
                    MNN_ERROR("MNN_HIAI_RELEASE_AUDIT: HCL V600 "
                              "ModelManager_Deinit failed\n");
                    released = false;
                }
            }
            managerInitialized = false;
            if (destroyManager != nullptr) {
                destroyManager(&manager);
            }
        }
        if (builtModel != nullptr && destroyBuiltModel != nullptr) {
            destroyBuiltModel(&builtModel);
        }
        return released;
    }

    bool loadSymbols() {
        hclLibrary = dlopen(kHclLibrary, RTLD_NOW | RTLD_LOCAL);
        if (hclLibrary == nullptr) {
            (void)dlerror();
            hclLibrary = OpenFromEnhancePlugin(kHclPluginLibrary);
        }
        if (hclLibrary != nullptr) {
            ownsHclLibrary = true;
        } else {
            // Some vendor clients publish the HCL symbols globally after
            // their legacy bootstrap. This is a final compatibility probe.
            (void)dlerror();
            hclLibrary = RTLD_DEFAULT;
        }
        tensorLibrary = dlopen(kTensorLibrary, RTLD_NOW | RTLD_LOCAL);
        if (tensorLibrary == nullptr) {
            (void)dlerror();
            tensorLibrary = OpenFromEnhancePlugin(kTensorLibrary);
        }
        if (tensorLibrary != nullptr) {
            ownsTensorLibrary = true;
        } else {
            (void)dlerror();
            tensorLibrary = RTLD_DEFAULT;
        }

        const bool requiredSymbolsLoaded =
            Resolve(hclLibrary, "HIAI_MR_GetVersion", &getVersion, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelBuildOptions_Create", &createBuildOptions, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelBuildOptions_Destroy", &destroyBuildOptions, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelBuildOptions_SetPrecisionModeOption", &setPrecision, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelDeviceConfig_Create", &createDeviceConfig, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelDeviceConfig_SetDeviceConfigMode", &setDeviceConfigMode, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelDeviceConfig_SetFallBackMode", &setFallbackMode, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelDeviceConfig_SetModelDeviceOrder", &setModelDeviceOrder, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelBuildOptions_SetModelDeviceConfig", &setModelDeviceConfig, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelBuilder_BuildV2", &buildV2, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_BuiltModel_Destroy", &destroyBuiltModel, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_BuiltModel_GetInputTensorNum", &getInputCount, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_BuiltModel_GetOutputTensorNum", &getOutputCount, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_BuiltModel_GetInputTensorDesc", &getInputDesc, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_BuiltModel_GetOutputTensorDesc", &getOutputDesc, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelManager_Create", &createManager, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelManager_Destroy", &destroyManager, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelInitOptions_Create", &createInitOptions, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelInitOptions_Destroy", &destroyInitOptions, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelInitOptions_SetPerfMode", &setPerfMode, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelManager_InitV2", &initV2, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelManager_Deinit", &deinit, &error) &&
            Resolve(hclLibrary, "HIAI_HCL_ModelManager_RunV3", &runV3, &error) &&
            Resolve(hclLibrary, "HIAI_MR_NDTensorBuffer_CreateFromNDTensorDesc", &createTensorBuffer, &error) &&
            Resolve(hclLibrary, "HIAI_MR_NDTensorBuffer_Destroy", &destroyTensorBuffer, &error) &&
            Resolve(hclLibrary, "HIAI_MR_NDTensorBuffer_GetSize", &getTensorBufferSize, &error) &&
            Resolve(hclLibrary, "HIAI_MR_NDTensorBuffer_GetData", &getTensorBufferData, &error) &&
            Resolve(tensorLibrary, "HIAI_NDTensorDesc_Destroy", &destroyTensorDesc, &error);
        if (!requiredSymbolsLoaded) {
            return false;
        }

        createTuningConfig = reinterpret_cast<TuningConfigCreateFn>(
            dlsym(hclLibrary, "HIAI_HCL_TuningConfig_Create"));
        setTuningMode = reinterpret_cast<TuningConfigSetOptionFn>(
            dlsym(hclLibrary, "HIAI_HCL_TuningConfig_SetTuningMode"));
        setTuningObjective = reinterpret_cast<TuningConfigSetOptionFn>(
            dlsym(hclLibrary, "HIAI_HCL_TuningConfig_SetTuningObjective"));
        setTuningCacheDir = reinterpret_cast<TuningConfigSetCacheDirFn>(
            dlsym(hclLibrary, "HIAI_HCL_TuningConfig_SetCacheDir"));
        setTuningMemoryReusePlan = reinterpret_cast<TuningConfigSetOptionFn>(
            dlsym(hclLibrary,
                  "HIAI_HCL_TuningConfig_SetDeviceMemoryReusePlan"));
        destroyTuningConfig = reinterpret_cast<TuningConfigDestroyFn>(
            dlsym(hclLibrary, "HIAI_HCL_TuningConfig_Destroy"));
        setTuningConfig = reinterpret_cast<ModelBuildOptionsSetTuningConfigFn>(
            dlsym(hclLibrary,
                  "HIAI_HCL_ModelBuildOptions_SetTuningConfig"));
        setPriority = reinterpret_cast<ModelManagerSetPriorityFn>(
            dlsym(hclLibrary, "HIAI_HCL_ModelManager_SetPriority"));
        createNativeTensorBuffer = reinterpret_cast<TensorBufferCreateNativeFn>(
            dlsym(hclLibrary,
                  "HIAI_MR_NDTensorBuffer_CreateFromNativeHandle"));
        createNativeHandle = reinterpret_cast<NativeHandleCreateFn>(
            dlsym(tensorLibrary, "HIAI_NativeHandle_Create"));
        destroyNativeHandle = reinterpret_cast<NativeHandleDestroyFn>(
            dlsym(tensorLibrary, "HIAI_NativeHandle_Destroy"));

        // Persistent-cache symbols were added independently of the core V600
        // build/run ABI on some ROM branches. Keep them optional so an older
        // V600 runtime still builds and runs normally without a disk cache.
        saveBuiltModelToFile =
            reinterpret_cast<BuiltModelSaveToFileFn>(dlsym(hclLibrary, "HIAI_HCL_BuiltModel_SaveToFile"));
        restoreBuiltModelFromFile =
            reinterpret_cast<BuiltModelRestoreFromFileFn>(dlsym(hclLibrary, "HIAI_HCL_BuiltModel_RestoreFromFile"));
        checkBuiltModelCompatibility = reinterpret_cast<BuiltModelCheckCompatibilityFn>(
            dlsym(hclLibrary, "HIAI_HCL_BuiltModel_CheckCompatibility"));
        return true;
    }

    bool initialize() {
        if (!loadSymbols())
            return false;
        const char* version = getVersion();
        if (version == nullptr) {
            error = "HIAI_MR_GetVersion returned null";
            return false;
        }
        runtimeVersion = version;
        if (!VersionAtLeast(runtimeVersion, kMinimumHclVersion)) {
            error = "HiAI HCL version is older than 100.600";
            return false;
        }
        return true;
    }

    bool allocateIoBuffers() {
        const int32_t inputCount = getInputCount(builtModel);
        const int32_t outputCount = getOutputCount(builtModel);
        if (inputCount <= 0 || outputCount <= 0) {
            error = "built model returned invalid IO counts";
            return false;
        }
        inputDescs.reserve(static_cast<size_t>(inputCount));
        hostInputs.reserve(static_cast<size_t>(inputCount));
        for (int32_t i = 0; i < inputCount; ++i) {
            HclTensorDesc* desc = getInputDesc(builtModel, static_cast<size_t>(i));
            HclTensorBuffer* buffer = desc == nullptr ? nullptr : createTensorBuffer(desc);
            if (desc == nullptr || buffer == nullptr) {
                if (desc != nullptr)
                    destroyTensorDesc(&desc);
                error = "failed to allocate HCL input buffer " + std::to_string(i);
                return false;
            }
            inputDescs.push_back(desc);
            hostInputs.push_back(buffer);
        }
        outputDescs.reserve(static_cast<size_t>(outputCount));
        hostOutputs.reserve(static_cast<size_t>(outputCount));
        for (int32_t i = 0; i < outputCount; ++i) {
            HclTensorDesc* desc = getOutputDesc(builtModel, static_cast<size_t>(i));
            HclTensorBuffer* buffer = desc == nullptr ? nullptr : createTensorBuffer(desc);
            if (desc == nullptr || buffer == nullptr) {
                if (desc != nullptr)
                    destroyTensorDesc(&desc);
                error = "failed to allocate HCL output buffer " + std::to_string(i);
                return false;
            }
            outputDescs.push_back(desc);
            hostOutputs.push_back(buffer);
        }
        nativeInputs.resize(hostInputs.size(), nullptr);
        nativeOutputs.resize(hostOutputs.size(), nullptr);
        inputNativeHandles.resize(hostInputs.size(), nullptr);
        outputNativeHandles.resize(hostOutputs.size(), nullptr);
        return true;
    }

    bool initializeBuiltModel() {
        manager = createManager();
        if (manager == nullptr) {
            error = "HIAI_HCL_ModelManager_Create failed";
            return false;
        }
        HclModelInitOptions* initOptions = createInitOptions();
        if (initOptions == nullptr) {
            error = "HIAI_HCL_ModelInitOptions_Create failed";
            return false;
        }
        constexpr int32_t kPerfModeExtreme = 4;
        setPerfMode(initOptions, kPerfModeExtreme);
        const int32_t initStatus = initV2(manager, initOptions, builtModel, nullptr);
        destroyInitOptions(&initOptions);
        if (initStatus != kHiaiSuccess) {
            error = "HIAI_HCL_ModelManager_InitV2 failed: " + std::to_string(initStatus);
            return false;
        }
        managerInitialized = true;
        MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: performance=EXTREME band=UNSET\n");
        // HCL's direct model-manager implementation accepts the
        // non-preemptive HIGH/MIDDLE/LOW range (5..7).
        constexpr int32_t kPriorityHigh = 5;
        if (setPriority != nullptr) {
            const int32_t priorityStatus = setPriority(manager, kPriorityHigh);
            if (priorityStatus != kHiaiSuccess) {
                MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: high priority rejected: %d\n", priorityStatus);
            }
        }
        return allocateIoBuffers();
    }

    bool supportsPersistentCache() const {
        return saveBuiltModelToFile != nullptr && restoreBuiltModelFromFile != nullptr &&
               checkBuiltModelCompatibility != nullptr;
    }

    bool restoreCachedModel(const std::string& cacheFile) {
        if (cacheFile.empty()) {
            return false;
        }
        if (!supportsPersistentCache()) {
            MNN_PRINT(
                "MNN_HIAI_CACHE_AUDIT: DISABLED abi=V600 "
                "detail=cache symbols unavailable file=%s\n",
                cacheFile.c_str());
            return false;
        }
        if (access(cacheFile.c_str(), F_OK) != 0) {
            MNN_PRINT("MNN_HIAI_CACHE_AUDIT: MISS abi=V600 file=%s\n", cacheFile.c_str());
            return false;
        }

        builtModel = restoreBuiltModelFromFile(cacheFile.c_str());
        if (builtModel == nullptr) {
            (void)std::remove(cacheFile.c_str());
            MNN_ERROR(
                "MNN_HIAI_CACHE_AUDIT: INVALID abi=V600 file=%s "
                "detail=restore failed\n",
                cacheFile.c_str());
            return false;
        }

        int32_t compatibility = 1;
        const int32_t compatibilityStatus = checkBuiltModelCompatibility(builtModel, &compatibility);
        if (compatibilityStatus == kHiaiSuccess && compatibility == 0 && initializeBuiltModel()) {
            MNN_PRINT("MNN_HIAI_CACHE_AUDIT: HIT abi=V600 file=%s\n", cacheFile.c_str());
            return true;
        }

        const std::string cacheError = error;
        resetModel();
        error.clear();
        (void)std::remove(cacheFile.c_str());
        MNN_ERROR(
            "MNN_HIAI_CACHE_AUDIT: INVALID abi=V600 file=%s "
            "compat_status=%d compatibility=%d detail=%s\n",
            cacheFile.c_str(), compatibilityStatus, compatibility,
            cacheError.empty() ? "restore rejected" : cacheError.c_str());
        return false;
    }

    void saveCachedModel(const std::string& cacheFile) const {
        if (cacheFile.empty() || !supportsPersistentCache()) {
            return;
        }
        const std::string temporaryFile = UniqueCacheTemporaryFile(cacheFile);
        (void)std::remove(temporaryFile.c_str());
        const int32_t saveStatus = saveBuiltModelToFile(builtModel, temporaryFile.c_str());
        if (saveStatus == kHiaiSuccess && std::rename(temporaryFile.c_str(), cacheFile.c_str()) == 0) {
            MNN_PRINT("MNN_HIAI_CACHE_AUDIT: WRITE abi=V600 file=%s\n", cacheFile.c_str());
            return;
        }
        (void)std::remove(temporaryFile.c_str());
        MNN_ERROR(
            "MNN_HIAI_CACHE_AUDIT: WRITE_FAILED abi=V600 "
            "file=%s status=%d\n",
            cacheFile.c_str(), saveStatus);
    }
};

HiaiHclV600Runtime::HiaiHclV600Runtime(std::unique_ptr<Impl> impl) : mImpl(std::move(impl)) {}

HiaiHclV600Runtime::~HiaiHclV600Runtime() = default;

std::unique_ptr<HiaiHclV600Runtime> HiaiHclV600Runtime::CreateIfSupported(bool enableAutoTuning,
                                                                          const std::string& tuningCacheDirectory) {
    std::unique_ptr<Impl> impl(new Impl);
    impl->enableAutoTuning = enableAutoTuning;
    impl->tuningCacheDirectory = tuningCacheDirectory;
    if (!impl->initialize()) {
        return nullptr;
    }
    return std::unique_ptr<HiaiHclV600Runtime>(new HiaiHclV600Runtime(std::move(impl)));
}

bool HiaiHclV600Runtime::IsSupported(std::string* version) {
    auto runtime = CreateIfSupported();
    if (runtime == nullptr)
        return false;
    if (version != nullptr)
        *version = runtime->version();
    return true;
}

bool HiaiHclV600Runtime::buildAndLoad(const void* irData, size_t irSize, const std::string& modelName, bool preferFp16,
                                      const std::string& cacheFile) {
    mImpl->resetModel();
    mImpl->error.clear();
    if (irData == nullptr || irSize == 0 || modelName.empty()) {
        mImpl->error = "invalid serialized GE model";
        return false;
    }

    if (mImpl->restoreCachedModel(cacheFile)) {
        return true;
    }

    HclModelBuildOptions* options = mImpl->createBuildOptions();
    if (options == nullptr) {
        mImpl->error = "HIAI_HCL_ModelBuildOptions_Create failed";
        return false;
    }
    if (preferFp16 && mImpl->setPrecision(options, 1) != kHiaiSuccess) {
        mImpl->destroyBuildOptions(&options);
        mImpl->error = "failed to select FP16 HCL compilation";
        return false;
    }
    HclModelDeviceConfig* deviceConfig = mImpl->createDeviceConfig();
    if (deviceConfig == nullptr) {
        mImpl->destroyBuildOptions(&options);
        mImpl->error = "HIAI_HCL_ModelDeviceConfig_Create failed";
        return false;
    }
    constexpr int32_t kDeviceConfigAuto = 0;
    constexpr int32_t kFallbackDisabled = 1;
    // MODEL_LEVEL + an explicit NPU device order is rejected with
    // NOT_SUPPORT by some otherwise V600-capable Kirin runtimes. AUTO lets
    // the compiler select the supported device configuration.
    // Keep the HCL graph accelerator-only. If this compiler/runtime cannot
    // accept the complete graph, NPUBackend explicitly retries an older HiAI
    // model-build ABI instead of silently placing operators on CPUCL.
    mImpl->setDeviceConfigMode(deviceConfig, kDeviceConfigAuto);
    mImpl->setFallbackMode(deviceConfig, kFallbackDisabled);
    // ModelBuildOptions owns the device configuration after this call.
    mImpl->setModelDeviceConfig(options, deviceConfig);
    MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: device=AUTO fallback=OFF\n");
    const std::string tuningCacheDirectory = mImpl->enableAutoTuning ? mImpl->tuningCacheDirectory : std::string();
    if (!tuningCacheDirectory.empty()) {
        if (mImpl->createTuningConfig == nullptr ||
            mImpl->setTuningMode == nullptr ||
            mImpl->setTuningObjective == nullptr ||
            mImpl->setTuningCacheDir == nullptr ||
            mImpl->setTuningMemoryReusePlan == nullptr ||
            mImpl->destroyTuningConfig == nullptr ||
            mImpl->setTuningConfig == nullptr) {
            mImpl->destroyBuildOptions(&options);
            mImpl->error =
                "HCL automatic tuning requested but tuning symbols are unavailable";
            return false;
        }
        HclTuningConfig* tuningConfig = mImpl->createTuningConfig();
        constexpr int32_t kTuningModeAuto = 1;
        constexpr int32_t kTuningObjectivePerformance = 0;
        // Huawei documents LOW reuse as the faster plan: it spends more NPU
        // memory to avoid aggressive buffer aliasing during execution.
        constexpr int32_t kMemoryReuseLow = 1;
        if (tuningConfig == nullptr || mImpl->setTuningMode(tuningConfig, kTuningModeAuto) != kHiaiSuccess ||
            mImpl->setTuningObjective(tuningConfig, kTuningObjectivePerformance) != kHiaiSuccess ||
            mImpl->setTuningCacheDir(tuningConfig, tuningCacheDirectory.c_str()) != kHiaiSuccess ||
            mImpl->setTuningMemoryReusePlan(tuningConfig, kMemoryReuseLow) != kHiaiSuccess ||
            mImpl->setTuningConfig(options, tuningConfig) != kHiaiSuccess) {
            if (tuningConfig != nullptr) {
                mImpl->destroyTuningConfig(&tuningConfig);
            }
            mImpl->destroyBuildOptions(&options);
            mImpl->error = "failed to enable HCL automatic performance tuning";
            return false;
        }
        MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: tuning=AUTO objective=PERFORMANCE memory_reuse=LOW cache=%s\n",
                  tuningCacheDirectory.c_str());
    }
    const int32_t buildStatus = mImpl->buildV2(options, modelName.c_str(), irData, irSize, &mImpl->builtModel);
    mImpl->destroyBuildOptions(&options);
    if (buildStatus != kHiaiSuccess || mImpl->builtModel == nullptr) {
        mImpl->error = "HIAI_HCL_ModelBuilder_BuildV2 failed: " + std::to_string(buildStatus);
        mImpl->resetModel();
        return false;
    }

    if (!mImpl->initializeBuiltModel()) {
        mImpl->resetModel();
        return false;
    }
    mImpl->saveCachedModel(cacheFile);
    return true;
}

bool HiaiHclV600Runtime::resetModel() {
    return mImpl->resetModel();
}

bool HiaiHclV600Runtime::release() {
    return mImpl->release();
}

int HiaiHclV600Runtime::run() {
    if (!mImpl->managerInitialized)
        return -1;
    std::vector<HclTensorBuffer*> inputs = mImpl->hostInputs;
    std::vector<HclTensorBuffer*> outputs = mImpl->hostOutputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (mImpl->nativeInputs[i] != nullptr)
            inputs[i] = mImpl->nativeInputs[i];
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (mImpl->nativeOutputs[i] != nullptr)
            outputs[i] = mImpl->nativeOutputs[i];
    }
    const int status = mImpl->runV3(mImpl->manager, inputs.data(), static_cast<int32_t>(inputs.size()), outputs.data(),
                                    static_cast<int32_t>(outputs.size()));
    if (status == kHiaiSuccess && !mImpl->loggedRunV3Success) {
        mImpl->loggedRunV3Success = true;
        MNN_PRINT(
            "MNN_HIAI_HCL_V600_AUDIT: first RunV3 PASS "
            "inputs=%lu outputs=%lu\n",
            inputs.size(), outputs.size());
    }
    return status;
}

size_t HiaiHclV600Runtime::inputCount() const {
    return mImpl->hostInputs.size();
}

size_t HiaiHclV600Runtime::outputCount() const {
    return mImpl->hostOutputs.size();
}

size_t HiaiHclV600Runtime::inputSize(size_t index) const {
    return index < mImpl->hostInputs.size() ? mImpl->getTensorBufferSize(mImpl->hostInputs[index]) : 0;
}

size_t HiaiHclV600Runtime::outputSize(size_t index) const {
    return index < mImpl->hostOutputs.size() ? mImpl->getTensorBufferSize(mImpl->hostOutputs[index]) : 0;
}

void* HiaiHclV600Runtime::inputData(size_t index) const {
    return index < mImpl->hostInputs.size() ? mImpl->getTensorBufferData(mImpl->hostInputs[index]) : nullptr;
}

void* HiaiHclV600Runtime::outputData(size_t index) const {
    return index < mImpl->hostOutputs.size() ? mImpl->getTensorBufferData(mImpl->hostOutputs[index]) : nullptr;
}

bool HiaiHclV600Runtime::bindNativeHandleIo(size_t inputIndex, int inputFd, size_t inputBytes, size_t outputIndex,
                                            int outputFd, size_t outputBytes) {
    if (mImpl->createNativeTensorBuffer == nullptr ||
        mImpl->createNativeHandle == nullptr ||
        mImpl->destroyNativeHandle == nullptr ||
        inputIndex >= mImpl->inputDescs.size() || outputIndex >= mImpl->outputDescs.size() || inputFd < 0 ||
        outputFd < 0 || inputBytes == 0 || outputBytes == 0 || inputBytes > static_cast<size_t>(INT_MAX) ||
        outputBytes > static_cast<size_t>(INT_MAX)) {
        mImpl->error = "invalid HCL NativeHandle binding";
        return false;
    }
    mImpl->clearNativeHandleIo();
    auto* inputHandle = mImpl->createNativeHandle(inputFd, static_cast<int>(inputBytes), 0);
    auto* outputHandle = mImpl->createNativeHandle(outputFd, static_cast<int>(outputBytes), 0);
    if (inputHandle == nullptr || outputHandle == nullptr) {
        if (inputHandle != nullptr)
            mImpl->destroyNativeHandle(&inputHandle);
        if (outputHandle != nullptr)
            mImpl->destroyNativeHandle(&outputHandle);
        mImpl->error = "HIAI_NativeHandle_Create failed";
        return false;
    }
    HclTensorBuffer* inputBuffer = mImpl->createNativeTensorBuffer(mImpl->inputDescs[inputIndex], inputHandle);
    HclTensorBuffer* outputBuffer = mImpl->createNativeTensorBuffer(mImpl->outputDescs[outputIndex], outputHandle);
    if (inputBuffer == nullptr || outputBuffer == nullptr) {
        if (inputBuffer != nullptr)
            mImpl->destroyTensorBuffer(&inputBuffer);
        if (outputBuffer != nullptr)
            mImpl->destroyTensorBuffer(&outputBuffer);
        mImpl->destroyNativeHandle(&inputHandle);
        mImpl->destroyNativeHandle(&outputHandle);
        mImpl->error = "HIAI_MR_NDTensorBuffer_CreateFromNativeHandle failed";
        return false;
    }
    mImpl->inputNativeHandles[inputIndex] = inputHandle;
    mImpl->outputNativeHandles[outputIndex] = outputHandle;
    mImpl->nativeInputs[inputIndex] = inputBuffer;
    mImpl->nativeOutputs[outputIndex] = outputBuffer;
    return true;
}

void HiaiHclV600Runtime::clearNativeHandleIo() {
    mImpl->clearNativeHandleIo();
}

const std::string& HiaiHclV600Runtime::version() const {
    return mImpl->runtimeVersion;
}

const std::string& HiaiHclV600Runtime::lastError() const {
    return mImpl->error;
}

} // namespace MNN
