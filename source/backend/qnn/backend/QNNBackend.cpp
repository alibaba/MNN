//
//  QNNBackend.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBackend.hpp"
#include "QNNOpPackageUtils.hpp"
#include "core/MNNFileUtils.h"
#include "QnnTypeMacros.hpp"
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/FileLoader.hpp"
#include "QnnBackendConfig.hpp"
#include "QnnTensorConvert.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <utility>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
// #define QNN_PROFILE_OP
// #define QNN_PROFILE_SUMMARIZE
// #define QNN_VERBOSE
namespace MNN {
static const std::string& extraIoPrefix() {
    static const std::string prefix = "_mnn";
    return prefix;
}
namespace QNN {
static std::string environmentValue(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR <= 27
struct QnnV66LayerNormOpPackage {
    std::string libraryName;
    std::string interfaceProvider;
    std::string packageName;
    std::string resourceDirectory;
};

static bool qnnV66LayerNormLibraryName(const std::string& manifestPath,
                                       const std::string& configuredInterfaceProvider,
                                       const std::string& configuredPackageName, std::string* libraryName) {
    if (libraryName == nullptr) {
        return false;
    }
    if (manifestPath.empty()) {
        if (configuredInterfaceProvider != "TbliveV66LayerNormPackageInterfaceProvider" ||
            configuredPackageName != "TbliveV66LayerNormPackage") {
            return false;
        }
        *libraryName = "libTbliveV66LayerNorm.so";
        return true;
    }
    std::ifstream manifest(manifestPath);
    if (!manifest) {
        return false;
    }
    std::string line;
    while (std::getline(manifest, line)) {
        constexpr const char* kKey = "v66_layernorm_op_package=";
        if (line.rfind(kKey, 0) != 0) {
            continue;
        }
        *libraryName = line.substr(std::strlen(kKey));
        if (!libraryName->empty() && libraryName->back() == '\r') {
            libraryName->pop_back();
        }
        return !libraryName->empty();
    }
    return false;
}

static bool qnnV66LayerNormOpPackageRequested(const std::string& manifestPath,
                                              const std::string& configuredInterfaceProvider,
                                              const std::string& configuredPackageName) {
    if (!configuredInterfaceProvider.empty() || !configuredPackageName.empty()) {
        return true;
    }
    std::string libraryName;
    return qnnV66LayerNormLibraryName(manifestPath, configuredInterfaceProvider, configuredPackageName,
                                      &libraryName);
}

static bool qnnV66LayerNormOpPackage(const std::string& manifestPath, const std::string& resourceDirectory,
                                     const std::string& configuredInterfaceProvider,
                                     const std::string& configuredPackageName, QnnV66LayerNormOpPackage* opPackage) {
    if (opPackage == nullptr)
        return false;
    if (resourceDirectory.empty()) {
        return false;
    }
    std::string libraryName;
    if (!qnnV66LayerNormLibraryName(manifestPath, configuredInterfaceProvider, configuredPackageName,
                                    &libraryName)) {
        return false;
    }
    if (libraryName == "." || libraryName == ".." || libraryName.find('/') != std::string::npos ||
        libraryName.find('\\') != std::string::npos) {
        return false;
    }
    std::string derivedInterfaceProvider;
    std::string derivedPackageName;
    // The basename is derived from the fixed V66 contract when no legacy
    // runtime manifest is supplied. Always validate the exact
    // lib<X>V66LayerNorm.so form before accepting explicit registration names.
    if (!deriveV66LayerNormOpPackageNames(libraryName, &derivedInterfaceProvider, &derivedPackageName)) {
        return false;
    }
    std::string directory(resourceDirectory);
    if (!directory.empty() && directory.back() != '/')
        directory.push_back('/');
    const std::string hostPath = directory + libraryName;
    std::ifstream package(hostPath, std::ios::binary);
    if (!package.good())
        return false;
    // QNN DSP forwards this string to the remote FastRPC loader. The loader
    // searches ADSP_LIBRARY_PATH but cannot resolve the app-private Android
    // absolute path directly, so register the verified basename.
    opPackage->libraryName = libraryName;
    opPackage->interfaceProvider =
        configuredInterfaceProvider.empty() ? derivedInterfaceProvider : configuredInterfaceProvider;
    opPackage->packageName = configuredPackageName.empty() ? derivedPackageName : configuredPackageName;
    opPackage->resourceDirectory = resourceDirectory;
    return true;
}
#endif

static const char* qnnDataTypeName(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_FLOAT_16: return "FLOAT_16";
        case QNN_DATATYPE_FLOAT_32: return "FLOAT_32";
        case QNN_DATATYPE_FLOAT_64: return "FLOAT_64";
        case QNN_DATATYPE_SFIXED_POINT_4: return "SFIXED_POINT_4";
        case QNN_DATATYPE_SFIXED_POINT_8: return "SFIXED_POINT_8";
        case QNN_DATATYPE_SFIXED_POINT_16: return "SFIXED_POINT_16";
        case QNN_DATATYPE_SFIXED_POINT_32: return "SFIXED_POINT_32";
        case QNN_DATATYPE_UFIXED_POINT_4: return "UFIXED_POINT_4";
        case QNN_DATATYPE_UFIXED_POINT_8: return "UFIXED_POINT_8";
        case QNN_DATATYPE_UFIXED_POINT_16: return "UFIXED_POINT_16";
        case QNN_DATATYPE_UFIXED_POINT_32: return "UFIXED_POINT_32";
        case QNN_DATATYPE_INT_8: return "INT_8";
        case QNN_DATATYPE_INT_16: return "INT_16";
        case QNN_DATATYPE_INT_32: return "INT_32";
        case QNN_DATATYPE_INT_64: return "INT_64";
        case QNN_DATATYPE_UINT_8: return "UINT_8";
        case QNN_DATATYPE_UINT_16: return "UINT_16";
        case QNN_DATATYPE_UINT_32: return "UINT_32";
        case QNN_DATATYPE_UINT_64: return "UINT_64";
        case QNN_DATATYPE_BOOL_8: return "BOOL_8";
        default: return "OTHER";
    }
}

struct QnnContext {
    QNN_INTERFACE_VER_TYPE QnnInterface{};
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    int soc_id = 0;
    int dsp_arch = 0;
};

static QnnContext gContext;
static std::mutex gQnnContextMutex;
static std::atomic<uint32_t> gIndependentContextCount{0};

static void releaseQnnContext(QnnContext& context) {
    if (context.deviceHandle != nullptr &&
        context.QnnInterface.deviceFree != nullptr) {
        context.QnnInterface.deviceFree(context.deviceHandle);
    }
    if (context.backendHandle != nullptr &&
        context.QnnInterface.backendFree != nullptr) {
        context.QnnInterface.backendFree(context.backendHandle);
    }
    if (context.logHandle != nullptr &&
        context.QnnInterface.logFree != nullptr) {
        context.QnnInterface.logFree(context.logHandle);
    }
    context = {};
}

struct IndependentQnnContextOwner {
    ~IndependentQnnContextOwner() {
        if (!active) {
            return;
        }
        std::lock_guard<std::mutex> lock(gQnnContextMutex);
        releaseQnnContext(context);
        gIndependentContextCount.fetch_sub(1, std::memory_order_release);
    }

    QnnContext context;
    bool active = false;
};

static bool queryQnnSystemInterface(
    QNN_SYSTEM_INTERFACE_VER_TYPE* systemInterface) {
    if (systemInterface == nullptr) {
        return false;
    }
    *systemInterface = {};
#ifndef ENABLE_QNN_CONVERT_MODE
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
    if (QNN::QnnSystemInterface_getProviders == nullptr) {
        return true;
    }
    QnnSystemInterface_t** interfaceProviders = nullptr;
    uint32_t numProviders = 0;
    if (QNN::QnnSystemInterface_getProviders(
            (const QnnSystemInterface_t***)&interfaceProviders,
            &numProviders) != QNN_SUCCESS ||
        interfaceProviders == nullptr || numProviders == 0) {
        MNN_PRINT("MNN_QNN: Failed to get QNN System interface providers.\n");
        return false;
    }
    for (size_t index = 0; index < numProviders; ++index) {
        if (QNN_SYSTEM_API_VERSION_MAJOR ==
                interfaceProviders[index]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <=
                interfaceProviders[index]->systemApiVersion.minor) {
            *systemInterface =
                interfaceProviders[index]->QNN_SYSTEM_INTERFACE_VER_NAME;
            return true;
        }
    }
    MNN_PRINT("MNN_QNN: Failed to find a compatible QNN System interface.\n");
    return false;
#else
    return true;
#endif
#else
    *systemInterface = QNN::gQnnConvertorSystemInterface;
    return true;
#endif
}

static bool createQnnContextForLoadedBackend(const std::string& runtimeManifestPath,
                                             const std::string& acceleratorResourceDirectory,
                                             const std::string& opPackageInterfaceProvider,
                                             const std::string& opPackageName,
                                             bool verboseLogging,
                                             QnnContext* createdContext,
                                             std::string* resolvedOpPackageName) {
    if (createdContext == nullptr) {
        return false;
    }
    *createdContext = {};
    if (resolvedOpPackageName != nullptr) {
        resolvedOpPackageName->clear();
    }
    QnnContext context{};
    QNN_INTERFACE_VER_TYPE qnnInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    {
        QnnInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return false;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return false;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return false;
        }
        bool foundValidInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return false;
        }
    }
#else
    qnnInterface = QNN::gQnnConvertorInterface;
#endif
    context.QnnInterface = qnnInterface;

    // Create Log.
    {
        QnnLog_Callback_t logCallback = nullptr;
        QnnLog_Level_t logLevel = QNN_LOG_LEVEL_ERROR;
        if (verboseLogging) {
            logLevel = QNN_LOG_LEVEL_VERBOSE;
            logCallback = [](const char* format, QnnLog_Level_t level,
                             uint64_t timestamp, va_list args) {
            (void)timestamp;
#ifdef __ANDROID__
            int androidLevel = ANDROID_LOG_VERBOSE;
            if (level == QNN_LOG_LEVEL_ERROR) {
                androidLevel = ANDROID_LOG_ERROR;
            } else if (level == QNN_LOG_LEVEL_WARN) {
                androidLevel = ANDROID_LOG_WARN;
            } else if (level == QNN_LOG_LEVEL_INFO) {
                androidLevel = ANDROID_LOG_INFO;
            }
            __android_log_vprint(androidLevel, "MNN_QNN_LIB", format, args);
#else
            (void)level;
            vprintf(format, args);
#endif
            };
        }
#ifdef QNN_DEBUG
        if (!verboseLogging) {
            logCallback = [](const char* format, QnnLog_Level_t level,
                             uint64_t timestamp, va_list args) {
                (void)timestamp;
                if (level <= QNN_LOG_LEVEL_ERROR) {
                    char buffer[512];
                    vsnprintf(buffer, sizeof(buffer), format, args);
                    MNN_PRINT("QNN_LOG[%d]: %s\n", level, buffer);
                }
            };
        }
#endif
        if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(
                 logCallback, logLevel, &context.logHandle)) !=
             QNN_SUCCESS) ||
            (context.logHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to initialize logging in the backend.\n");
            releaseQnnContext(context);
            return false;
        }
    }

    // Create Backend.
    {
        const QnnBackend_Config_t** backendConfig = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(
                 context.logHandle, backendConfig, &context.backendHandle)) !=
             QNN_SUCCESS) ||
            (context.backendHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to create the %s backend.\n",
                      QNN::getLoadedQNNBackendName());
            releaseQnnContext(context);
            return false;
        }
    }

    // Create Device.
    int dspArch = QNN::getLoadedQNNBackend() ==
                          QNN::QnnBackendKind::Dsp
                      ? 66
                      : 0;
    uint32_t socId = 0;
    {
        const bool isHtpBackend =
            QNN::getLoadedQNNBackend() == QNN::QnnBackendKind::Htp;
        // Check whether the device API is supported.
        bool supportDevice = QNN::checkCapability(qnnInterface, QNN_PROPERTY_GROUP_DEVICE);
        if (supportDevice) {
            const QnnDevice_Config_t ** deviceConfig = nullptr;
            auto qnnStatus = qnnInterface.deviceCreate(
                context.logHandle, deviceConfig, &context.deviceHandle);
            if(qnnStatus != QNN_SUCCESS || (context.deviceHandle == nullptr)) {
                if (isHtpBackend) {
                    MNN_PRINT(
                        "MNN_QNN: Failed to create a device for the %s backend, "
                        "error:%lu\n",
                        QNN::getLoadedQNNBackendName(),
                        (unsigned long)qnnStatus);
                    releaseQnnContext(context);
                    return false;
                }
                MNN_PRINT(
                    "MNN_QNN: DSP backend has no device handle; continuing "
                    "with the backend and a null device handle.\n");
            }

            if (!isHtpBackend) {
                // QNN DSP does not expose the HTP device-info extension.
            } else if (qnnInterface.deviceGetPlatformInfo == nullptr) {
                MNN_PRINT("[Warning]: No QnnDevice_getPlatformInfo API");
            } else {
                const QnnDevice_PlatformInfo_t* backendPlatformInfoPtr = nullptr;
                qnnStatus = qnnInterface.deviceGetPlatformInfo(
                    context.logHandle, &backendPlatformInfoPtr);
                if(qnnStatus != QNN_SUCCESS || backendPlatformInfoPtr == nullptr) {
                    MNN_PRINT("[Warning]: deviceGetPlatformInfo Failed to query platform info");
                } else {
                    QnnDevice_HardwareDeviceInfo_t* hwDeviceInfo = backendPlatformInfoPtr->v1.hwDevices;
                    if (hwDeviceInfo != nullptr &&
                        hwDeviceInfo->v1.deviceInfoExtension != nullptr) {
                        dspArch = hwDeviceInfo->v1.deviceInfoExtension
                                      ->onChipDevice.arch;
                        socId = hwDeviceInfo->v1.deviceInfoExtension
                                    ->onChipDevice.socModel;
                    }
                    if (qnnInterface.deviceFreePlatformInfo != nullptr) {
                        qnnInterface.deviceFreePlatformInfo(
                            context.logHandle, backendPlatformInfoPtr);
                    }
                }
            }
        } else if (isHtpBackend) {
            MNN_PRINT("MNN_QNN: HTP backend does not support the required device API.\n");
            releaseQnnContext(context);
            return false;
        } else {
            MNN_PRINT(
                "MNN_QNN: DSP backend does not expose the optional device "
                "API; continuing with a null device handle.\n");
        }
    }

#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR <= 27
    const bool opPackageRequested = qnnV66LayerNormOpPackageRequested(
        runtimeManifestPath, opPackageInterfaceProvider, opPackageName);
    if (QNN::getLoadedQNNBackend() == QNN::QnnBackendKind::Dsp && opPackageRequested) {
        QnnV66LayerNormOpPackage opPackage;
        if (!qnnV66LayerNormOpPackage(runtimeManifestPath, acceleratorResourceDirectory, opPackageInterfaceProvider,
                                      opPackageName, &opPackage)) {
            MNN_PRINT(
                "MNN_QNN_V66_OP_PACKAGE_AUDIT: registered=0 "
                "reason=MISSING_OR_INVALID_RESOURCE\n");
            releaseQnnContext(context);
            return false;
        }
        if (qnnInterface.backendRegisterOpPackage == nullptr) {
            MNN_PRINT(
                "MNN_QNN_V66_OP_PACKAGE_AUDIT: registered=0 "
                "reason=UNSUPPORTED_INTERFACE\n");
            releaseQnnContext(context);
            return false;
        }
        const Qnn_ErrorHandle_t status = qnnInterface.backendRegisterOpPackage(
            context.backendHandle, opPackage.libraryName.c_str(), opPackage.interfaceProvider.c_str(), nullptr);
        if (QNN_GET_ERROR_CODE(status) != QNN_SUCCESS) {
            MNN_PRINT(
                "MNN_QNN_V66_OP_PACKAGE_AUDIT: registered=0 "
                "reason=REGISTER_FAILED error=%lu package=%s\n",
                (unsigned long)status, opPackage.libraryName.c_str());
            releaseQnnContext(context);
            return false;
        }
        if (resolvedOpPackageName != nullptr) {
            *resolvedOpPackageName = opPackage.packageName;
        }
        MNN_PRINT(
            "MNN_QNN_V66_OP_PACKAGE_AUDIT: registered=1 package=%s "
            "provider=%s "
            "target=default\n",
            opPackage.libraryName.c_str(), opPackage.interfaceProvider.c_str());
    }
#endif

    if (!queryQnnSystemInterface(&context.systemInterface)) {
        releaseQnnContext(context);
        return false;
    }
    context.soc_id = socId;
    context.dsp_arch = dspArch;
    *createdContext = context;
    MNN_PRINT("MNN_QNN: Using %s backend, DSP architecture v%d.\n",
              QNN::getLoadedQNNBackendName(), dspArch);
    return true;
}

static bool createQnnContext(const std::string& runtimeManifestPath = {},
                             const std::string& acceleratorResourceDirectory = {},
                             const std::string& opPackageInterfaceProvider = {}, const std::string& opPackageName = {},
                             bool allowFallback = true, std::string* resolvedOpPackageName = nullptr) {
    std::lock_guard<std::mutex> lck(gQnnContextMutex);
    if (resolvedOpPackageName != nullptr) {
        resolvedOpPackageName->clear();
    }
    if (gContext.backendHandle != nullptr) {
        return true;
    }
    QnnContext context;
    if (createQnnContextForLoadedBackend(runtimeManifestPath, acceleratorResourceDirectory, opPackageInterfaceProvider,
                                         opPackageName, false, &context, resolvedOpPackageName)) {
        gContext = context;
        return true;
    }
    if (!allowFallback) {
        MNN_PRINT(
            "MNN_QNN: Forced %s backend initialization failed; automatic "
            "fallback is disabled.\n",
            QNN::getLoadedQNNBackendName());
        return false;
    }

    const auto failedBackend = QNN::getLoadedQNNBackend();
    const auto fallbackBackend =
        failedBackend == QNN::QnnBackendKind::Dsp
            ? QNN::QnnBackendKind::Htp
            : QNN::QnnBackendKind::Dsp;
    MNN_PRINT("MNN_QNN: %s initialization failed; trying the alternate backend.\n",
              QNN::getLoadedQNNBackendName());
    if (!QNN::loadQNNSymbol(fallbackBackend)) {
        MNN_PRINT("MNN_QNN: No usable HTP or DSP backend was found.\n");
        return false;
    }
    context = {};
    if (!createQnnContextForLoadedBackend(runtimeManifestPath, acceleratorResourceDirectory, opPackageInterfaceProvider,
                                          opPackageName, false, &context, resolvedOpPackageName)) {
        MNN_PRINT("MNN_QNN: No usable HTP or DSP backend was found.\n");
        return false;
    }
    gContext = context;
    return true;
}

static std::shared_ptr<IndependentQnnContextOwner> createIndependentQnnContext(
    const std::string& runtimeManifestPath, const std::string& acceleratorResourceDirectory,
    const std::string& opPackageInterfaceProvider, const std::string& opPackageName,
    std::string* resolvedOpPackageName) {
    auto owner = std::make_shared<IndependentQnnContextOwner>();
    {
        std::lock_guard<std::mutex> lock(gQnnContextMutex);
        if (!createQnnContextForLoadedBackend(runtimeManifestPath, acceleratorResourceDirectory,
                                              opPackageInterfaceProvider, opPackageName, true, &owner->context,
                                              resolvedOpPackageName)) {
            return nullptr;
        }
        owner->active = true;
        gIndependentContextCount.fetch_add(1, std::memory_order_release);
    }
    return owner;
}

static bool ensureQnnSystemInterface() {
    std::lock_guard<std::mutex> lck(gQnnContextMutex);
    return queryQnnSystemInterface(&gContext.systemInterface);
}

static bool prepareQnnPluginRuntime(bool dedicatedQnnSession) {
#ifndef ENABLE_QNN_CONVERT_MODE
    if (dedicatedQnnSession) {
        if (gContext.backendHandle == nullptr) {
            if (!QNN::loadQNNSymbol() || !QNN::loadQNNSystemSymbol({}) ||
                !createQnnContext()) {
                return false;
            }
        } else if (!QNN::loadQNNSystemSymbol({})) {
            return false;
        }
        return ensureQnnSystemInterface() &&
               gContext.systemInterface.systemContextCreate != nullptr;
    }
    const auto loadedBackend = QNN::getLoadedQNNBackend();
    if (loadedBackend != QNN::QnnBackendKind::None &&
        loadedBackend != QNN::QnnBackendKind::Htp) {
        return false;
    }
    if (gContext.backendHandle == nullptr) {
        if (!QNN::loadQNNSymbol(QNN::QnnBackendKind::Htp) ||
            !QNN::loadQNNSystemSymbol({}) ||
            !createQnnContext({}, {}, {}, {}, false)) {
            return false;
        }
    } else if (!QNN::loadQNNSystemSymbol({})) {
        return false;
    }
    return ensureQnnSystemInterface() &&
           gContext.systemInterface.systemContextCreate != nullptr;
#else
    return createQnnContext() && ensureQnnSystemInterface();
#endif
}

#ifdef QNN_PROFILE_SUMMARIZE
static std::string getOpTypeFromName(const std::string& nodeName) {
    // The pattern is usually "OpType_..."
    size_t pos = nodeName.find('_');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // Fallback for names without '_', like "Input OpId_2 (cycles)"
    pos = nodeName.find(' ');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // If no delimiter is found, return the whole name as the type
    return nodeName;
}
#endif

static void createProfileHandle(const QNN_INTERFACE_VER_TYPE& QnnInterface, const Qnn_BackendHandle_t& backend_handle,
                                Qnn_ProfileHandle_t* profile_handle_ptr) {
#if defined(QNN_PROFILE_SUMMARIZE) || defined(QNN_PROFILE_OP)
    if (*profile_handle_ptr == nullptr) {
        // set QNN_PROFILE_LEVEL_DETAILED
        QnnProfile_Level_t profileLevel = QNN_PROFILE_LEVEL_DETAILED;
        MNN_PRINT("[QNN Profile] Creating QNN Profile Handle with DETAILED level.\n");
        auto profile_err = QnnInterface.profileCreate(backend_handle, profileLevel, profile_handle_ptr);
        if (profile_err != QNN_SUCCESS || *profile_handle_ptr == nullptr) {
            MNN_ERROR("[QNN Profile] Failed to create QNN Profile Handle, error: %d\n", (int)profile_err);
            *profile_handle_ptr = nullptr;
        }
    }
#endif
}

static void doProfile(const QNN_INTERFACE_VER_TYPE& QnnInterface, const Qnn_ProfileHandle_t& profile_handle) {
#ifdef QNN_PROFILE_OP
    if (profile_handle) {
        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = QnnInterface.profileGetEvents(profile_handle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        MNN_PRINT("\n--- QNN Node-level Performance Report ---\n");
        bool foundNodeData = false;

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            QnnInterface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) {
                MNN_PRINT("Found EXECUTE event. Total time: %llu us. Querying sub-events...\n", (unsigned long long)eventData.value);

                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;

                // 3. GetSubEvents
                auto get_sub_err = QnnInterface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) {
                    MNN_PRINT("[QNN Profile] Failed to get sub-events for EXECUTE event. Error: %d\n", (int)get_sub_err);
                    continue;
                }

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    QnnInterface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        foundNodeData = true;
                        const char* nodeName = subEventData.identifier;
                        uint64_t value = subEventData.value;

                        switch (subEventData.unit) {
                            case QNN_PROFILE_EVENTUNIT_MICROSEC:
                                MNN_PRINT("Node: %-45s | Time: %10llu us (%.3f ms)\n",
                                        nodeName, (unsigned long long)value, (double)value / 1000.0);
                                break;
                            case QNN_PROFILE_EVENTUNIT_CYCLES:
                                MNN_PRINT("Node: %-45s | Cycles: %.2f*10^6\n", nodeName, (double)value / 1000000.0);
                                break;
                            // ... other dealing ...
                            default:
                                MNN_PRINT("Node: %-45s | Value: %10llu (Unit: %u - Unknown)\n",
                                        nodeName, (unsigned long long)value, subEventData.unit);
                                break;
                        }
                    }
                }
            }
        }

        if (!foundNodeData) {
            MNN_PRINT("No node-specific performance data found. Please ensure you have set:\n");
            MNN_PRINT("1. Profile level to QNN_PROFILE_LEVEL_DETAILED.\n");
            MNN_PRINT("2. HTP graph config with QNN_HTP_GRAPH_CONFIG_OPTION_PERF_PROFILE (if available).\n");
        }
        MNN_PRINT("-----------------------------------------\n");
    }
#endif

#ifdef QNN_PROFILE_SUMMARIZE
    if (profile_handle) {
        std::map<std::string, uint64_t> opCycleStats;
        uint64_t totalNodeCycles = 0;

        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = QnnInterface.profileGetEvents(profile_handle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            QnnInterface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) { // == QNN_PROFILE_EVENTTYPE_EXECUTE) {
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;
                auto get_sub_err = QnnInterface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) continue;

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    QnnInterface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        if (subEventData.identifier) {
                            std::string opType = getOpTypeFromName(subEventData.identifier);
                            opCycleStats[opType] += subEventData.value;
                            totalNodeCycles += subEventData.value;
                        }
                    }
                }
            }
        }

        if (!opCycleStats.empty()) {
            MNN_PRINT("\n--- QNN Operator-wise Performance Summary ---\n");
            MNN_PRINT("%-20s | %15s | %s\n", "Operator Type", "Total Cycles", "Percentage");
            MNN_PRINT("--------------------------------------------------\n");

            std::vector<std::pair<std::string, uint64_t>> sortedStats(opCycleStats.begin(), opCycleStats.end());
            std::sort(sortedStats.begin(), sortedStats.end(), [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
                return a.second > b.second; // sort by large -> small
            });

            for (const auto& pair : sortedStats) {
                double percentage = (totalNodeCycles > 0) ? ((double)pair.second * 100.0 / totalNodeCycles) : 0.0;
                MNN_PRINT("%-20s | %15llu | %.2f%%\n", pair.first.c_str(), pair.second, percentage);
            }
            MNN_PRINT("--------------------------------------------------\n");
            MNN_PRINT("%-20s | %15llu | 100.00%%\n", "Total", totalNodeCycles);
        }
    }
    // =========================================================
#endif
}

// Helper: get byte size per element for a QNN data type
static uint32_t getQnnDataTypeSize(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_BOOL_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
            return 1;
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
            return 2;
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            return 4;
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_FLOAT_64:
            return 8;
        default:
            return 0;
    }
}

// Helper: calculate total data size in bytes for a QNN tensor
static size_t calcQnnTensorDataSize(const Qnn_Tensor_t& tensor) {
    uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
    if (rank == 0 || dims == nullptr) {
        return 0;
    }
    size_t elementCount = 1;
    for (uint32_t i = 0; i < rank; i++) {
        elementCount *= dims[i];
    }
    uint32_t elementSize = getQnnDataTypeSize(QNN_TENSOR_GET_DATA_TYPE(tensor));
    return elementCount * elementSize;
}

// Helper: ensure all output tensors have memory allocated for graphExecute.
// Returns a vector of (index, pointer) pairs for temporarily allocated buffers that must be freed after execution.
static std::vector<std::pair<int, void*>> ensureOutputTensorsMemory(Qnn_Tensor_t* outputTensors,
                                                                    uint32_t numOutputTensors) {
    std::vector<std::pair<int, void*>> tempBuffers;
    for (uint32_t i = 0; i < numOutputTensors; i++) {
        auto& tensor = outputTensors[i];
        if (QNN_TENSOR_GET_MEM_TYPE(tensor) == QNN_TENSORMEMTYPE_RAW) {
            auto clientBuf = QNN_TENSOR_GET_CLIENT_BUF(tensor);
            if (clientBuf.data == nullptr) {
                size_t dataSize = calcQnnTensorDataSize(tensor);
                if (dataSize > 0) {
                    void* tempData = malloc(dataSize);
                    if (tempData != nullptr) {
                        Qnn_ClientBuffer_t buf = {tempData, (uint32_t)dataSize};
                        QNN_TENSOR_SET_CLIENT_BUF(tensor, buf);
                        tempBuffers.push_back({(int)i, tempData});
                    }
                }
            }
        }
    }
    return tempBuffers;
}

// Helper: free temporarily allocated output tensor buffers and reset their clientBuf.
static void freeOutputTensorsTempMemory(Qnn_Tensor_t* outputTensors, std::vector<std::pair<int, void*>>& tempBuffers) {
    for (auto& p : tempBuffers) {
        Qnn_ClientBuffer_t emptyBuf = {nullptr, 0};
        QNN_TENSOR_SET_CLIENT_BUF(outputTensors[p.first], emptyBuf);
        free(p.second);
    }
}

class QNNTensorDumper {
public:
    explicit QNNTensorDumper(bool enabled, const std::string& outputDirectory = "") : mEnabled(enabled) {
        if (!mEnabled) {
            return;
        }
        const char* configuredDirectory = std::getenv("MNN_QNN_DUMP_DIR");
        if (configuredDirectory != nullptr && configuredDirectory[0] != '\0') {
            mOutputDirectory = configuredDirectory;
            return;
        }
        if (!outputDirectory.empty()) {
            mOutputDirectory = outputDirectory;
            return;
        }
        mOutputDirectory = "qnn_intermediate_outputs";
    }

    void dump(const Qnn_Tensor_t* tensors, uint32_t tensorCount) {
        if (!mEnabled || tensors == nullptr || tensorCount == 0) {
            return;
        }
        if (!mDirectoryReady) {
            if (!MNNCreateDir(mOutputDirectory.c_str())) {
                MNN_ERROR("MNN_QNN: Failed to create intermediate dump directory: %s\n",
                          mOutputDirectory.c_str());
                mEnabled = false;
                return;
            }
            mDirectoryReady = true;
        }
        char manifestName[64];
        std::snprintf(manifestName, sizeof(manifestName), "manifest_%06llu.tsv",
                      static_cast<unsigned long long>(mExecution));
        const auto manifestPath = MNNFilePathConcat(mOutputDirectory, manifestName);
        FILE* manifest = nullptr;

        for (uint32_t index = 0; index < tensorCount; ++index) {
            const auto& tensor = tensors[index];
            if (QNN_TENSOR_GET_MEM_TYPE(tensor) != QNN_TENSORMEMTYPE_RAW) {
                continue;
            }
            const auto buffer = QNN_TENSOR_GET_CLIENT_BUF(tensor);
            const size_t bytes = buffer.dataSize > 0 ? buffer.dataSize : calcQnnTensorDataSize(tensor);
            if (buffer.data == nullptr || bytes == 0) {
                continue;
            }
            std::string name = QNN_TENSOR_GET_NAME(tensor) == nullptr
                                   ? "unnamed"
                                   : QNN_TENSOR_GET_NAME(tensor);
            for (char& character : name) {
                const auto value = static_cast<unsigned char>(character);
                if (!std::isalnum(value) && character != '-' &&
                    character != '_' && character != '.') {
                    character = '_';
                }
            }
            char fileName[1024];
            std::snprintf(fileName, sizeof(fileName),
                          "execution_%06llu_tensor_%04u_%s.raw",
                          static_cast<unsigned long long>(mExecution), index,
                          name.c_str());
            const auto path =
                MNNFilePathConcat(mOutputDirectory, fileName);
            FILE* output = std::fopen(path.c_str(), "wb");
            if (output == nullptr) {
                MNN_ERROR("MNN_QNN: Failed to open intermediate tensor dump: %s\n",
                          path.c_str());
                continue;
            }
            const size_t written = std::fwrite(buffer.data, 1, bytes, output);
            std::fclose(output);
            if (written != bytes) {
                MNN_ERROR("MNN_QNN: Incomplete intermediate tensor dump: %s\n",
                          path.c_str());
                continue;
            }

            if (manifest == nullptr) {
                manifest = std::fopen(manifestPath.c_str(), "w");
                if (manifest == nullptr) {
                    MNN_ERROR("MNN_QNN: Failed to open intermediate dump manifest: %s\n",
                              manifestPath.c_str());
                } else {
                    std::fprintf(manifest,
                                 "index\tname\tfile\tdata_type\tdimensions\tquant_encoding\tscale\toffset\n");
                }
            }
            if (manifest != nullptr) {
                const auto quant = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
                float scale = 0.0f;
                int32_t offset = 0;
                if (quant.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    scale = quant.scaleOffsetEncoding.scale;
                    offset = quant.scaleOffsetEncoding.offset;
                }
                const char* nativeName = QNN_TENSOR_GET_NAME(tensor);
                std::fprintf(manifest, "%u\t%s\t%s\t%u\t", index, nativeName == nullptr ? "" : nativeName,
                             fileName, static_cast<unsigned int>(QNN_TENSOR_GET_DATA_TYPE(tensor)));
                const auto rank = QNN_TENSOR_GET_RANK(tensor);
                const auto dimensions = QNN_TENSOR_GET_DIMENSIONS(tensor);
                for (uint32_t dimension = 0; dimension < rank; ++dimension) {
                    std::fprintf(manifest, "%s%u", dimension == 0 ? "" : "x", dimensions[dimension]);
                }
                std::fprintf(manifest, "\t%u\t%.9g\t%d\n",
                             static_cast<unsigned int>(quant.quantizationEncoding), scale, offset);
            }
        }
        if (manifest != nullptr) {
            std::fclose(manifest);
        }
        ++mExecution;
    }

private:
    bool mEnabled = false;
    bool mDirectoryReady = false;
    uint64_t mExecution = 0;
    std::string mOutputDirectory;
};
}
}

#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)

#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "shape/SizeComputer.hpp"
#include "flatbuffers/flexbuffers.h"
#include "core/OpCommonUtils.hpp"
#include "dsprpc_interface.h"

// 在 MSVC / Windows 平台上没有 RPC (ION) 内存机制，定义 MNN_QNN_NO_RPC_MEM 进入退化分支：
//   - 直接使用普通堆内存 (malloc / free) 替代 rpcmem_alloc / rpcmem_free
//   - setToTensor 走 QNN_TENSORMEMTYPE_RAW + clientBuf 路径，避免调用 memRegister(ION fd)
// 该退化主要用于让代码在 MSVC 上能够编译通过；真正运行 QNN 推理仍需在 Android/Linux DSP 平台。
#if defined(_MSC_VER) || defined(_WIN32)
#define MNN_QNN_NO_RPC_MEM 1
#endif

// MSVC 不支持 __fp16 扩展类型，需要用 uint16_t + 软件转换替代。
// 其它平台（GCC/Clang on ARM 等）保持原生 __fp16 行为。
#if defined(_MSC_VER) || defined(_WIN32)
#include "half.hpp"
namespace MNN {
namespace plugin {
typedef uint16_t mnn_qnn_fp16_t;
static inline mnn_qnn_fp16_t mnn_qnn_float_to_fp16(float v) {
    half_float::half h(v);
    mnn_qnn_fp16_t bits = 0;
    static_assert(sizeof(half_float::half) == sizeof(uint16_t), "half size mismatch");
    ::memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
}
static inline float mnn_qnn_fp16_to_float(mnn_qnn_fp16_t v) {
    half_float::half h;
    ::memcpy(&h, &v, sizeof(uint16_t));
    return (float)h;
}
} // namespace plugin
} // namespace MNN
#else
namespace MNN {
namespace plugin {
typedef __fp16 mnn_qnn_fp16_t;
static inline mnn_qnn_fp16_t mnn_qnn_float_to_fp16(float v) {
    return (__fp16)v;
}
static inline float mnn_qnn_fp16_to_float(mnn_qnn_fp16_t v) {
    return (float)v;
}
} // namespace plugin
} // namespace MNN
#endif

namespace MNN {
namespace plugin {

class RPCBuffer {
public:
    void* mPtr = nullptr;
    size_t mSize;
    int mFd;
    bool mReg = false;
    Qnn_MemHandle_t mHandle;
    ~ RPCBuffer() {
#ifdef MNN_QNN_NO_RPC_MEM
        if (mPtr) {
            free(mPtr);
            mPtr = nullptr;
        }
#else
        rpcmem_free(mPtr);
#endif
    }
    static RPCBuffer* alloc(size_t size) {
#ifdef MNN_QNN_NO_RPC_MEM
        // MSVC / Windows 上使用普通堆内存，fd 设为 -1（无效）。
        void* data = malloc(size);
        if (nullptr == data) {
            FUNC_PRINT(1);
            return nullptr;
        }
        return new RPCBuffer(data, -1, size);
#else
        void * data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
        if (nullptr == data) {
            FUNC_PRINT(1);
            return nullptr;
        }
        auto fd = rpcmem_to_fd(data);
        if (fd == -1) {
            FUNC_PRINT(1);
            rpcmem_free(data);
            return nullptr;
        }
        return new RPCBuffer(data, fd, size);
#endif
    }
    bool setToTensor(Qnn_Tensor_t* tensor, QNN_INTERFACE_VER_TYPE* QnnInterface, Qnn_ContextHandle_t context) {
#ifdef MNN_QNN_NO_RPC_MEM
        // Windows 平台没有 ION fd，使用 RAW client buffer 模式，让普通内存直接作为 tensor 数据。
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t clientBuf = {mPtr, (uint32_t)mSize};
        QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuf);
        return true;
#else
        if (!mReg) {
            Qnn_MemDescriptor_t memDescriptor = {
                {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
                QNN_TENSOR_GET_DATA_TYPE(tensor),
                QNN_MEM_TYPE_ION,
                {{-1}}};
            int curFd = mFd;
            memDescriptor.ionInfo.fd = curFd;
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);

            mHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
            auto res = QnnInterface->memRegister(context, &memDescriptor, 1, &(mHandle));
            if (res != QNN_SUCCESS) {
                const char* tname = QNN_TENSOR_GET_NAME(tensor);
                MNN_ERROR("memRegister fail %s (ctx=%p fd=%d), error: %llu\n", tname, context, curFd, res);
                return false;
            }
            mReg = true;
        }
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, mHandle);
        return true;
#endif
    }

private:
    RPCBuffer(void* ptr, int fd, size_t size) {
        mPtr = ptr;
        mFd = fd;
        mSize = size;
    }
};

namespace shape_inference {
class QNNPluginShapeRaw : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};
static bool computeIndex(PluginContext* ctx, int & index) {
    const std::vector<Tensor *> & inputs = ctx->inputs();
    auto attrAllShape = ctx->getAttr("allInputShape");
    if (nullptr == attrAllShape || nullptr == attrAllShape->list() || nullptr == attrAllShape->list()->i()) {
        MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allInputShape' attr.\n");
        return false;
    }
    int dimSum = 0;
    // MNN_PRINT("All Inputs Begin\n");
    for (int i = 0; i < inputs.size(); i++) {
        // inputs[i]->printShape();
        auto inputDim = inputs[i]->dimensions();
        dimSum += inputDim;
    }
    // MNN_PRINT("All Inputs End\n");
    if (0 == dimSum) {
        // Scalar
        index = 0;
        return true;
    }
    auto indexNumber = attrAllShape->list()->i()->size() / dimSum;
    for (int si=0; si<indexNumber; ++si) {
        auto dstSi = attrAllShape->list()->i()->data() + si * dimSum;
        bool valid = true;
        for (int i=0; i<inputs.size(); ++i) {
            auto inputDim = inputs[i]->dimensions();
            for (int j = 0; j < inputDim; j++) {
                if (inputs[i]->length(j) != dstSi[j]) {
                    valid = false;
                    break;
                }
            }
            dstSi += inputDim;
            if (!valid) {
                break;
            }
        }
        if (valid) {
            index = si;
            return true;
        }
    }
    return false;
}

bool QNNPluginShapeRaw::compute(InferShapeContext* ctx) {
    if (ctx->hasAttr("op")) {
        auto attr = ctx->getAttr("op");
        if (nullptr != attr->tensor() && nullptr != attr->tensor()->int8s()) {
            auto realop = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
            return SizeComputer::computeOutputSize(realop, ctx->inputs(), ctx->outputs());
        }
    } else {
        int shapeIndex = 0;
        if (!(computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to compute shape for Plugin Op.\n");
            return false;
        }

        std::string prefix = "o_" + std::to_string(shapeIndex) + "_";
        for (int i=0; i<ctx->outputs().size(); ++i) {
            auto dst = ctx->output(i);
            std::string key = prefix + std::to_string(i);
            auto attr = ctx->getAttr(key.c_str());

            if (nullptr == attr || nullptr == attr->tensor()) {
                MNN_ERROR("MNN_QNN: Failed to find raw shape %s.\n", key.c_str());
                return false;
            }
            auto blob = attr->tensor();
            dst->setType(blob->dataType());
            if (nullptr != blob->dims()) {
                dst->buffer().dimensions = blob->dims()->size();
                for (int j=0; j<blob->dims()->size(); ++j) {
                    dst->setLength(j, blob->dims()->data()[j]);
                }
            } else {
                dst->buffer().dimensions = 0;
            }
            TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
        }
        return true;
    }
    return false;
}
}

namespace backend {
static bool freeQnnTensor(Qnn_Tensor_t &tensor) {
  // free all pointer allocations in struct
  free((void *)QNN_TENSOR_GET_NAME(tensor));
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
  free(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor));

  auto quant    = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  auto encoding = quant.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    free(quant.axisScaleOffsetEncoding.scaleOffset);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    free(quant.bwAxisScaleOffsetEncoding.scales);
    if (quant.bwAxisScaleOffsetEncoding.offsets != nullptr) {
      free(quant.bwAxisScaleOffsetEncoding.offsets);
    }
  }
  return true;
}

static bool freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensor(tensors[i]);
  }
  free(tensors);

  return true;
}

struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
};

static bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src) {
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version           = src->version;
  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, ::strdup(tensorName));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }

  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      ::memcpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                             QNN_TENSOR_GET_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
          dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
      ::memcpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
                             QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

static bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                                 Qnn_Tensor_t *&tensorWrappers,
                                 uint32_t tensorsCount) {
  auto returnStatus = true;
  tensorWrappers    = (Qnn_Tensor_t *)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    MNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
#ifdef QNN_VERBOSE
        MNN_PRINT("Extracting tensorInfo for tensor Idx: %d.\n", (int)tIdx);
#endif
        tensorWrappers[tIdx] = QNN_TENSOR_INIT;
        deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }
  }
  return returnStatus;
}

template <typename T>
static bool copyGraphsInfoFromSrc(const T *graphInfoSrc, GraphInfo *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName = ::strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

static bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                                const uint32_t numGraphs,
                                GraphInfo **&graphsInfo) {
  if (!graphsInput) {
    MNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (GraphInfo **)calloc(numGraphs, sizeof(GraphInfo *));
  GraphInfo *graphInfoArr =
      (GraphInfo *)calloc(numGraphs, sizeof(GraphInfo));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    MNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
#ifdef QNN_VERBOSE
        MNN_PRINT("Extracting graphsInfo for graph Idx: %d", (int)gIdx);
#endif
        if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            copyGraphsInfoFromSrc(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
        } else if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            copyGraphsInfoFromSrc(&graphsInput[gIdx].graphInfoV3, &graphInfoArr[gIdx]);
        }
        graphsInfo[gIdx] = graphInfoArr + gIdx;
    }
  }
  if (true != returnStatus) {
    MNN_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensors(graphsInfo[gIdx]->inputTensors,
                                          graphsInfo[gIdx]->numInputTensors);
          freeQnnTensors(graphsInfo[gIdx]->outputTensors,
                                          graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }
  return true;
}

template<typename T>
static bool copyGraphsInfoFromBinaryInfo(const T & binaryInfo, GraphInfo **& graphsInfo, uint32_t & graphsCount) {
    if (binaryInfo.graphs) {
        if (!copyGraphsInfo(binaryInfo.graphs, binaryInfo.numGraphs, graphsInfo)) {
            MNN_ERROR("MNN_QNN: Failed while copying graphs Info.\n");
            return false;
        }
        graphsCount = binaryInfo.numGraphs;
        return true;
    }
    return false;
}

static bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                                          GraphInfo **&graphsInfo,
                                          uint32_t &graphsCount) {
    if (nullptr == binaryInfo) {
        MNN_ERROR("MNN_QNN: binaryInfo is nullptr.\n");
        return false;
    }
    graphsCount = 0;
    switch (binaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV1, graphsInfo, graphsCount);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV2, graphsInfo, graphsCount);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV3, graphsInfo, graphsCount);
        default:
            MNN_ERROR("MNN_QNN: Unrecognized system context binary info version.\n");
            return false;
    }
}

static bool freeGraphsInfo(GraphInfo ***graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    free((*graphsInfo)[i]->graphName);
    freeQnnTensors((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
    freeQnnTensors((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
  }
  free(**graphsInfo);
  free(*graphsInfo);
  *graphsInfo = nullptr;
  return true;
}

static size_t qnnTensorByteSize(const Qnn_Tensor_t* tensor) {
    size_t elements = 1;
    for (uint32_t i = 0; i < QNN_TENSOR_GET_RANK(tensor); ++i) {
        elements *= QNN_TENSOR_GET_DIMENSIONS(tensor)[i];
    }
    size_t bytesPerElement = 0;
    switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8:
            bytesPerElement = 1;
            break;
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
            bytesPerElement = 2;
            break;
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            bytesPerElement = 4;
            break;
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_FLOAT_64:
            bytesPerElement = 8;
            break;
        default:
            bytesPerElement = 4;
            break;
    }
    return elements * bytesPerElement;
}

class MMapReader {
private:
    void* mAddr = nullptr;
    file_t mFile = INVALID_FILE;
    size_t mSize = 0;
    void _clean() {
        if (nullptr != mAddr) {
            MNNUnmapFile(mAddr, mSize);
            mAddr = nullptr;
        }
        if (mFile != INVALID_FILE) {
            MNNCloseFile(mFile);
            mFile = INVALID_FILE;
        }
        mSize = 0;
    }
public:
    void* addr() const {
        return mAddr;
    }
    size_t size() const {
        return mSize;
    }
    MMapReader() {
        // Do nothing
    }
    ~MMapReader() {
        _clean();
    }
    bool open(const char* filename) {
        _clean();
        mFile = MNNOpenFile(filename, MNN_FILE_READ);
        if (mFile == INVALID_FILE) {
            return false;
        }
        mSize = MNNGetFileSize(mFile);
        if (mSize == 0) {
            _clean();
            return false;
        }
        mAddr = MNNMmapFile(mFile, mSize, true);
        if (mAddr == nullptr) {
            _clean();
            return false;
        }
        return true;
    }
};

class RawExecutorWrapper {
private:
    std::shared_ptr<void> mBackendContextOwner;
    QNN::QnnContext* mContext = nullptr;
    Qnn_ContextHandle_t mQnnContextHandle = nullptr;
    const QnnContext_Config_t** mQnnContextConfig = nullptr;
    std::vector<Qnn_GraphHandle_t> mQnnGraphHandleVec = {};
    QnnHtpGraph_CustomConfig_t mQnnHtpGraphCustomConfig{};
    QnnGraph_Config_t mQnnGraphConfig{};
    Qnn_ProfileHandle_t mQnnProfileHandle = nullptr;
    GraphInfo **mGraphsInfo = nullptr;
    uint32_t mGraphCount = 0;
    std::string mPath;
    std::unique_ptr<QNN::QNNPerf> mPerf;
    std::unique_ptr<QNN::QNNTensorDumper> mTensorDumper;
    // IO tensors are bound to rpcmem buffers registered with the QNN context;
    // the HTP backend does not accept plain heap memory through clientBuf.
    // When the graph data type differs from the MNN tensor data type (e.g.
    // fp16 graph vs fp32 host tensor), data is converted during the copy.
    struct IoStaging {
        int tensorIndex = -1;
        std::shared_ptr<RPCBuffer> buffer;
        bool graphIsFp16 = false;
        bool hostIsFp32 = false;
    };
    std::vector<IoStaging> mInputStaging;
    std::vector<IoStaging> mOutputStaging;
    std::vector<std::pair<const MNN::Tensor*, std::string>> mStagedInputs;
    std::vector<std::pair<const MNN::Tensor*, std::string>> mStagedOutputs;

public:
    RawExecutorWrapper(QNN::QnnContext* context, std::shared_ptr<void> backendContextOwner)
        : mBackendContextOwner(std::move(backendContextOwner)), mContext(context) {
        if (QNN::getLoadedQNNBackend() ==
            QNN::QnnBackendKind::Htp) {
            mPerf = QNN::QNNPerf::create(&mContext->QnnInterface);
            if (mPerf) {
                mPerf->setPowerConfigBurst();
                mPerf->setRpcLatencyAndPolling();
            }
        }
    }
    ~ RawExecutorWrapper() {
        if (mQnnProfileHandle) {
            mContext->QnnInterface.profileFree(mQnnProfileHandle);
            mQnnProfileHandle = nullptr;
        }
        if (nullptr != mQnnContextHandle) {
            CALL_QNN(mContext->QnnInterface.contextFree(mQnnContextHandle, nullptr));
        }
        freeGraphsInfo(&mGraphsInfo, mGraphCount);
    }

    void setTensorDump(bool enabled, const std::string& outputDirectory) {
        if (enabled) {
            mTensorDumper.reset(
                new QNN::QNNTensorDumper(true, outputDirectory));
        } else {
            mTensorDumper.reset();
        }
    }

    bool compileModel(const std::string& path, size_t offset, size_t size, const std::vector<std::string>& allGraphName) {
        mPath = path;
        void* buffer = nullptr;
        std::vector<char> bufferVec;
        MMapReader reader;
        if (!reader.open(path.c_str())) {
            MNN_ERROR("MNN_QNN: Failed to map Context binary: %s.\n", path.c_str());
            return false;
        }
        if (size == 0 && offset != 0) {
            MNN_ERROR(
                "MNN_QNN: Context binary offset requires an explicit size: "
                "offset=%zu.\n",
                offset);
            return false;
        }
        if (size > 0) {
            if (offset > reader.size() || size > reader.size() - offset) {
                MNN_ERROR(
                    "MNN_QNN: Context binary range is invalid: "
                    "offset=%zu size=%zu file=%zu.\n",
                    offset, size, reader.size());
                return false;
            }
            bufferVec.resize(size);
            buffer = bufferVec.data();
            ::memcpy(buffer, static_cast<const char*>(reader.addr()) + offset, size);
        } else {
            buffer = reader.addr();
            size = reader.size();
        }

        // 1. Set mGraphsInfo and mGraphCount from the buffer.
        {
            QnnSystemContext_Handle_t systemContextHandle = nullptr;
            if (QNN_SUCCESS != mContext->systemInterface.systemContextCreate(&systemContextHandle)) {
                MNN_ERROR("Could not create system context handle.");
                return false;
            }
            Qnn_ContextBinarySize_t binarySize = 0;
            const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
            const auto metadataStatus = mContext->systemInterface.systemContextGetBinaryInfo(
                systemContextHandle, buffer, size, &binaryInfo, &binarySize);
            const bool metadataValid = (metadataStatus & 0xFFFF) == QNN_SUCCESS &&
                                       copyMetadataToGraphsInfo(binaryInfo, mGraphsInfo, mGraphCount) &&
                                       mGraphsInfo != nullptr && mGraphCount > 0;
            const auto freeStatus = mContext->systemInterface.systemContextFree(systemContextHandle);
            if ((freeStatus & 0xFFFF) != QNN_SUCCESS) {
                MNN_ERROR("Could not free system context handle.");
                return false;
            }
            if (!metadataValid) {
                MNN_ERROR(
                    "MNN_QNN: Invalid or empty Context graph metadata "
                    "(status=%d).\n",
                    static_cast<int>(metadataStatus & 0xFFFF));
                return false;
            }
            if (allGraphName.size() != mGraphCount) {
                MNN_ERROR(
                    "MNN_QNN: Context graph-name metadata count does not "
                    "match: "
                    "names=%zu graphs=%u.\n",
                    allGraphName.size(), mGraphCount);
                return false;
            }
            for (uint32_t i = 0; i < mGraphCount; ++i) {
                if (mGraphsInfo[i] == nullptr || mGraphsInfo[i]->graphName == nullptr ||
                    mGraphsInfo[i]->graphName[0] == '\0' || allGraphName[i].empty()) {
                    MNN_ERROR(
                        "MNN_QNN: Context graph metadata contains an "
                        "empty graph entry at index %u.\n",
                        i);
                    return false;
                }
            }
        }

#ifdef QNN_VERBOSE
        for (uint32_t g = 0; g < mGraphCount; ++g) {
            auto* graphInfo = mGraphsInfo[g];
            MNN_PRINT("MNN_QNN: graph[%u] name=%s inputs=%u outputs=%u\n", g, graphInfo->graphName,
                      graphInfo->numInputTensors, graphInfo->numOutputTensors);
            for (uint32_t t = 0; t < graphInfo->numInputTensors; ++t) {
                auto& tensor = graphInfo->inputTensors[t];
                MNN_PRINT("MNN_QNN:   in[%u] name=%s dtype=%u rank=%u bytes=%zu\n", t, QNN_TENSOR_GET_NAME(tensor),
                          QNN_TENSOR_GET_DATA_TYPE(tensor), QNN_TENSOR_GET_RANK(tensor), qnnTensorByteSize(&tensor));
            }
            for (uint32_t t = 0; t < graphInfo->numOutputTensors; ++t) {
                auto& tensor = graphInfo->outputTensors[t];
                MNN_PRINT("MNN_QNN:   out[%u] name=%s dtype=%u rank=%u bytes=%zu\n", t, QNN_TENSOR_GET_NAME(tensor),
                          QNN_TENSOR_GET_DATA_TYPE(tensor), QNN_TENSOR_GET_RANK(tensor), qnnTensorByteSize(&tensor));
            }
        }
#endif

        // 2. Retrieve graphs.
        {
            auto error = mContext->QnnInterface.contextValidateBinary(
                mContext->backendHandle, mContext->deviceHandle, mQnnContextConfig, buffer, size);
            if (QNN_SUCCESS != error) {
                MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
                return false;
            }

            // Create Graph profile
            MNN::QNN::createProfileHandle(mContext->QnnInterface, mContext->backendHandle, &mQnnProfileHandle);

            const auto createStatus = mContext->QnnInterface.contextCreateFromBinary(
                mContext->backendHandle, mContext->deviceHandle, mQnnContextConfig, buffer, size,
                &mQnnContextHandle, mQnnProfileHandle);
            if ((createStatus & 0xFFFF) != QNN_SUCCESS || mQnnContextHandle == nullptr) {
                MNN_ERROR("MNN_QNN: contextCreateFromBinary failed: %d.\n", static_cast<int>(createStatus & 0xFFFF));
                return false;
            }

            mQnnGraphHandleVec.resize(mGraphCount, nullptr);

            std::vector<GraphInfo*> sortedGraphsInfo(mGraphCount, nullptr);
            std::map<std::string, GraphInfo*> graphInfoMap;
            for (int i = 0; i < mGraphCount; ++i) {
                graphInfoMap[mGraphsInfo[i]->graphName] = mGraphsInfo[i];
            }

            for (int i = 0; i < mGraphCount; ++i) {
                auto it = graphInfoMap.find(allGraphName[i]);
                if (it == graphInfoMap.end()) {
                    MNN_ERROR("MNN_QNN: Context graph '%s' was not found.\n", allGraphName[i].c_str());
                    return false;
                }
                sortedGraphsInfo[i] = it->second;
            }
            for (int i = 0; i < mGraphCount; ++i) {
                mGraphsInfo[i] = sortedGraphsInfo[i];
            }

            for (int i = 0; i < mGraphCount; i++) {
                const auto retrieveStatus = mContext->QnnInterface.graphRetrieve(
                    mQnnContextHandle, mGraphsInfo[i]->graphName, &(mQnnGraphHandleVec[i]));
                if ((retrieveStatus & 0xFFFF) != QNN_SUCCESS || mQnnGraphHandleVec[i] == nullptr) {
                    MNN_ERROR("MNN_QNN: graphRetrieve failed for '%s': %d.\n", mGraphsInfo[i]->graphName,
                              static_cast<int>(retrieveStatus & 0xFFFF));
                    return false;
                }
#ifdef QNN_VERBOSE
                MNN_PRINT("MNN_QNN: executor=%p compileModel graph[%d] name=%s context=%p graphHandle=%p\n",
                          (void*)this, i, mGraphsInfo[i]->graphName, (void*)mQnnContextHandle,
                          (void*)mQnnGraphHandleVec[i]);
#endif
            }
        }


        return true;
    }
    Qnn_Tensor_t* _findInput(const std::string& name, int index) {
        GraphInfo* graph = mGraphsInfo[index];
        for (int j=0; j<graph->numInputTensors; ++j) {
            auto& dstT = graph->inputTensors[j];
#ifdef QNN_VERBOSE
            MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
#endif
            if (name == dstT.v1.name) {
                return &dstT;
            }
        }
        return nullptr;
    }
    Qnn_Tensor_t* _findOutput(const std::string& name, int index) {
        GraphInfo* graph = mGraphsInfo[index];
        for (int j=0; j<graph->numOutputTensors; ++j) {
            auto& dstT = graph->outputTensors[j];
#ifdef QNN_VERBOSE
            MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
#endif
            if (name == dstT.v1.name) {
                return &dstT;
            }
        }
        return nullptr;
    }
    const Qnn_Tensor_t* findInput(const std::string& name, int index) {
        if (index < 0 || static_cast<uint32_t>(index) >= mGraphCount ||
            mGraphsInfo == nullptr || mGraphsInfo[index] == nullptr) {
            return nullptr;
        }
        return _findInput(name, index);
    }
    // Bind every declared IO tensor to a registered rpcmem staging buffer.
    bool _setupStaging(const std::vector<std::pair<const MNN::Tensor*, std::string>>& ioTensors,
                       Qnn_Tensor_t* graphTensors, uint32_t numGraphTensors,
                       std::vector<IoStaging>& staging) {
        staging.clear();
        for (size_t i = 0; i < ioTensors.size(); ++i) {
            Qnn_Tensor_t* match = nullptr;
            for (uint32_t j = 0; j < numGraphTensors; ++j) {
                if (ioTensors[i].second == graphTensors[j].v1.name) {
                    match = &graphTensors[j];
                    break;
                }
            }
            if (nullptr == match) {
                MNN_ERROR("MNN_QNN: staging tensor was not found: %s\n", ioTensors[i].second.c_str());
                return false;
            }
            auto size = qnnTensorByteSize(match);
            std::shared_ptr<RPCBuffer> buffer(RPCBuffer::alloc(size));
            if (nullptr == buffer) {
                MNN_ERROR("MNN_QNN: staging alloc failed for %s (%zu bytes)\n", ioTensors[i].second.c_str(), size);
                return false;
            }
            buffer->mSize = size;
            if (!buffer->setToTensor(match, &mContext->QnnInterface, mQnnContextHandle)) {
                MNN_ERROR("MNN_QNN: staging bind failed for %s\n", ioTensors[i].second.c_str());
                return false;
            }
            IoStaging item;
            item.tensorIndex = (int)i;
            item.buffer = buffer;
            item.graphIsFp16 = QNN_TENSOR_GET_DATA_TYPE(match) == QNN_DATATYPE_FLOAT_16;
            item.hostIsFp32 = ioTensors[i].first->getType().code == halide_type_float &&
                              ioTensors[i].first->getType().bits == 32;
            staging.push_back(item);
        }
        return true;
    }
    bool setupIoStaging(const std::vector<std::pair<const MNN::Tensor*, std::string>>& inputs,
                        const std::vector<std::pair<const MNN::Tensor*, std::string>>& outputs, int shapeIndex) {
        mStagedInputs = inputs;
        mStagedOutputs = outputs;
        GraphInfo* graph = mGraphsInfo[shapeIndex];
        if (!_setupStaging(inputs, graph->inputTensors, graph->numInputTensors, mInputStaging)) {
            return false;
        }
        if (!_setupStaging(outputs, graph->outputTensors, graph->numOutputTensors, mOutputStaging)) {
            return false;
        }
        return true;
    }
    void convertInputsToGraph() {
        for (auto& stage : mInputStaging) {
            auto srcTensor = mStagedInputs[stage.tensorIndex].first;
            if (stage.graphIsFp16 && stage.hostIsFp32) {
                auto src = srcTensor->host<float>();
                auto dst = (mnn_qnn_fp16_t*)stage.buffer->mPtr;
                const size_t count = stage.buffer->mSize / sizeof(mnn_qnn_fp16_t);
                for (size_t k = 0; k < count; ++k) {
                    dst[k] = mnn_qnn_float_to_fp16(src[k]);
                }
            } else {
                ::memcpy(stage.buffer->mPtr, srcTensor->host<void>(), stage.buffer->mSize);
            }
        }
    }
    void convertOutputsFromGraph() {
        for (auto& stage : mOutputStaging) {
            auto dstTensor = mStagedOutputs[stage.tensorIndex].first;
            if (stage.graphIsFp16 && stage.hostIsFp32) {
                auto src = (const mnn_qnn_fp16_t*)stage.buffer->mPtr;
                auto dst = dstTensor->host<float>();
                const size_t count = stage.buffer->mSize / sizeof(mnn_qnn_fp16_t);
                for (size_t k = 0; k < count; ++k) {
                    dst[k] = mnn_qnn_fp16_to_float(src[k]);
                }
            } else {
                ::memcpy(dstTensor->host<void>(), stage.buffer->mPtr, stage.buffer->mSize);
            }
        }
    }
    void setupAddress(const std::vector<std::pair<const MNN::Tensor *, std::string>>& inputs, std::vector<std::pair<const MNN::Tensor *, std::string>>& outputs, int shapeIndex) {
        GraphInfo* graph = mGraphsInfo[shapeIndex];
        Qnn_GraphHandle_t qnnGraphHandle = mQnnGraphHandleVec[shapeIndex];

        // MNN_PRINT("%s, Input:%d, output:%d\n", mPath.c_str(), inputs.size(), outputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            auto t = inputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numInputTensors; ++j) {
                auto& dstT = graph->inputTensors[j];
#ifdef QNN_VERBOSE
                MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
#endif
                if (inputs[i].second == dstT.v1.name) {
                    dstT.v1.clientBuf.data = t->host<void>();
                    dstT.v1.clientBuf.dataSize = t->usize();
                    find = true;
                    break;
                }
            }
            if (!find) {
                MNN_ERROR("%s, can't find %d input: %s\n", mPath.c_str(), i, inputs[i].second.c_str());
            }
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto t = outputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numOutputTensors; ++j) {
                auto& dstT = graph->outputTensors[j];
#ifdef QNN_VERBOSE
                MNN_PRINT("output name: %s %s\n", outputs[i].second.c_str(), dstT.v1.name);
#endif
                if (outputs[i].second == dstT.v1.name) {
                    dstT.v1.clientBuf.data = t->host<void>();
                    dstT.v1.clientBuf.dataSize = t->usize();
                    find = true;
                    break;
                }
            }
            if (!find) {
                MNN_ERROR("%s, can't find %d output: %s\n", mPath.c_str(), i, outputs[i].second.c_str());
            }
        }
    }
    void setupState(RPCBuffer* mask, std::vector<RPCBuffer*> statesInputs, std::vector<RPCBuffer*> statesOutput, int index) {
        auto maskTensor = _findInput(extraIoPrefix() + "_mask", index);
        if (nullptr == maskTensor) {
            MNN_ERROR("Can't find mask from qnn model\n");
            return;
        }
        mask->setToTensor(maskTensor, &mContext->QnnInterface, mQnnContextHandle);
        for (int i=0; i<statesInputs.size(); ++i) {
            auto t = _findInput(extraIoPrefix() + "_i" + std::to_string(i), index);
            if (nullptr == t) {
                MNN_ERROR("Can't find %d input tensor of state\n", i);
                continue;
            }
            statesInputs[i]->setToTensor(t, &mContext->QnnInterface, mQnnContextHandle);
        }
        for (int i=0; i<statesOutput.size(); ++i) {
            auto t = _findOutput(extraIoPrefix() + "_o" + std::to_string(i), index);
            if (nullptr == t) {
                MNN_ERROR("Can't find %d output tensor of state\n", i);
                continue;
            }
            statesOutput[i]->setToTensor(t, &mContext->QnnInterface, mQnnContextHandle);
        }
    }

    bool invokModel(int shapeIndex) {
        GraphInfo* graph = mGraphsInfo[shapeIndex];
        Qnn_GraphHandle_t qnnGraphHandle = mQnnGraphHandleVec[shapeIndex];
        convertInputsToGraph();
        // Ensure all output tensors have memory allocated; allocate temp buffers for those without.
        auto tempBuffers = MNN::QNN::ensureOutputTensorsMemory(graph->outputTensors, graph->numOutputTensors);
        const auto executeStatus = mContext->QnnInterface.graphExecute(
            qnnGraphHandle, graph->inputTensors, graph->numInputTensors, graph->outputTensors, graph->numOutputTensors,
            mQnnProfileHandle, nullptr);
        if ((executeStatus & 0xFFFF) != QNN_SUCCESS) {
            MNN_ERROR("MNN_QNN: offline graphExecute failed: %d.\n", static_cast<int>(executeStatus & 0xFFFF));
            MNN::QNN::freeOutputTensorsTempMemory(graph->outputTensors, tempBuffers);
            return false;
        }
        MNN::QNN::doProfile(mContext->QnnInterface, mQnnProfileHandle);
        if (mTensorDumper != nullptr) {
            std::vector<Qnn_Tensor_t> dumpTensors(
                graph->outputTensors,
                graph->outputTensors + graph->numOutputTensors);
            for (const auto& stage : mOutputStaging) {
                if (stage.tensorIndex < 0 ||
                    stage.tensorIndex >= mStagedOutputs.size()) {
                    continue;
                }
                const auto& name = mStagedOutputs[stage.tensorIndex].second;
                for (auto& tensor : dumpTensors) {
                    if (QNN_TENSOR_GET_NAME(tensor) == nullptr ||
                        name != QNN_TENSOR_GET_NAME(tensor)) {
                        continue;
                    }
                    Qnn_ClientBuffer_t clientBuffer = {
                        stage.buffer->mPtr,
                        static_cast<uint32_t>(stage.buffer->mSize)};
                    QNN_TENSOR_SET_MEM_TYPE(tensor,
                                            QNN_TENSORMEMTYPE_RAW);
                    QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);
                    break;
                }
            }
            mTensorDumper->dump(dumpTensors.data(), dumpTensors.size());
        }
        // Free temporarily allocated output tensor buffers.
        MNN::QNN::freeOutputTensorsTempMemory(graph->outputTensors, tempBuffers);
        convertOutputsFromGraph();
        return true;
    }
};

class QNNPluginExecuteRaw : public CPUComputeKernel {
private:
    std::unique_ptr<RawExecutorWrapper> mRawExecutor;
    std::shared_ptr<void> mBackendContextOwner;
    QNN::QnnContext* mQnnContext = nullptr;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mInputs;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mOutputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealInputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealOutputs;
    // Offline ctx graphs bake fixed-point I/O (e.g. int8 image/audio inputs)
    // while the wrapper .mnn session tensors stay float32. For every ctx
    // input declared SFIXED8 keep its quantization scale so compute() can
    // quantize the float session tensor into the raw int8 client buffer.
    std::vector<bool> mInputIsFixed8;
    std::vector<float> mInputFixed8Scale;
    int mShapeIndex;
    bool mDedicatedQnnSession = false;
    bool mIoStagingReady = false;

    struct StateTensor {
        std::shared_ptr<RPCBuffer> data;
        int inside;
        int outside;
        std::vector<std::shared_ptr<RPCBuffer>> update;
    };
    std::vector<StateTensor> mStateInput;
    int mStateCurrent = 0;
    int mStateMaxSize = 0;
    std::vector<int> mSeqLen;
    std::shared_ptr<RPCBuffer> mMask;
    const float mMinValue = -32700.0f;
    // Quantize a float32 session tensor into the raw int8 client buffer
    // expected by the offline ctx graph (round-half-away-from-zero plus
    // symmetric saturation, matching the DSP inputIO contract).
    bool _quantizeFloatToInt8(const MNN::Tensor* src, MNN::Tensor* dst, float scale) {
        if (src->getType() != halide_type_of<float>() || !(scale > 0.0f)) return false;
        std::shared_ptr<Tensor> unpacked(Tensor::create<float>(dst->shape(), nullptr, dst->getDimensionType()));
        if (qnnConvertTensor(src, unpacked.get()) != NO_ERROR) return false;
        const float* input = unpacked->host<float>();
        auto* output = dst->host<int8_t>();
        for (int index = 0; index < dst->elementSize(); ++index) {
            const float value = std::round(input[index] / scale);
            if (!std::isfinite(value)) return false;
            output[index] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, value)));
        }
        return true;
    }
    void _loadState(int stateNumber, std::vector<int> seqLen) {
        if (stateNumber == 0) {
            return;
        }
        mMask.reset(RPCBuffer::alloc(mStateMaxSize * sizeof(mnn_qnn_fp16_t)));
        auto maskPtr = (mnn_qnn_fp16_t*)mMask->mPtr;
        const mnn_qnn_fp16_t minValueFp16 = mnn_qnn_float_to_fp16(mMinValue);
        for (int i=0; i<mStateMaxSize; ++i) {
            maskPtr[i] = minValueFp16;
        }
        for (int i=0; i<mStateInput.size(); ++i) {
            mStateInput[i].data.reset(RPCBuffer::alloc(mStateMaxSize * mStateInput[i].inside * mStateInput[i].outside *
                                                       sizeof(mnn_qnn_fp16_t)));
            mStateInput[i].update.resize(seqLen.size());
            for (int j=0; j<seqLen.size(); ++j) {
                mStateInput[i].update[j].reset(RPCBuffer::alloc(mStateInput[i].inside * mStateInput[i].outside *
                                                                seqLen[j] * sizeof(mnn_qnn_fp16_t)));
            }
        }
    }

public:
    QNNPluginExecuteRaw() = default;
    QNNPluginExecuteRaw(QNN::QnnContext* context, std::shared_ptr<void> backendContextOwner)
        : mBackendContextOwner(std::move(backendContextOwner)), mQnnContext(context),
          mDedicatedQnnSession(true) {}
    ~QNNPluginExecuteRaw() {
        mRealInputs.clear();
        mRealOutputs.clear();
        mRawExecutor.reset();
    }
    bool init(CPUKernelContext* ctx) override {

        if (mDedicatedQnnSession) {
            if (mQnnContext == nullptr || mQnnContext->backendHandle == nullptr ||
                mQnnContext->systemInterface.systemContextCreate == nullptr ||
                mQnnContext->systemInterface.systemContextGetBinaryInfo == nullptr ||
                mQnnContext->systemInterface.systemContextFree == nullptr) {
                MNN_ERROR("MNN_QNN: selected offline runtime context is invalid.\n");
                return false;
            }
        } else {
            if (!QNN::prepareQnnPluginRuntime(false)) {
                return false;
            }
            mQnnContext = &QNN::gContext;
        }
        const auto* pathAttr = ctx->getAttr("path");
        if (!pathAttr || !pathAttr->s() || pathAttr->s()->size() == 0) return false;
        // Stateful LLM wrappers retain their legacy path. The strict NPU API
        // currently accepts stateless contexts only and rejects this explicitly.
        if (mDedicatedQnnSession && ctx->getAttr("state") != nullptr) return false;
        auto seqLen = ctx->getAttr("seq_len");
        if (nullptr != seqLen && nullptr != seqLen->list()) {
            mSeqLen.resize(seqLen->list()->i()->size());
            ::memcpy(mSeqLen.data(), seqLen->list()->i()->data(), mSeqLen.size() * sizeof(int));
            for (int i=0; i<mSeqLen.size(); ++i) {
                FUNC_PRINT(mSeqLen[i]);
            }
        }
        auto state = ctx->getAttr("state");
        int stateNumber = 0;
        if (nullptr != state) {
            int axis = 0;
            auto ref = flexbuffers::GetRoot(state->tensor()->uint8s()->data(), state->tensor()->uint8s()->size());
            auto refMap = ref.AsMap();
            auto keys = refMap.Keys();
            std::vector<std::vector<int>> stateShape;
            for (int i=0; i<keys.size(); ++i) {
                auto key = keys[i].AsKey();
                if (std::string(key) == "number") {
                    stateNumber = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "max_length") {
                    mStateMaxSize = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "axis") {
                    axis = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "shape") {
                    auto shapeVectors = refMap.Values()[i].AsVector();
                    for (int u=0; u<shapeVectors.size(); ++u) {
                        auto shapeV = shapeVectors[u].AsVector();
                        std::vector<int> shapes;
                        for (int v=0; v<shapeV.size(); ++v) {
                            shapes.emplace_back(shapeV[v].AsInt32());
                        }
                        stateShape.emplace_back(shapes);
                    }
                    continue;
                }
            }
            mStateInput.resize(stateShape.size());
            for (int i=0; i<stateShape.size(); ++i) {
                auto& shape = stateShape[i];
                auto& input = mStateInput[i];
                input.outside = 1;
                for (int j=0; j<axis; ++j) {
                    input.outside *= shape[j];
                }
                auto axisLength = shape[axis];
                MNN_ASSERT(1 == axisLength);
                input.inside = 1;
                for (int j=axis+1; j<shape.size(); ++j) {
                    input.inside *= shape[j];
                }
            }
        }
        FUNC_PRINT(stateNumber);
        _loadState(stateNumber, mSeqLen);
        std::string qnnDirPath = ctx->dir_path();
        auto path = MNNFilePathConcat(qnnDirPath, ctx->getAttr("path")->s()->str());

        std::vector<std::string> allGraphName;
        auto allGraphNameAttr = ctx->getAttr("allGraphName");
        if (allGraphNameAttr && allGraphNameAttr->list() && allGraphNameAttr->list()->s()) {
            auto graphNames = allGraphNameAttr->list()->s();
            for (int i = 0; i < graphNames->size(); ++i) {
                allGraphName.push_back(graphNames->GetAsString(i)->str());
            }
        } else {
            MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allGraphName' attr.\n");
            return false;
        }

        size_t binaryOffset = 0;
        auto offsetAttr = ctx->getAttr("offset");
        if (offsetAttr && offsetAttr->list() && offsetAttr->list()->i() && offsetAttr->list()->i()->size() == 2) {
            const int * dataPtr = offsetAttr->list()->i()->data();
            int lowSrc = dataPtr[0];
            int highSrc = dataPtr[1];

            uint32_t lowDst, highDst;
            ::memcpy(&lowDst, &lowSrc, sizeof(uint32_t));
            ::memcpy(&highDst, &highSrc, sizeof(uint32_t));

            binaryOffset = (static_cast<size_t>(highDst) << 32) | static_cast<size_t>(lowDst);
        }

        size_t binarySize = 0;
        auto sizeAttr = ctx->getAttr("size");
        if (sizeAttr && sizeAttr->list() && sizeAttr->list()->i() && sizeAttr->list()->i()->size() == 2) {
            const int * dataPtr = sizeAttr->list()->i()->data();
            int lowSrc = dataPtr[0];
            int highSrc = dataPtr[1];

            uint32_t lowDst, highDst;
            ::memcpy(&lowDst, &lowSrc, sizeof(uint32_t));
            ::memcpy(&highDst, &highSrc, sizeof(uint32_t));

            binarySize = (static_cast<size_t>(highDst) << 32) | static_cast<size_t>(lowDst);
        }
        mRawExecutor.reset(new RawExecutorWrapper(mQnnContext, mBackendContextOwner));
        const auto dumpAttr = ctx->getAttr("dump_intermediate_outputs");
        const bool dumpIntermediateOutputs =
            dumpAttr != nullptr && dumpAttr->i() != 0;
        const auto dumpDirectory =
            ctx->dir_path().empty()
                ? std::string("qnn_intermediate_outputs")
                : MNNFilePathConcat(ctx->dir_path(),
                                    "qnn_intermediate_outputs");
        mRawExecutor->setTensorDump(dumpIntermediateOutputs, dumpDirectory);
        return mRawExecutor->compileModel(path, binaryOffset, binarySize, allGraphName);
    }

    bool resize(CPUKernelContext* ctx) override {
        mIoStagingReady = false;
        int shapeIndex = 0;
        if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
            return false;
        }
        mShapeIndex = shapeIndex;
        const auto* graphs = ctx->getAttr("allGraphName");
        const auto* in = ctx->getAttr("inputs");
        const auto* out = ctx->getAttr("outputs");
        if (!graphs || !graphs->list() || !graphs->list()->s() || shapeIndex < 0 ||
            shapeIndex >= graphs->list()->s()->size() || !in || !in->list() || !in->list()->s() ||
            !out || !out->list() || !out->list()->s() ||
            in->list()->s()->size() != ctx->inputs().size() ||
            out->list()->s()->size() != ctx->outputs().size()) return false;

        auto inputs = ctx->getAttr("inputs")->list();
        auto inputTensor = ctx->inputs();
        MNN_ASSERT(inputs->s()->size() == inputTensor.size());
        mInputs.resize(inputs->s()->size());
        mRealInputs.resize(inputTensor.size());
        mInputIsFixed8.assign(inputTensor.size(), false);
        mInputFixed8Scale.assign(inputTensor.size(), 0.0f);
        for (int i=0; i<inputs->s()->size(); ++i) {
            mInputs[i].second = inputs->s()->GetAsString(i)->str();
            // The offline context binary is authoritative for I/O types:
            // allocate the raw input buffer with the baked tensor dtype so
            // setupAddress maps a clientBuf whose size matches the graph.
            const Qnn_Tensor_t* ctxInput =
                mDedicatedQnnSession
                    ? mRawExecutor->findInput(mInputs[i].second, shapeIndex)
                    : nullptr;
            if (mDedicatedQnnSession && ctxInput != nullptr &&
                ctxInput->v1.dataType == QNN_DATATYPE_SFIXED_POINT_8 &&
                ctxInput->v1.quantizeParams.scaleOffsetEncoding.scale > 0.0f) {
                std::vector<int> dims(ctxInput->v1.rank);
                for (int d=0; d<static_cast<int>(ctxInput->v1.rank); ++d) {
                    dims[d] = static_cast<int>(ctxInput->v1.dimensions[d]);
                }
                mRealInputs[i].reset(Tensor::create<int8_t>(dims, nullptr, Tensor::CAFFE));
                mInputIsFixed8[i] = true;
                mInputFixed8Scale[i] = ctxInput->v1.quantizeParams.scaleOffsetEncoding.scale;
            } else {
                mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::CAFFE));
            }
            mInputs[i].first = mRealInputs[i].get();
        }
        auto outputs = ctx->getAttr("outputs")->list();
        auto outputTensor = ctx->outputs();
        mOutputs.resize(outputs->s()->size());
        MNN_ASSERT(outputs->s()->size() == outputTensor.size());
        mRealOutputs.resize(outputTensor.size());
        for (int i=0; i<outputs->s()->size(); ++i) {
            mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::CAFFE));
            mOutputs[i].second = outputs->s()->GetAsString(i)->str();
            mOutputs[i].first = mRealOutputs[i].get();
        }
        mRawExecutor->setupAddress(mInputs, mOutputs, mShapeIndex);
        if (mDedicatedQnnSession) {
            if (!mRawExecutor->setupIoStaging(mInputs, mOutputs, mShapeIndex)) {
                return false;
            }
            mIoStagingReady = true;
        }
        if (mStateMaxSize > 0) {
            std::vector<RPCBuffer*> states(mStateInput.size());
            for (int i=0; i<mStateInput.size(); ++i) {
                states[i] = mStateInput[i].data.get();
            }
            std::vector<RPCBuffer*> statesOutput(mStateInput.size());
            for (int i=0; i<mStateInput.size(); ++i) {
                statesOutput[i] = mStateInput[i].update[mShapeIndex].get();
            }

            mRawExecutor->setupState(mMask.get(), states, statesOutput, mShapeIndex);
        }
        return true;
    }

    bool compute(CPUKernelContext* ctx) override {
        AUTOTIME;
        if (mDedicatedQnnSession && !mIoStagingReady) {
            MNN_ERROR("MNN_QNN: offline I/O staging is not ready.\n");
            return false;
        }
        int shapeIndex = mShapeIndex;
        std::string graphName = ctx->getAttr("allGraphName")->list()->s()->GetAsString(shapeIndex)->str();

#ifdef QNN_VERBOSE
        MNN_PRINT("Graph name:%s, %d\n", graphName.c_str(), shapeIndex);
#endif
        auto inputTensor = ctx->inputs();
        auto outputTensor = ctx->outputs();

        for (int i=0; i<mInputs.size(); ++i) {
            if (mInputIsFixed8[i]) {
                if (!_quantizeFloatToInt8(inputTensor[i], mRealInputs[i].get(),
                                         mInputFixed8Scale[i])) return false;
            } else if (mDedicatedQnnSession) {
                if (qnnConvertTensor(inputTensor[i], mRealInputs[i].get()) != NO_ERROR) return false;
            } else {
                // Legacy CPU plugins may receive backend-owned FP16/packed tensors.
                ctx->backend()->onCopyBuffer(inputTensor[i], mRealInputs[i].get());
            }
        }
        // If has remove, remove invalid state
        auto meta = (KVMeta*)(ctx->backend()->getMetaPtr());
        if (nullptr != meta && mStateInput.size() > 0) {
            auto maskPtr = (mnn_qnn_fp16_t*)mMask->mPtr;
            if (meta->remove > 0) {
                if (meta->remove > mStateCurrent) {
                    MNN_ERROR("QNN: Error: Remove %d larger than current = %d\n", meta->remove, mStateCurrent);
                    return false;
                }
                mStateCurrent-= meta->remove;
                const mnn_qnn_fp16_t minValueFp16 = mnn_qnn_float_to_fp16(mMinValue);
                for (int i=0; i<meta->remove; ++i) {
                    maskPtr[i + mStateCurrent] = minValueFp16;
                }
            }
        }
        if (!mRawExecutor->invokModel(shapeIndex)) {
            return false;
        }
        for (int i=0; i<mOutputs.size(); ++i) {
            if (mDedicatedQnnSession) {
                if (qnnConvertTensor(mRealOutputs[i].get(), outputTensor[i]) != NO_ERROR) return false;
            } else {
                // Preserve the owning CPU backend's storage precision and packing.
                ctx->backend()->onCopyBuffer(mRealOutputs[i].get(), outputTensor[i]);
            }
        }
        // Update State
        if (nullptr != meta && mStateInput.size() > 0) {
            auto maskPtr = (mnn_qnn_fp16_t*)mMask->mPtr;
            if (meta->add + mStateCurrent > mStateMaxSize) {
                MNN_ERROR("QNN: Error: KV length %d larger than max size = %d\n", meta->add + mStateCurrent, mStateMaxSize);
                return false;
            }
            const mnn_qnn_fp16_t zeroFp16 = mnn_qnn_float_to_fp16(0.0f);
            for (int i=0; i<meta->add; ++i) {
                maskPtr[i + mStateCurrent] = zeroFp16;
            }
            // Temply use StateOutputs[0] size to compute seq_len
            int bytes = 2;
            int seqLen = mSeqLen[mShapeIndex];
            for (int i=0; i<mStateInput.size(); ++i) {
                auto& input = mStateInput[i];
                for (int y=0; y<input.outside; ++y) {
                    auto dstOffset = y * input.inside * mStateMaxSize + mStateCurrent * input.inside;
                    auto srcOffset = y * input.inside * seqLen;
                    auto dst = (uint8_t*)input.data->mPtr + dstOffset * bytes;
                    auto src = (uint8_t*)input.update[mShapeIndex]->mPtr + srcOffset * bytes;
                    ::memcpy(dst, src, meta->add * input.inside * bytes);
                }
            }
            mStateCurrent += meta->add;
        }
        return true;
    }
};

} // namespace backend
}
}

#endif

namespace MNN {
namespace QNN {

#ifdef ENABLE_QNN_ONLINE_FINALIZE
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
// The kernel context is an existing generic attribute carrier, not a CPU
// backend. All allocation, execution and error propagation stay on QNN.
class QnnOfflineExecution final : public Execution {
public:
    QnnOfflineExecution(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs, std::shared_ptr<QnnBackendOptions> options,
                        QnnContext* context, std::shared_ptr<void> backendContextOwner)
        : Execution(backend), mOptions(std::move(options)), mKernel(context, std::move(backendContextOwner)),
          mContext("QNN", backend, inputs, outputs, mOptions->modelDirectory) {
        const auto* plugin = op->main_as_Plugin();
        if (!plugin || !plugin->attr() || !plugin->type() || plugin->type()->str() != "QNN") return;
        for (const auto* attr : *plugin->attr()) {
            if (!attr || !attr->key()) return;
            mContext.setAttr(attr->key()->str(), attr);
        }
        mInitialized = mKernel.init(&mContext);
        if (!mInitialized) mOptions->report(MNN_QNN_STAGE_RESIZE, INVALID_VALUE, "QNN offline context initialization failed");
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mContext.reset(inputs, outputs);
        const auto code = mInitialized && mKernel.resize(&mContext) ? NO_ERROR : INVALID_VALUE;
        mOptions->report(MNN_QNN_STAGE_RESIZE, code, code == NO_ERROR ?
            "QNN offline I/O ready" : "QNN offline resize failed");
        return code;
    }
    ErrorCode onExecute(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override {
        const auto staging = static_cast<QnnBackend*>(backend())->runGraphOnce();
        if (staging != NO_ERROR) return staging;
        const auto code = mInitialized && mKernel.compute(&mContext) ? NO_ERROR : INVALID_VALUE;
        mOptions->report(MNN_QNN_STAGE_EXECUTE, code, code == NO_ERROR ?
            "QNN offline execution completed" : "QNN offline execution failed");
        return code;
    }
private:
    std::shared_ptr<QnnBackendOptions> mOptions;
    plugin::backend::QNNPluginExecuteRaw mKernel;
    plugin::CPUKernelContext mContext;
    bool mInitialized = false;
};
#endif

class QnnHostMemObj final : public Backend::MemObj {
public:
    QnnHostMemObj(std::shared_ptr<BufferAllocator> allocator, MemChunk chunk)
        : mAllocator(std::move(allocator)), mChunk(chunk) {
    }
    ~QnnHostMemObj() override {
        if (mAllocator && mChunk.ptr() != nullptr) {
            mAllocator->free(mChunk);
        }
    }
    MemChunk chunk() override {
        return mChunk;
    }
private:
    std::shared_ptr<BufferAllocator> mAllocator;
    MemChunk mChunk;
};

QnnBackend::QnnBackend(const QnnRuntime* runtime) : Backend(runtime->mInfo.type), mPower(runtime->mPower) {
    mRuntime = runtime;
    if (runtime->mQnnOfflineContextModel) {
        mOfflineHostAllocator.reset(new EagerBufferAllocator(BufferAllocator::Allocator::createDefault()));
    }
    mDumpIntermediateOutputs = runtime->mDumpIntermediateOutputs;
    if (mDumpIntermediateOutputs) {
        mTensorDumper.reset(new QNNTensorDumper(true));
    }
    mUseHtpBackend = !isDedicatedQnnSession() || runtime->mBackendKind == QNN::QnnBackendKind::Htp;
#ifdef ENABLE_QNN_CONVERT_MODE
    // Convert mode runs on an x86 host with no QNN library loaded, but the
    // produced offline context is consumed by the on-device HTP runtime.
    // Force HTP graph semantics so the emitted graph matches what the device
    // runtime builds; the DSP-specific decompositions (e.g. the LayerNorm
    // MatMul mean expansion) are rejected by HTP v68/v69.
    if (isDedicatedQnnSession() && std::getenv("MNN_QNN_CONVERT_FORCE_HTP") != nullptr) {
        mUseHtpBackend = true;
    }
#endif
    mRequireInt8Graph = runtime->mRequireInt8Graph;
    mUseDirectInt8NhwcIo = runtime->mUseDirectInt8NhwcIo;
    mUseFP16 = mUseHtpBackend && !mRequireInt8Graph &&
               runtime->mPrecision != BackendConfig::Precision_High;
    if (mUseHtpBackend) {
        mPerf = QNNPerf::create(&mRuntime->mQnnInterface);
        if (mPower == BackendConfig::Power_High && mPerf) {
            mPerf->setPowerConfigBurst();
            mPerf->setRpcLatencyAndPolling();
        }

        if (!mRequireInt8Graph) {
            // QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION is an HTP-only extension.
            mQnnHtpGraphCustomConfig.option =
                QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
            mQnnHtpGraphCustomConfig.precision = QNN_PRECISION_FLOAT16;
            mQnnGraphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            mQnnGraphConfig.customConfig = &mQnnHtpGraphCustomConfig;
        }
    } else {
#ifdef MNN_QNN_DSP_RUNTIME
        mDspPerf = QNNDspPerf::create(&mRuntime->mQnnInterface);
        if (mDspPerf) {
            mDspPerf->setPowerConfigBurst();
        }

        mQnnDspEncodingCustomConfig.option =
            QNN_DSP_GRAPH_CONFIG_OPTION_ENCODING;
        mQnnDspEncodingCustomConfig.encoding =
            QNN_DSP_GRAPH_ENCODING_STATIC;
        mQnnDspEncodingGraphConfig.option =
            QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        mQnnDspEncodingGraphConfig.customConfig =
            &mQnnDspEncodingCustomConfig;

        mQnnDspPriorityCustomConfig.option =
            QNN_DSP_GRAPH_CONFIG_OPTION_PRIORITY;
        mQnnDspPriorityCustomConfig.priority = QNN_PRIORITY_HIGH;
        mQnnDspPriorityGraphConfig.option =
            QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        mQnnDspPriorityGraphConfig.customConfig =
            &mQnnDspPriorityCustomConfig;

        MNN_PRINT(
            "MNN_QNN: DSP V66 static INT8 graph mode enabled with HIGH "
            "priority and DSP-native performance control; HTP FP16 "
            "extensions are disabled.\n");
#endif
    }
    if (mUseDirectInt8NhwcIo) {
        MNN_PRINT(
            "MNN_QNN_INT8_IO_AUDIT: enabled=1 backend=%s "
            "public_input=INT8_NHWC public_output=INT8_NHWC "
            "input_layout=QNN_TRANSPOSE "
            "output_dequantize=OPENGL_WARP_BLEND.\n",
            mUseHtpBackend ? "HTP" : "DSP_V66");
    }
}

QnnBackend::~QnnBackend() {
    clean();
    if (mPower == BackendConfig::Power_High && mPerf) {
        mPerf->setPowerConfigBalanced();
    }
}

static inline std::map<OpType, QnnBackend::Creator*>* getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, QnnBackend::Creator*>* ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, QnnBackend::Creator*>; });
    return ret;
}

Execution* QnnBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    MNN_ASSERT(op != nullptr);
    // Core synthesizes quantization boundaries after shape inference.
    // Repair their still-unresolved output before this backend allocates it.
    if (isDedicatedQnnSession() &&
        (op->type() == OpType_FloatToInt8 || op->type() == OpType_Int8ToFloat) &&
        inputs.size() == 1 && outputs.size() == 1 &&
        outputs[0]->dimensions() == 0 && inputs[0]->dimensions() > 0) {
        TensorUtils::copyShape(inputs[0], outputs[0], true);
        outputs[0]->buffer().type = op->type() == OpType_FloatToInt8 ?
            halide_type_of<int8_t>() : halide_type_of<float>();
    }
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
    if (mRuntime->mQnnOfflineContextModel) {
        if (op->type() == OpType_Plugin) {
            return new QnnOfflineExecution(this, op, inputs, outputs, mRuntime->mQnnOptions,
                                           mRuntime->mSelectedContext, mRuntime->mBackendContextOwner);
        }
        return new QnnRejectedExecution(this, NOT_SUPPORT);
    }
#endif
    auto map = getCreatorMap();
    auto iter = map->find(op->type());

    // MNN_PRINT("MNN_QNN::onCreate Type %d, Name %s.\n", op->type(), op->name()->c_str());

    if (iter == map->end()) {
        if (isDedicatedQnnSession()) {
            mCpuFallbackDetected = true;
        }
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Not registered type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Not registered type %d.\n", op->type());
        }
        return isDedicatedQnnSession() ? new QnnRejectedExecution(this, NOT_SUPPORT) : nullptr;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    if (nullptr == exe) {
        if (isDedicatedQnnSession()) {
            mCpuFallbackDetected = true;
        }
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Don't support type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Don't support type %d.\n", op->type());
        }
        return isDedicatedQnnSession() ? new QnnRejectedExecution(this, NOT_SUPPORT) : nullptr;
    }

    return exe;
}

bool QnnBackend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("MNN_QNN: %d type has be added.\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}


void QnnBackend::onExecuteBegin() const {
    mGraphExecuted = false;
    mLastQnnError = QNN_SUCCESS;
    if (mResetStatusOnNextExecute) mExecutionStatus = NO_ERROR;
    mResetStatusOnNextExecute = false;
    if (mTensorCounter == 0) {
        // No op runs on this online backend; skip the per-frame power churn.
        return;
    }
    if (mPower == BackendConfig::Power_Normal && mPerf) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    return;
}

void QnnBackend::startProfile() const{
    MNN::QNN::doProfile(mRuntime->mQnnInterface, mQnnProfileHandle);
}
const Runtime* QnnBackend::getRuntime() {
    return mRuntime;
}

ErrorCode QnnBackend::runGraphOnce() const {
    if (!mGraphExecuted) {
        mGraphExecuted = true;
        if (mExecutionStatus == NO_ERROR) executeGraph();
        if (mRuntime->mQnnOptions) mRuntime->mQnnOptions->report(
            MNN_QNN_STAGE_EXECUTE, mExecutionStatus, mExecutionStatus == NO_ERROR ?
            "QNN execution completed" : "QNN execution or input copy failed", mLastQnnError);
    }
    return mExecutionStatus;
}

void QnnBackend::onExecuteEnd() const {
    mResetStatusOnNextExecute = true;
    if (mTensorCounter == 0) {
        return;
    }
    if (!isDedicatedQnnSession()) {
        // Legacy QNN executes after all of its per-op callbacks have completed.
        runGraphOnce();
    }
    if (mPower == BackendConfig::Power_Normal && mPerf) {
        mPerf->setPowerConfigBalanced();
    }
    startProfile();
    return;
}

void QnnBackend::onResizeBegin() {
    clean();
    mLastQnnError = QNN_SUCCESS;
    mExecutionStatus = NO_ERROR;
#ifdef ENABLE_QNN_CONVERT_MODE
    // Convert mode records tensors during resize (onAcquire ->
    // QnnConvertorTensor_CreateGraphTensor), so the graph must exist before
    // any op schedules; create it eagerly instead of lazily.
    ensureContextAndGraph();
    return;
#endif
    if (!isDedicatedQnnSession()) {
        ensureContextAndGraph();
        return;
    }
    // The online context/graph is created lazily by ensureContextAndGraph()
    // when the first op actually schedules onto this backend. Models handled
    // by the offline plugin path never trigger it, which keeps QNN from
    // loading libQnnHtpPrepare.so (~39 MB PSS) for online compilation.
    return;
}

ErrorCode QnnBackend::onResizeEnd() {
    #ifdef QNN_VERBOSE
    MNN_PRINT("start finalize\n");
    #endif
    ErrorCode result = NO_ERROR;
    if (mCpuFallbackDetected) {
        MNN_ERROR("MNN_QNN: CPU fallback was requested by an unsupported op; "
                  "strict QNN session creation failed.\n");
        result = NOT_SUPPORT;
    } else if (mLastQnnError != QNN_SUCCESS) {
        result = INVALID_VALUE;
    } else {
        buildOutputCast();
        buildOutputDequant();
        result = finalizeGraph();
    }
    mGraphFinalized = result == NO_ERROR;
    if (mRuntime->mQnnOptions) {
        if (result == NO_ERROR) mRuntime->mQnnOptions->ready();
        else mRuntime->mQnnOptions->report(MNN_QNN_STAGE_RESIZE, result, "QNN graph finalization failed", mLastQnnError);
    }
    if (!isDedicatedQnnSession() && result != NO_ERROR) {
        result = NOT_SUPPORT;
    }
    for(auto func : mReleaseFunc){
        func();
    }
    mReleaseFunc.clear();
    #ifdef QNN_VERBOSE
    MNN_PRINT("end finalize\n");
    #endif
    return result;
}

Backend::MemObj* QnnBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    if (mRuntime->mQnnOfflineContextModel) {
        QnnHostShape shape;
        if (!qnnHostShape(tensor, shape) || !mOfflineHostAllocator) return nullptr;
        // Pipeline releases DYNAMIC MemObj instances after resize while their
        // addresses remain live for execution. Keep the physical allocation in
        // a backend-owned reuse pool instead of freeing it with the MemObj.
        auto chunk = mOfflineHostAllocator->alloc(shape.bytes, storageType == DYNAMIC_SEPERATE);
        auto host = chunk.ptr();
        if (host == nullptr) return nullptr;
        const_cast<Tensor*>(tensor)->buffer().host = host;
        TensorUtils::getDescribeOrigin(tensor)->offset = 0;
        return new QnnHostMemObj(mOfflineHostAllocator, chunk);
    }
    // onAcquire() runs before the first Execution adds its node.  The online
    // graph is created lazily so offline Plugin models do not load QNN prepare
    // libraries, but an online graph tensor must never be created against a
    // null graph handle.
    ensureContextAndGraph();
    if (mLastQnnError != QNN_SUCCESS || mQnnGraphHandle == nullptr) {
        return nullptr;
    }

    std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);
    if (TensorUtils::getDescribe(tensor)->index >= 0) {
        tName = std::string("t") + std::to_string(TensorUtils::getDescribe(tensor)->index);
    }

    bool isInput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    MNN_ASSERT(!isConst);

    Qnn_TensorType_t tType = QNN_TENSOR_TYPE_NATIVE;
    if (isInput) {
        tType = QNN_TENSOR_TYPE_APP_WRITE;
    }
    if (isOutput) {
        tType = QNN_TENSOR_TYPE_APP_READ;
    }

    Qnn_DataType_t tDataType;
    Qnn_QuantizeParams_t tQuantizeParams{};
    tQuantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    Qnn_ScaleOffset_t tScaleOffsetEncoding;
    tScaleOffsetEncoding.scale = 0.0f;
    tScaleOffsetEncoding.offset = 0;
    auto quant = TensorUtils::getDescribe(tensor)->quantAttr.get();
    // MNN PTQ keeps float boundaries around quantized operators and toggles
    // applyQuant only for the tensors currently entering a quantized kernel.
    // QNN DSP V66 cannot consume those float boundary tensors. The calibration
    // metadata nevertheless contains a scale for every calibrated activation,
    // so keep the whole DSP graph in fixed point whenever that metadata exists.
    bool isQuant =
        quant != nullptr &&
        (TensorUtils::getDescribe(tensor)->applyQuant || !mUseHtpBackend ||
         mRequireInt8Graph);
    const bool useDirectInt8NhwcIo =
        mUseDirectInt8NhwcIo && isQuant &&
        (isInput || isOutput) && tensor->dimensions() == 4 &&
        quant->type == DataType_DT_INT8 &&
        TensorUtils::getDescribe(tensor)->dimensionFormat ==
            MNN_DATA_FORMAT_NCHW;
    //MNN_ASSERT((tensor->getType().code == halide_type_float) || (tensor->getType().code == halide_type_int && tensor->getType().bits == 32));
    if (mUseFP16 && tensor->getType().code == halide_type_float) {
        tType = QNN_TENSOR_TYPE_NATIVE;
        tDataType = QNN_DATATYPE_FLOAT_16;
    } else if (tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_32;
    } else if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
        tDataType = QNN_DATATYPE_INT_32;
    } else if (isDedicatedQnnSession() && isQuant && quant->type == DataType_DT_INT8 && tensor->getType().code == halide_type_int &&
               tensor->getType().bits == 8) {
        // Dynamic FloatToInt8 boundaries carry a real int8 MNN tensor. Accept
        // it before applying the calibrated fixed-point QNN metadata below.
        tDataType = QNN_DATATYPE_SFIXED_POINT_8;
    } else {
        MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
        return nullptr;
    }
    if(isQuant) {
        if ((mUseHtpBackend && !mRequireInt8Graph) ||
            (!isInput && !isOutput)) {
            tType = QNN_TENSOR_TYPE_NATIVE;
        }
        auto quantType = TensorUtils::getDescribe(tensor)->quantAttr->type;
        if(quantType == DataType_DT_INT8){
            tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            if(quant->zero != 0){
                MNN_PRINT("MNN_QNN: Not supported asymmetric quant in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = 0;
            tDataType = QNN_DATATYPE_SFIXED_POINT_8;
            if (isOutput) {
                tType = QNN_TENSOR_TYPE_NATIVE;
            }
        }else if(quantType == DataType_DT_INT16){
            // uint16
            tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = quant->zero;
            tDataType = QNN_DATATYPE_UFIXED_POINT_16;
            if (isOutput) {
                tType = QNN_TENSOR_TYPE_NATIVE;
            }
        }
    }
    const bool useNhwcHostStaging =
        isDedicatedQnnSession() && !isQuant &&
        (isInput || isOutput) && tensor->dimensions() == 4 &&
        tensor->getType().code == halide_type_float &&
        TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NCHW;
    if (useNhwcHostStaging) {
        // Keep the model-facing tensor NCHW and native. The application-facing
        // tensor is created below as NHWC and connected with QNN graph nodes.
        tType = QNN_TENSOR_TYPE_NATIVE;
        MNN_PRINT("MNN_QNN: NHWC host %s staging is fused into the QNN "
                  "graph for %s [%d,%d,%d,%d].\n",
                  isInput ? "input" : "output", tName.c_str(),
                  tensor->length(0), tensor->length(1),
                  tensor->length(2), tensor->length(3));
    }
    if (useDirectInt8NhwcIo) {
        // The model-facing quantized tensor remains NCHW. A signed INT8 NHWC
        // application tensor and an all-fixed-point Transpose node are added
        // below, keeping layout conversion inside the QNN graph.
        tType = QNN_TENSOR_TYPE_NATIVE;
    }
    bool isDebugTensor = tType == QNN_TENSOR_TYPE_NATIVE && canDumpTensor(tDataType, tName);
    if (isDebugTensor) {
        tType = QNN_TENSOR_TYPE_APP_READ;
    }
    tQuantizeParams.scaleOffsetEncoding = tScaleOffsetEncoding;
    Tensor::DimensionType tensorDimType = tensor->getDimensionType();

    std::vector<int> tDims = tensor->shape();
    if(TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4){
        tensorDimType = gQnnTensorDimType;
        std::unique_ptr<Tensor> tempTensor(new Tensor(tensor, tensorDimType, false));
        if (!(tempTensor->shape().empty())) {
            tDims = tempTensor->shape();
        } else {
            tDims = {1};
        }
    }
    if ((!mUseHtpBackend || mRequireInt8Graph) && tDims.empty()) {
        // MNN represents a scalar with rank 0. QNN DSP tensors require rank
        // >= 1; strict HTP fixed-point tensors use the same explicit [1]
        // representation. Element count and scalar broadcast semantics are
        // unchanged.
        tDims = {1};
    }

    std::string suffix = "";
    if (useDirectInt8NhwcIo) {
        suffix = "_model_nchw";
    }
    if(isInput && mUseFP16 && tensor->getType().code == halide_type_float){
        suffix = "_cast";
    }
    if(isOutput && isQuant && !useDirectInt8NhwcIo){
        // Convert mode runs on x86 without a loaded HTP library, so the
        // model-facing quantized output tensor must still get a distinct
        // name; otherwise it collides with the FP32 APP_READ host tensor
        // in the generated model cpp. The device HTP int8 runtime already
        // uses this exact naming, so offline ctx names stay consistent.
        suffix = "_dequant";
    }
    if(isOutput && mUseFP16 && tensor->getType().code == halide_type_float){
        suffix = "_cast";
    }
    #ifdef QNN_VERBOSE
    if (!mUseHtpBackend) {
        MNN_PRINT(
            "MNN_QNN_DSP: acquire name=%s usage=%d type=%d dtype=0x%x "
            "rank=%zu shape=",
            (tName + suffix).c_str(),
            static_cast<int>(TensorUtils::getDescribe(tensor)->usage),
            static_cast<int>(tType),
            static_cast<unsigned int>(tDataType), tDims.size());
        for (const int dimension : tDims) {
            MNN_PRINT("%d,", dimension);
        }
        MNN_PRINT("\n");
    }
    #endif
    std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper = QNNTensorWrapper::create(tName + suffix, tType, tDataType, tDims, tQuantizeParams);

    Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
    if (isDebugTensor && !prepareDebugTensor(qnnTensorWrapper, tensorDimType)) {
        QNN_TENSOR_SET_TYPE(*qnnTensor, QNN_TENSOR_TYPE_NATIVE);
        isDebugTensor = false;
    }
    if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                          mQnnGraphHandle, qnnTensor),
                      "tensorCreateGraphTensor")) {
        return nullptr;
    }
    if (isDebugTensor && !registerDebugTensor(qnnTensorWrapper)) {
        MNN_ERROR("MNN_QNN: Intermediate tensor %s will not be dumped.\n", tName.c_str());
    }
    mQNNTensorWrappers.push_back(qnnTensorWrapper);
    mTensorMap.insert({TensorUtils::getDescribe(tensor), mTensorCounter});

    if (isInput) {
        if (useDirectInt8NhwcIo) {
            mTensorCounter += 1;
            const std::vector<int> nhwcDims = {
                tensor->length(0), tensor->length(2),
                tensor->length(3), tensor->length(1)
            };
            std::shared_ptr<Tensor> stageTensor(
                Tensor::create<int8_t>(nhwcDims, nullptr,
                                       Tensor::TENSORFLOW));
            auto hostTensorWrapper = QNNTensorWrapper::create(
                tName, QNN_TENSOR_TYPE_APP_WRITE,
                QNN_DATATYPE_SFIXED_POINT_8, nhwcDims, tQuantizeParams);
            if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                  mQnnGraphHandle,
                                  hostTensorWrapper->getNativeTensor()),
                              "tensorCreateGraphTensor")) {
                return nullptr;
            }
            mInputCastTensorMap.insert(
                {TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mInputNhwcTensorMap.insert(
                {TensorUtils::getDescribe(tensor), nullptr});
            mQNNTensorWrappers.push_back(hostTensorWrapper);
            mTensorMap.insert(
                {TensorUtils::getDescribe(
                     const_cast<const Tensor*>(stageTensor.get())),
                 mTensorCounter});
            mInputTensorIndexes.push_back(mTensorCounter);
            hostTensorWrapper->alloc(Tensor::TENSORFLOW);
            buildInputCast(tensor);
        } else if (isQuant && (!mUseHtpBackend || mRequireInt8Graph)) {
            // DSP keeps calibrated model inputs fixed point and quantizes the
            // application's FP32 host tensor in inputIO(). This tensor is an
            // APP_WRITE graph input; registering it as an output leaves
            // graphExecute with numInputs == 0.
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnTensorWrapper->alloc(tensorDimType);
        } else if (useNhwcHostStaging) {
            mTensorCounter += 1;
            const std::vector<int> nhwcDims = {
                tensor->length(0), tensor->length(2),
                tensor->length(3), tensor->length(1)
            };
            std::shared_ptr<Tensor> stageTensor(
                Tensor::create<float>(nhwcDims, nullptr, Tensor::TENSORFLOW));
            Qnn_QuantizeParams_t stageQuantize = QNN_QUANTIZE_PARAMS_INIT;
            auto hostTensorWrapper = QNNTensorWrapper::create(
                tName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                nhwcDims, stageQuantize);
            if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                  mQnnGraphHandle,
                                  hostTensorWrapper->getNativeTensor()),
                              "tensorCreateGraphTensor")) {
                return nullptr;
            }
            mInputCastTensorMap.insert(
                {TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(hostTensorWrapper);
            mTensorMap.insert(
                {TensorUtils::getDescribe(
                     const_cast<const Tensor*>(stageTensor.get())),
                 mTensorCounter});
            mInputTensorIndexes.push_back(mTensorCounter);
            hostTensorWrapper->alloc(Tensor::TENSORFLOW);

            std::shared_ptr<QNNTensorWrapper> nhwcNativeTensor;
            if (mUseFP16) {
                mTensorCounter += 1;
                nhwcNativeTensor = QNNTensorWrapper::create(
                    tName + "_nhwc_fp16", QNN_TENSOR_TYPE_NATIVE,
                    QNN_DATATYPE_FLOAT_16, nhwcDims, stageQuantize);
                if (!checkQnnCall(
                        mRuntime->mQnnInterface.tensorCreateGraphTensor(
                            mQnnGraphHandle,
                            nhwcNativeTensor->getNativeTensor()),
                        "tensorCreateGraphTensor")) {
                    return nullptr;
                }
                mQNNTensorWrappers.push_back(nhwcNativeTensor);
            }
            mInputNhwcTensorMap.insert(
                {TensorUtils::getDescribe(tensor), nhwcNativeTensor});
            buildInputCast(tensor);
        } else if (mUseFP16 &&
                   tensor->getType().code == halide_type_float) {
            // Create a stage tensor for the legacy same-layout cast path.
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnCastTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, tDims, tQuantizeParamstmp);
            if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                  mQnnGraphHandle,
                                  qnnCastTensorWrapper->getNativeTensor()),
                              "tensorCreateGraphTensor")) {
                return nullptr;
            }
            mInputCastTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnCastTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnCastTensorWrapper->alloc(tensorDimType);
            buildInputCast(tensor);
        }else{
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnTensorWrapper->alloc(tensorDimType);
        }
    }
    if (isOutput) {
        if (useDirectInt8NhwcIo) {
            mTensorCounter += 1;
            const std::vector<int> nhwcDims = {
                tensor->length(0), tensor->length(2),
                tensor->length(3), tensor->length(1)
            };
            std::shared_ptr<Tensor> stageTensor(
                Tensor::create<int8_t>(nhwcDims, nullptr,
                                       Tensor::TENSORFLOW));
            auto hostTensorWrapper = QNNTensorWrapper::create(
                tName, QNN_TENSOR_TYPE_APP_READ,
                QNN_DATATYPE_SFIXED_POINT_8, nhwcDims, tQuantizeParams);
            if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                  mQnnGraphHandle,
                                  hostTensorWrapper->getNativeTensor()),
                              "tensorCreateGraphTensor")) {
                return nullptr;
            }
            mOutputCastTensorMap.insert(
                {TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mOutputNhwcTensorMap.insert(
                {TensorUtils::getDescribe(tensor), nullptr});
            mQNNTensorWrappers.push_back(hostTensorWrapper);
            mTensorMap.insert(
                {TensorUtils::getDescribe(
                     const_cast<const Tensor*>(stageTensor.get())),
                 mTensorCounter});
            mOutputTensorIndexes.push_back(mTensorCounter);
            hostTensorWrapper->alloc(Tensor::TENSORFLOW);
        } else if (useNhwcHostStaging) {
            mTensorCounter += 1;
            const std::vector<int> nhwcDims = {
                tensor->length(0), tensor->length(2),
                tensor->length(3), tensor->length(1)
            };
            std::shared_ptr<Tensor> stageTensor(
                Tensor::create<float>(nhwcDims, nullptr, Tensor::TENSORFLOW));
            Qnn_QuantizeParams_t stageQuantize = QNN_QUANTIZE_PARAMS_INIT;
            auto hostTensorWrapper = QNNTensorWrapper::create(
                tName, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                nhwcDims, stageQuantize);
            if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                  mQnnGraphHandle,
                                  hostTensorWrapper->getNativeTensor()),
                              "tensorCreateGraphTensor")) {
                return nullptr;
            }
            mOutputCastTensorMap.insert(
                {TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(hostTensorWrapper);
            mTensorMap.insert(
                {TensorUtils::getDescribe(
                     const_cast<const Tensor*>(stageTensor.get())),
                 mTensorCounter});
            mOutputTensorIndexes.push_back(mTensorCounter);
            hostTensorWrapper->alloc(Tensor::TENSORFLOW);

            std::shared_ptr<QNNTensorWrapper> nhwcNativeTensor;
            if (mUseFP16) {
                mTensorCounter += 1;
                nhwcNativeTensor = QNNTensorWrapper::create(
                    tName + "_nhwc_fp16", QNN_TENSOR_TYPE_NATIVE,
                    QNN_DATATYPE_FLOAT_16, nhwcDims, stageQuantize);
                if (!checkQnnCall(
                        mRuntime->mQnnInterface.tensorCreateGraphTensor(
                            mQnnGraphHandle,
                            nhwcNativeTensor->getNativeTensor()),
                        "tensorCreateGraphTensor")) {
                    return nullptr;
                }
                mQNNTensorWrappers.push_back(nhwcNativeTensor);
            }
            mOutputNhwcTensorMap.insert(
                {TensorUtils::getDescribe(tensor), nhwcNativeTensor});
        } else if(isQuant){
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
            if (tensor->getType().code == halide_type_float) {
                tDataType = QNN_DATATYPE_FLOAT_32;
            } else {
                MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnOutputTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_READ, tDataType, tDims, tQuantizeParamstmp);
            if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                  mQnnGraphHandle,
                                  qnnOutputTensorWrapper->getNativeTensor()),
                              "tensorCreateGraphTensor")) {
                return nullptr;
            }
            mDeQuantOutputTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnOutputTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mOutputTensorIndexes.push_back(mTensorCounter);
            qnnOutputTensorWrapper->alloc(tensorDimType);
        } else{
            if (mUseFP16 && tensor->getType().code == halide_type_float) {
                mTensorCounter += 1;
                std::shared_ptr<Tensor> stageTensor;
                stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
                Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
                std::shared_ptr<QNNTensorWrapper> qnnCastTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, tDims, tQuantizeParamstmp);
                if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                                      mQnnGraphHandle,
                                      qnnCastTensorWrapper->getNativeTensor()),
                                  "tensorCreateGraphTensor")) {
                    return nullptr;
                }
                mOutputCastTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
                mQNNTensorWrappers.push_back(qnnCastTensorWrapper);
                mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
                mOutputTensorIndexes.push_back(mTensorCounter);
                qnnCastTensorWrapper->alloc(tensorDimType);
            }else{
                mOutputTensorIndexes.push_back(mTensorCounter);
                qnnTensorWrapper->alloc(tensorDimType);
            }
        }
    }

    mTensorCounter += 1;
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif
    return new Backend::MemObj();
}


bool QnnBackend::onClearBuffer() {
    if (mOfflineHostAllocator) {
        mOfflineHostAllocator->release(false);
    }
    return true;
}


void QnnBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    QnnCopyDiagnostic diagnostic(mRuntime->mQnnOptions.get(), mExecutionStatus);
    if (mRuntime->mQnnOfflineContextModel) {
        if (TensorUtils::getDescribe(dstTensor)->usage == Tensor::InsideDescribe::INPUT && mResetStatusOnNextExecute) {
            mExecutionStatus = NO_ERROR;
            mResetStatusOnNextExecute = false;
        }
        const auto code = qnnConvertTensor(srcTensor, dstTensor);
        if (code != NO_ERROR) mExecutionStatus = code;
        if (mExecutionStatus != NO_ERROR) mRuntime->mQnnOptions->report(
            MNN_QNN_STAGE_COPY, mExecutionStatus, "QNN offline host copy failed");
        return;
    }
    bool isInput = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    // MNN_ASSERT(!isConst);

    if (isConst) {
        MNN_ASSERT(isInput);
    }

    MNN_ASSERT(isInput || isOutput);

    if (isInput) {
        if (mResetStatusOnNextExecute) {
            mExecutionStatus = NO_ERROR;
            mResetStatusOnNextExecute = false;
        }
        inputIO(srcTensor, dstTensor);
    } else if (isOutput) {
        outputIO(srcTensor, dstTensor);
    } else {
        // Not support.
    }
}

void QnnBackend::inputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto iter = mInputCastTensorMap.find(TensorUtils::getDescribe(dstTensor));
    int dstIndex = -1;
    if (iter != mInputCastTensorMap.end()) {
        dstIndex = getTensorIdx(iter->second.second.get());
    } else {
        dstIndex = getTensorIdx(dstTensor);
    }
    if (dstIndex < 0 || static_cast<size_t>(dstIndex) >= mQNNTensorWrappers.size()) {
        mExecutionStatus = INPUT_DATA_ERROR;
        return;
    }
    std::shared_ptr<QNNTensorWrapper> dstQnnTensorWrapper = mQNNTensorWrappers[dstIndex];
    std::shared_ptr<Tensor> dstDataContainer = dstQnnTensorWrapper->getDataContainer();
    const Qnn_Tensor_t* nativeTensor = dstQnnTensorWrapper->getNativeTensor();

    if (isDedicatedQnnSession() && nativeTensor->v1.dataType == QNN_DATATYPE_SFIXED_POINT_8) {
        const float scale = nativeTensor->v1.quantizeParams.scaleOffsetEncoding.scale;
        const int32_t offset = nativeTensor->v1.quantizeParams.scaleOffsetEncoding.offset;
        if (!(scale > 0.0f)) {
            MNN_ERROR("MNN_QNN: invalid DSP input quantization scale.\n");
            mExecutionStatus = INVALID_VALUE;
            return;
        }

        if (srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 8) {
            if (TensorUtils::getDescribe(srcTensor)->dimensionFormat !=
                    TensorUtils::getDescribe(dstDataContainer.get())->dimensionFormat ||
                srcTensor->elementSize() != dstDataContainer->elementSize() || srcTensor->host<int8_t>() == nullptr) {
                MNN_ERROR("MNN_QNN: direct INT8 input contract mismatch.\n");
                mExecutionStatus = INVALID_VALUE;
                return;
            }
            ::memcpy(dstDataContainer->host<int8_t>(), srcTensor->host<int8_t>(), srcTensor->elementSize());
            return;
        }

        QnnHostShape sourceShape, destinationShape;
        if (srcTensor->getType() != halide_type_of<float>() ||
            !qnnHostShape(srcTensor, sourceShape) || !qnnHostShape(dstDataContainer.get(), destinationShape) ||
            sourceShape.batch != destinationShape.batch || sourceShape.channel != destinationShape.channel ||
            sourceShape.area != destinationShape.area || !srcTensor->host<float>()) {
            mExecutionStatus = INPUT_DATA_ERROR;
            return;
        }
        // Quantize the application's FP32 tensor straight into the QNN
        // client buffer. The previous float staging tensor plus scalar
        // divide loop cost several milliseconds per frame on a mid cluster
        // core, which made the DSP full-inference time exceed the pure
        // graph-execute time by far more than on HTP.
        const float* input = nullptr;
        std::shared_ptr<Tensor> floatStage;
        if (TensorUtils::getDescribe(srcTensor)->dimensionFormat ==
            TensorUtils::getDescribe(dstDataContainer.get())->dimensionFormat) {
            input = srcTensor->host<float>();
        } else {
            floatStage = std::shared_ptr<Tensor>(
                Tensor::create<float>(dstDataContainer->shape(), nullptr, dstDataContainer->getDimensionType()),
                Tensor::destroy);
            if (qnnConvertTensor(srcTensor, floatStage.get()) != NO_ERROR) {
                MNN_ERROR("MNN_QNN: DSP input layout conversion failed.\n");
                mExecutionStatus = INVALID_VALUE;
                return;
            }
            input = floatStage->host<float>();
        }
        auto* output = dstDataContainer->host<int8_t>();
        const int count = dstDataContainer->elementSize();
        int index = 0;
#if defined(__aarch64__)
        // Bit-exact vectorization of the scalar reference below: FDIV keeps
        // the IEEE quotient identical to the scalar divide, FCVTAS matches
        // std::round's ties-away-from-zero, and the saturating narrows
        // reproduce the [-128, 127] clamp.
        const float32x4_t vScale = vdupq_n_f32(scale);
        const int32x4_t vOffset = vdupq_n_s32(offset);
        for (; index + 16 <= count; index += 16) {
            int32x4_t q0 = vsubq_s32(vcvtaq_s32_f32(vdivq_f32(vld1q_f32(input + index), vScale)), vOffset);
            int32x4_t q1 = vsubq_s32(vcvtaq_s32_f32(vdivq_f32(vld1q_f32(input + index + 4), vScale)), vOffset);
            int32x4_t q2 = vsubq_s32(vcvtaq_s32_f32(vdivq_f32(vld1q_f32(input + index + 8), vScale)), vOffset);
            int32x4_t q3 = vsubq_s32(vcvtaq_s32_f32(vdivq_f32(vld1q_f32(input + index + 12), vScale)), vOffset);
            const int16x8_t low = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            const int16x8_t high = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(output + index, vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)));
        }
#endif
        for (; index < count; ++index) {
            const int value = static_cast<int>(std::round(input[index] / scale)) - offset;
            output[index] = static_cast<int8_t>(std::max(-128, std::min(127, value)));
        }
        return;
    }

    const auto code = qnnConvertTensor(srcTensor, dstDataContainer.get());
    if (code != NO_ERROR) {
        mExecutionStatus = code;
        MNN_ERROR("MNN_QNN: input layout/type/shape mismatch.\n");
    }
}

void QnnBackend::outputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto iter = mDeQuantOutputTensorMap.find(TensorUtils::getDescribe(srcTensor));
    int srcIndex = -1;
    if (iter != mDeQuantOutputTensorMap.end()) {
        srcIndex = getTensorIdx(iter->second.second.get());
    } else {
        auto castIter = mOutputCastTensorMap.find(TensorUtils::getDescribe(srcTensor));
        if (castIter != mOutputCastTensorMap.end()) {
            srcIndex = getTensorIdx(castIter->second.second.get());
        } else if (mUseFP16) {
            mExecutionStatus = INVALID_VALUE;
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer for cast float to half.\n");
            return;
        } else {
            srcIndex = getTensorIdx(srcTensor);
        }
    }
    if (srcIndex < 0 || static_cast<size_t>(srcIndex) >= mQNNTensorWrappers.size()) {
        mExecutionStatus = INPUT_DATA_ERROR;
        return;
    }
    std::shared_ptr<QNNTensorWrapper> srcQnnTensorWrapper = mQNNTensorWrappers[srcIndex];
    std::shared_ptr<Tensor> srcDataContainer = srcQnnTensorWrapper->getDataContainer();
    const Qnn_Tensor_t* nativeTensor = srcQnnTensorWrapper->getNativeTensor();

    if (isDedicatedQnnSession() && nativeTensor->v1.dataType == QNN_DATATYPE_SFIXED_POINT_8) {
        const float scale = nativeTensor->v1.quantizeParams.scaleOffsetEncoding.scale;
        const int32_t offset = nativeTensor->v1.quantizeParams.scaleOffsetEncoding.offset;
        if (!(scale > 0.0f)) {
            MNN_ERROR("MNN_QNN: invalid DSP output quantization scale.\n");
            mExecutionStatus = INVALID_VALUE;
            return;
        }
        if (dstTensor->getType().code == halide_type_int && dstTensor->getType().bits == 8) {
            if (TensorUtils::getDescribe(dstTensor)->dimensionFormat !=
                    TensorUtils::getDescribe(srcDataContainer.get())->dimensionFormat ||
                dstTensor->elementSize() != srcDataContainer->elementSize() || dstTensor->host<int8_t>() == nullptr) {
                MNN_ERROR("MNN_QNN: direct INT8 output contract mismatch.\n");
                mExecutionStatus = INVALID_VALUE;
                return;
            }
            ::memcpy(dstTensor->host<int8_t>(), srcDataContainer->host<int8_t>(), dstTensor->elementSize());
            return;
        }
        auto floatStage = std::shared_ptr<Tensor>(
            Tensor::create<float>(srcDataContainer->shape(), nullptr, srcDataContainer->getDimensionType()),
            Tensor::destroy);
        const auto* input = srcDataContainer->host<int8_t>();
        auto* output = floatStage->host<float>();
        for (int index = 0; index < floatStage->elementSize(); ++index) {
            output[index] = scale * (static_cast<int32_t>(input[index]) + offset);
        }
        const auto code = qnnConvertTensor(floatStage.get(), dstTensor);
        if (code != NO_ERROR) mExecutionStatus = code;
        return;
    }

    const auto code = qnnConvertTensor(srcDataContainer.get(), dstTensor);
    if (code != NO_ERROR) {
        mExecutionStatus = code;
        MNN_ERROR("MNN_QNN: output layout/type/shape mismatch.\n");
    }
}
bool QnnBackend::useCache() const {
    return mRuntime->mUseCache;
}

bool QnnBackend::checkQnnCall(Qnn_ErrorHandle_t result, const char* call) const {
    const int errorCode = static_cast<int>(result & 0xFFFF);
    if (errorCode == QNN_SUCCESS) {
        return true;
    }
    if (mLastQnnError == QNN_SUCCESS) {
        mLastQnnError = errorCode;
    }
    mExecutionStatus = INVALID_VALUE;
    MNN_ERROR("MNN_QNN: %s failed with QNN error %d.\n", call, errorCode);
    return false;
}

void QnnBackend::ensureContextAndGraph() {
    if (mOnlineGraphReady) {
        return;
    }
    mOnlineGraphReady = true;
    createContextAndGraph();
}

void QnnBackend::createContextAndGraph() {
    if (!checkQnnCall(mRuntime->mQnnInterface.contextCreate(mRuntime->mQnnBackendHandle, mRuntime->mQnnDeviceHandle,
                         mQnnContextConfig, &mQnnContextHandle), "contextCreate")) {
        return;
    }
    if (mQnnContextHandle == nullptr) {
        mLastQnnError = QNN_COMMON_ERROR_GENERAL;
        mExecutionStatus = INVALID_VALUE;
        MNN_ERROR("MNN_QNN: contextCreate returned a null context.\n");
        return;
    }
    const QnnGraph_Config_t* htpGraphConfigs[] = {
        &mQnnGraphConfig, nullptr};
#ifdef MNN_QNN_DSP_RUNTIME
    const QnnGraph_Config_t* dspGraphConfigs[] = {
        &mQnnDspEncodingGraphConfig, &mQnnDspPriorityGraphConfig, nullptr};
    const QnnGraph_Config_t** graphConfigs =
        mUseHtpBackend
            ? (mRequireInt8Graph ? nullptr : htpGraphConfigs)
            : dspGraphConfigs;
#else
    const QnnGraph_Config_t** graphConfigs =
        mUseHtpBackend && !mRequireInt8Graph ? htpGraphConfigs : nullptr;
#endif
    if (mRuntime->mUseCache) {
        checkQnnCall(mRuntime->mQnnInterface.graphRetrieve(
                         mQnnContextHandle, mQnnGraphName.c_str(),
                         &mQnnGraphHandle),
                     "graphRetrieve");
    } else {
        checkQnnCall(mRuntime->mQnnInterface.graphCreate(
                         mQnnContextHandle, mQnnGraphName.c_str(),
                         graphConfigs, &mQnnGraphHandle),
                     "graphCreate");
    }
    if (mQnnGraphHandle == nullptr && mLastQnnError == QNN_SUCCESS) {
        mLastQnnError = QNN_COMMON_ERROR_GENERAL;
        mExecutionStatus = INVALID_VALUE;
        MNN_ERROR("MNN_QNN: graph creation returned a null graph.\n");
    }
}

ErrorCode QnnBackend::finalizeGraph() {
    // [TODO] Fix this. Add the following branch for empty resize.
    if (mTensorCounter == 0) {
        return NO_ERROR;
    }
    if (mLastQnnError != QNN_SUCCESS || mQnnGraphHandle == nullptr) {
        return INVALID_VALUE;
    }
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif

    // Create Prefile Handle
    MNN::QNN::createProfileHandle(mRuntime->mQnnInterface, mRuntime->mQnnBackendHandle, &mQnnProfileHandle);

    if (!checkQnnCall(mRuntime->mQnnInterface.graphFinalize(
                          mQnnGraphHandle, mQnnProfileHandle,
                          mQnnSignalHandle),
                      "graphFinalize")) {
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

void QnnBackend::executeGraph() const {
    if (mTensorCounter == 0) {
        // No op was scheduled onto this online graph (the model runs through
        // the offline plugin path). Executing an empty, unfinalized graph
        // makes the DSP reject it with "Invalid input / output parameters",
        // so skip the call entirely.
        return;
    }
    if (!mGraphFinalized) {
        mExecutionStatus = NO_EXECUTION;
        MNN_ERROR("MNN_QNN: graphExecute skipped because graph finalization did not succeed.\n");
        return;
    }
    if (mQnnGraphHandle == nullptr) {
        mLastQnnError = QNN_COMMON_ERROR_GENERAL;
        mExecutionStatus = INVALID_VALUE;
        MNN_ERROR("MNN_QNN: graphExecute called with a null graph.\n");
        return;
    }
    const size_t executeOutputCount = mOutputTensorIndexes.size() + mDebugTensorWrappers.size();
    if (mExecuteInputs.size() != mInputTensorIndexes.size() ||
        mExecuteOutputs.size() != executeOutputCount) {
        mExecuteInputs.clear();
        mExecuteOutputs.clear();
        mExecuteInputs.reserve(mInputTensorIndexes.size());
        mExecuteOutputs.reserve(mOutputTensorIndexes.size());
        for (const int tensorIndex : mInputTensorIndexes) {
            mExecuteInputs.push_back(*(
                mQNNTensorWrappers[tensorIndex]->getNativeTensor()));
        }
        for (const int tensorIndex : mOutputTensorIndexes) {
            mExecuteOutputs.push_back(*(
                mQNNTensorWrappers[tensorIndex]->getNativeTensor()));
        }
        for (const auto& tensor : mDebugTensorWrappers) {
            mExecuteOutputs.push_back(*tensor->getNativeTensor());
        }
    }

    // Ensure all output tensors have memory allocated; allocate temp buffers for those without.
    auto tempBuffers = ensureOutputTensorsMemory(
        mExecuteOutputs.data(), (uint32_t)mExecuteOutputs.size());

    const bool executed = checkQnnCall(mRuntime->mQnnInterface.graphExecute(
                                           mQnnGraphHandle, mExecuteInputs.data(),
                                           static_cast<uint32_t>(mExecuteInputs.size()),
                                           mExecuteOutputs.data(),
                                           static_cast<uint32_t>(mExecuteOutputs.size()),
                                           mQnnProfileHandle, mQnnSignalHandle),
                                       "graphExecute");
    if (executed && mTensorDumper != nullptr) {
        mTensorDumper->dump(mExecuteOutputs.data(), static_cast<uint32_t>(mExecuteOutputs.size()));
    }

    // Free temporarily allocated output tensor buffers.
    freeOutputTensorsTempMemory(mExecuteOutputs.data(), tempBuffers);
}

void QnnBackend::freeContextAndGraph() {
    if (mTensorCounter != 0) {
        mQnnGraphHandle = nullptr;
    }
    mExecuteInputs.clear();
    mExecuteOutputs.clear();
    if (mQnnContextHandle) {
        CALL_QNN(mRuntime->mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
    }
}

void QnnBackend::addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs) {
    ensureContextAndGraph();
    MNN_ASSERT(nodeName != nullptr && packageName != nullptr && nodeType != nullptr && !(inputs.empty()) && !(outputs.empty()));

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = version;
    opConfig.v1.name = nodeName;
    opConfig.v1.packageName = packageName;
    opConfig.v1.typeName = nodeType;
    opConfig.v1.numOfParams = params.size();
    opConfig.v1.params = params.data();
    opConfig.v1.numOfInputs = inputs.size();
    opConfig.v1.inputTensors = inputs.data();
    opConfig.v1.numOfOutputs = outputs.size();
    opConfig.v1.outputTensors = outputs.data();

    #ifdef QNN_VERBOSE
    if (!mUseHtpBackend) {
        MNN_PRINT("MNN_QNN_DSP: validate node=%s op=%s inputs=%u outputs=%u "
                  "params=%u\n",
                  nodeName, nodeType, opConfig.v1.numOfInputs,
                  opConfig.v1.numOfOutputs, opConfig.v1.numOfParams);
    }
    #endif
    const auto validationResult =
        mRuntime->mQnnInterface.backendValidateOpConfig(
            mRuntime->mQnnBackendHandle, opConfig);
    if ((!mUseHtpBackend || mRequireInt8Graph) &&
        validationResult != QNN_SUCCESS) {
        MNN_ERROR(
            "MNN_QNN_INT8_NODE: validation failed backend=%s node=%s op=%s "
            "code=%u inputs=%u outputs=%u\n",
            mUseHtpBackend ? "HTP" : "DSP", nodeName, nodeType,
            static_cast<unsigned int>(validationResult),
            opConfig.v1.numOfInputs, opConfig.v1.numOfOutputs);
        for (uint32_t index = 0; index < opConfig.v1.numOfInputs; ++index) {
            const Qnn_Tensor_t& tensor = opConfig.v1.inputTensors[index];
            MNN_ERROR(
                "MNN_QNN_INT8_NODE: input[%u] name=%s type=%d dtype=%s\n",
                index,
                QNN_TENSOR_GET_NAME(tensor) != nullptr
                    ? QNN_TENSOR_GET_NAME(tensor)
                    : "<unnamed>",
                static_cast<int>(QNN_TENSOR_GET_TYPE(tensor)),
                qnnDataTypeName(QNN_TENSOR_GET_DATA_TYPE(tensor)));
        }
        for (uint32_t index = 0; index < opConfig.v1.numOfOutputs; ++index) {
            const Qnn_Tensor_t& tensor = opConfig.v1.outputTensors[index];
            MNN_ERROR(
                "MNN_QNN_INT8_NODE: output[%u] name=%s type=%d dtype=%s\n",
                index,
                QNN_TENSOR_GET_NAME(tensor) != nullptr
                    ? QNN_TENSOR_GET_NAME(tensor)
                    : "<unnamed>",
                static_cast<int>(QNN_TENSOR_GET_TYPE(tensor)),
                qnnDataTypeName(QNN_TENSOR_GET_DATA_TYPE(tensor)));
        }
    }
    if (isDedicatedQnnSession()) {
        if (!checkQnnCall(validationResult, "backendValidateOpConfig")) {
            return;
        }
    } else if (validationResult != QNN_SUCCESS) {
        MNN_PRINT("QNN validate failed for node '%s' type '%s', error: %lu\n",
                  nodeName, nodeType,
                  static_cast<unsigned long>(validationResult));
    }

    const auto addResult =
        mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig);
    if ((!mUseHtpBackend || mRequireInt8Graph) &&
        addResult != QNN_SUCCESS) {
        MNN_ERROR(
            "MNN_QNN_INT8_NODE: add failed backend=%s node=%s op=%s code=%u\n",
            mUseHtpBackend ? "HTP" : "DSP", nodeName, nodeType,
            static_cast<unsigned int>(addResult));
    }
    if (isDedicatedQnnSession()) {
        checkQnnCall(addResult, "graphAddNode");
    } else if (addResult != QNN_SUCCESS) {
        MNN_PRINT("QNN graphAddNode failed for node '%s' type '%s', error: %lu\n",
                  nodeName, nodeType, static_cast<unsigned long>(addResult));
    }
}

int QnnBackend::getTensorIdx(const Tensor * tensor) const {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    int idx = -1;
    if (iter == mTensorMap.end()) {
        std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);;
        if (TensorUtils::getDescribe(tensor)->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
            MNN_PRINT("Tensor usage is %d.\n", (int) TensorUtils::getDescribe(tensor)->usage);
        }
        #ifdef QNN_VERBOSE
        MNN_PRINT("qnn tenor usage:%d, dimension:%d\n", TensorUtils::getDescribe(tensor)->usage, tensor->dimensions());
        #endif
        MNN_ASSERT(TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        // MNN_ASSERT(tensor->dimensions() <= 2);
        std::vector<uint32_t> tDims = getNHWCShape(tensor);
        Qnn_DataType_t tDataType;
        std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper;
        if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
            tDataType = QNN_DATATYPE_INT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticTensor(tName, tDataType, tDims, tensor->host<int>());
        } else if (tensor->getType().code == halide_type_float) {
            tDataType = mUseFP16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticFloatTensor(tName, tDataType, tDims, tensor->host<float>());
        } else {
            MNN_ASSERT(false);
        }
        Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
        checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                         mQnnGraphHandle, qnnTensor),
                     "tensorCreateGraphTensor");
        mQNNTensorWrappers.push_back(qnnTensorWrapper);
        mTensorMap.insert({tensorKey, mTensorCounter});
        idx = mTensorCounter;
        mTensorCounter += 1;
    } else {
        idx = iter->second;
    }
    return idx;
}

void QnnBackend::addTensor(Qnn_Tensor_t * staticTensor) {
    ensureContextAndGraph();
    #ifdef QNN_VERBOSE
    if (!mUseHtpBackend) {
        const auto quant = QNN_TENSOR_GET_QUANT_PARAMS(*staticTensor);
        MNN_PRINT(
            "MNN_QNN_DSP: tensorCreate name=%s type=%d dtype=0x%x rank=%u "
            "quantDef=%d quantEnc=%d\n",
            QNN_TENSOR_GET_NAME(*staticTensor) != nullptr
                ? QNN_TENSOR_GET_NAME(*staticTensor)
                : "<unnamed>",
            static_cast<int>(QNN_TENSOR_GET_TYPE(*staticTensor)),
            static_cast<unsigned int>(QNN_TENSOR_GET_DATA_TYPE(*staticTensor)),
            QNN_TENSOR_GET_RANK(*staticTensor),
            static_cast<int>(quant.encodingDefinition),
            static_cast<int>(quant.quantizationEncoding));
    }
    #endif
    const auto result = mRuntime->mQnnInterface.tensorCreateGraphTensor(
        mQnnGraphHandle, staticTensor);
    if (!mUseHtpBackend && result != QNN_SUCCESS) {
        MNN_ERROR("MNN_QNN_DSP: tensorCreate failed name=%s code=%u\n",
                  QNN_TENSOR_GET_NAME(*staticTensor) != nullptr
                      ? QNN_TENSOR_GET_NAME(*staticTensor)
                      : "<unnamed>",
                  static_cast<unsigned int>(result));
    }
    checkQnnCall(result, "tensorCreateGraphTensor");
}

Qnn_Tensor_t * QnnBackend::getNativeTensor(const Tensor * tensor) {
    int idx = getTensorIdx(tensor);
    return mQNNTensorWrappers[idx]->getNativeTensor();
}

std::shared_ptr<QNNTensorWrapper> QnnBackend::getTensorWrapper(const Tensor * tensor) {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    MNN_ASSERT(iter != mTensorMap.end());
    return mQNNTensorWrappers[iter->second];
}
Qnn_Tensor_t* QnnBackend::getMaskTensor(int maxKVSize) {
    ensureContextAndGraph();
    if (mMaskTensor.get() == nullptr) {
        std::vector<int> dimensions = {1, 1, 1, maxKVSize};
        std::shared_ptr<QNNTensorWrapper> tensorWrapper = QNNTensorWrapper::create(extraIoPrefix() + "_mask", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, dimensions);
        mMaskTensor = tensorWrapper;
        if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                              mQnnGraphHandle, mMaskTensor->getNativeTensor()),
                          "tensorCreateGraphTensor")) {
            return nullptr;
        }
    }
    return mMaskTensor->getNativeTensor();
}

Qnn_Tensor_t* QnnBackend::addExtraInput(Tensor* tensor) {
    ensureContextAndGraph();
    auto qnntensor = QNNTensorWrapper::create("", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, tensor->shape());
    qnntensor->setName(extraIoPrefix()+"_i" + std::to_string(mExtraInputs.size()));
    qnntensor->getNativeTensor()->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    mExtraInputs.emplace_back(qnntensor);
    if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                          mQnnGraphHandle, qnntensor->getNativeTensor()),
                      "tensorCreateGraphTensor")) {
        return nullptr;
    }

    return qnntensor->getNativeTensor();
}
Qnn_Tensor_t* QnnBackend::addExtraOutput(Tensor* tensor) {
    ensureContextAndGraph();
    auto qnntensor = QNNTensorWrapper::create("", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, tensor->shape());
    qnntensor->setName(extraIoPrefix()+"_o" + std::to_string(mExtraOutputs.size()));
    qnntensor->getNativeTensor()->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    mExtraOutputs.emplace_back(qnntensor);
    if (!checkQnnCall(mRuntime->mQnnInterface.tensorCreateGraphTensor(
                          mQnnGraphHandle, qnntensor->getNativeTensor()),
                      "tensorCreateGraphTensor")) {
        return nullptr;
    }
    return qnntensor->getNativeTensor();
}

bool QnnBackend::getUseFP16() const {
    return mUseFP16;
}

bool QnnBackend::isTensorDumpEnabled() const {
    return mDumpIntermediateOutputs;
}

bool QnnBackend::canDumpTensor(Qnn_DataType_t dataType, const std::string& name) const {
    if (!mDumpIntermediateOutputs) {
        return false;
    }
    if (QNNTensorWrapper::supportsHostBufferDataType(dataType)) {
        return true;
    }
    MNN_ERROR("MNN_QNN: Skip intermediate dump for %s because data type %u has no host-buffer mapping.\n",
              name.c_str(), static_cast<unsigned int>(dataType));
    return false;
}

bool QnnBackend::prepareDebugTensor(const std::shared_ptr<QNNTensorWrapper>& tensor,
                                    Tensor::DimensionType dimType) {
    MNN_ASSERT(tensor != nullptr);
    MNN_ASSERT(QNN_TENSOR_GET_TYPE(*tensor->getNativeTensor()) == QNN_TENSOR_TYPE_APP_READ);
    if (tensor->alloc(dimType, false) != nullptr) {
        return true;
    }
    const char* name = QNN_TENSOR_GET_NAME(*tensor->getNativeTensor());
    MNN_ERROR("MNN_QNN: Failed to allocate intermediate dump buffer for %s.\n",
              name == nullptr ? "<unnamed>" : name);
    return false;
}

bool QnnBackend::registerDebugTensor(const std::shared_ptr<QNNTensorWrapper>& tensor) {
    MNN_ASSERT(tensor != nullptr);
    MNN_ASSERT(QNN_TENSOR_GET_TYPE(*tensor->getNativeTensor()) == QNN_TENSOR_TYPE_APP_READ);
    if (!tensor->bindHostBuffer()) {
        return false;
    }
    mDebugTensorWrappers.emplace_back(tensor);
    return true;
}

bool QnnBackend::isDspBackend() const {
    return isDedicatedQnnSession() && !mUseHtpBackend;
}

bool QnnBackend::isExplicitQnnSession() const {
    return mRuntime->mQnnOptions != nullptr &&
           mRuntime->mQnnOptions->explicitConfig;
}

bool QnnBackend::isDedicatedQnnSession() const {
    return mRuntime->mInfo.type == MNN_FORWARD_QNN;
}

bool QnnBackend::requiresQuantizedGraph() const {
    return isDedicatedQnnSession() && (!mUseHtpBackend || mRequireInt8Graph);
}

const std::string& QnnBackend::v66LayerNormOpPackageName() const {
    return mRuntime->mV66LayerNormOpPackageName;
}

void QnnBackend::clean() {
    if (mQnnProfileHandle) {
        mRuntime->mQnnInterface.profileFree(mQnnProfileHandle);
        mQnnProfileHandle = nullptr;
    }
    freeContextAndGraph(); // This function must be called first.
    mOnlineGraphReady = false;
    mGraphFinalized = false;
    mTensorCounter = 0;
    mQNNTensorWrappers.clear();
    mTensorMap.clear();
    mInputTensorIndexes.clear();
    mOutputTensorIndexes.clear();
    mDebugTensorWrappers.clear();
    mDeQuantOutputTensorMap.clear();
    mInputCastTensorMap.clear();
    mOutputCastTensorMap.clear();
    mInputNhwcTensorMap.clear();
    mOutputNhwcTensorMap.clear();
    mIoParamTensorWrappers.clear();
    mCpuFallbackDetected = false;
}
void QnnBackend::buildOutputDequant(){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mDeQuantOutputTensorMap){
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = "Dequantize_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));
        mInputs.push_back(*(getNativeTensor(iter.second.first))); // input
        mOutputs.push_back(*(getNativeTensor(iter.second.second.get()))); // output
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

void QnnBackend::buildOutputCast(){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mOutputCastTensorMap){
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        auto nhwcIter = mOutputNhwcTensorMap.find(iter.first);
        if (nhwcIter != mOutputNhwcTensorMap.end()) {
            const uint32_t permData[] = {0, 2, 3, 1};
            auto perm = QNNParamTensorWrapper::create(
                "perm",
                "OutputNhwcPerm_" +
                    std::to_string(getTensorIdx(iter.second.first)) + "_PARAM",
                QNN_DATATYPE_UINT_32, std::vector<uint32_t>{4});
            ::memcpy(perm->alloc(), permData, sizeof(permData));
            addTensor(perm->getNativeTensor());
            mIoParamTensorWrappers.push_back(perm);

            mNodeType = "Transpose";
            std::string transposeName =
                "TransposeNchwToNhwc_I_" +
                std::to_string(getTensorIdx(iter.second.first));
            mParams.push_back(*(perm->getNativeParam()));
            mInputs.push_back(*(getNativeTensor(iter.second.first)));
            if (nhwcIter->second) {
                mOutputs.push_back(*(nhwcIter->second->getNativeTensor()));
            } else {
                mOutputs.push_back(
                    *(getNativeTensor(iter.second.second.get())));
            }
            addNodeToGraph(mOpConfigVersion, transposeName.c_str(),
                           mPackageName.c_str(), mNodeType.c_str(), mParams,
                           mInputs, mOutputs);

            if (nhwcIter->second) {
                mNodeType = "Cast";
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string castName =
                    "CastNhwcOutput_O_" +
                    std::to_string(
                        getTensorIdx(iter.second.second.get()));
                mInputs.push_back(*(nhwcIter->second->getNativeTensor()));
                mOutputs.push_back(
                    *(getNativeTensor(iter.second.second.get())));
                addNodeToGraph(mOpConfigVersion, castName.c_str(),
                               mPackageName.c_str(), mNodeType.c_str(),
                               mParams, mInputs, mOutputs);
            }
        } else {
            mNodeType = "Cast";
            std::string name = "Cast_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));
            mInputs.push_back(*(getNativeTensor(iter.second.first))); // input
            mOutputs.push_back(*(getNativeTensor(iter.second.second.get()))); // output
            addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }
}

void QnnBackend::buildInputCast(const Tensor *tensor){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    mNodeType.clear();
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();
    mNodeType = "Cast";
    auto iter = mInputCastTensorMap.find(TensorUtils::getDescribe(tensor));
    if(iter != mInputCastTensorMap.end()){
        auto nhwcIter = mInputNhwcTensorMap.find(iter->first);
        if (nhwcIter != mInputNhwcTensorMap.end()) {
            Qnn_Tensor_t *transposeInput =
                getNativeTensor(iter->second.second.get());
            if (nhwcIter->second) {
                std::string castName =
                    "CastNhwcInput_I_" +
                    std::to_string(
                        getTensorIdx(iter->second.second.get()));
                mInputs.push_back(
                    *(getNativeTensor(iter->second.second.get())));
                mOutputs.push_back(*(nhwcIter->second->getNativeTensor()));
                addNodeToGraph(mOpConfigVersion, castName.c_str(),
                               mPackageName.c_str(), mNodeType.c_str(),
                               mParams, mInputs, mOutputs);
                transposeInput = nhwcIter->second->getNativeTensor();
            }

            const uint32_t permData[] = {0, 3, 1, 2};
            auto perm = QNNParamTensorWrapper::create(
                "perm",
                "InputNhwcPerm_" +
                    std::to_string(getTensorIdx(iter->second.first)) + "_PARAM",
                QNN_DATATYPE_UINT_32, std::vector<uint32_t>{4});
            ::memcpy(perm->alloc(), permData, sizeof(permData));
            addTensor(perm->getNativeTensor());
            mIoParamTensorWrappers.push_back(perm);

            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string transposeName =
                "TransposeNhwcToNchw_O_" +
                std::to_string(getTensorIdx(iter->second.first));
            mParams.push_back(*(perm->getNativeParam()));
            mInputs.push_back(*transposeInput);
            mOutputs.push_back(*(getNativeTensor(iter->second.first)));
            addNodeToGraph(mOpConfigVersion, transposeName.c_str(),
                           mPackageName.c_str(), mNodeType.c_str(), mParams,
                           mInputs, mOutputs);
        } else {
            std::string name = "Cast_I_" + std::to_string(getTensorIdx(iter->second.second.get())) + "_O_" + std::to_string(getTensorIdx(iter->second.first));
            mInputs.push_back(*(getNativeTensor(iter->second.second.get()))); // input
            mOutputs.push_back(*(getNativeTensor(iter->second.first))); // output
            addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }
}

QnnRuntime::QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle,
                       Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle,
                       const std::string& v66LayerNormOpPackageName, QnnContext* selectedContext,
                       std::shared_ptr<void> backendContextOwner) {
    // MNN_PRINT("QnnRuntime is constructing.\n");
    mInfo.type = info.type;
    mInfo.numThread = info.numThread;
    mInfo.mode = info.mode;
    mQnnOptions = copyQnnOptions(info);
    mInfo.user = &mQnnOptions->config;
    const bool tblive = info.type == MNN_FORWARD_QNN;
    mQnnOfflineContextModel = tblive && (mQnnOptions->flags & MNN_QNN_CONFIG_OFFLINE_CONTEXT) != 0;
    // Default setting
    mPower = BackendConfig::Power_Normal;
    mMemory = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    // User setting
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
        mDumpIntermediateOutputs = (mQnnOptions->flags & MNN_QNN_CONFIG_DUMP_OUTPUTS) != 0;
    }
    mQnnInterface = qnnInterface;
    mQnnLogHandle = qnnLogHandle;
    mQnnBackendHandle = qnnBackendHandle;
    mQnnDeviceHandle = qnnDeviceHandle;
    mSelectedContext = selectedContext;
    mBackendContextOwner = std::move(backendContextOwner);
    mBackendKind = QNN::getLoadedQNNBackend();
    mV66LayerNormOpPackageName = v66LayerNormOpPackageName;
    std::string manifestPath;
    bool directInt8NhwcIo = false;
    if (tblive && mQnnOptions != nullptr &&
        mQnnOptions->explicitConfig) {
        manifestPath = mQnnOptions->runtimeManifestPath;
        directInt8NhwcIo = mQnnOptions->directInt8NhwcIo;
    }
    bool convertRequireInt8Graph = false;
#ifdef ENABLE_QNN_CONVERT_MODE
    convertRequireInt8Graph =
        info.type == MNN_FORWARD_QNN &&
        environmentValue("MNN_QNN_CONVERT_REQUIRE_INT8_GRAPH") == "1";
#endif
    mRequireInt8Graph = tblive && (
        convertRequireInt8Graph ||
        (mBackendKind == QNN::QnnBackendKind::Htp && directInt8NhwcIo));
    mUseDirectInt8NhwcIo = tblive &&
        (mBackendKind != QNN::QnnBackendKind::Htp || mRequireInt8Graph) &&
        directInt8NhwcIo;
}

QnnRuntime::~QnnRuntime() {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
    }
}
bool QnnRuntime::onSetCache(const void* buffer, size_t size) {
    // TODO: Fix bug and complete
    return false;
    if (nullptr == buffer) {
        return false;
    }
    auto error = mQnnInterface.contextValidateBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size);
    if (QNN_SUCCESS != error) {
        MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
        return false;
    }
    freeContext();
    CALL_QNN(mQnnInterface.contextCreateFromBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, nullptr));
    mUseCache = true;
    return true;
}
Qnn_ErrorHandle_t QnnRuntime::allocContext() const {
    return mQnnInterface.contextCreate(mQnnBackendHandle, mQnnDeviceHandle,
                                       mQnnContextConfig, &mQnnContextHandle);
}
void QnnRuntime::freeContext() const {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
        mBinaryBuffer.clear();
    }
}

std::pair<const void*, size_t> QnnRuntime::onGetCache() {
    return std::make_pair(nullptr, 0);
    if (!mBinaryBuffer.empty()) {
        return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
    }
    if (nullptr == mQnnContextHandle) {
        return std::make_pair(nullptr, 0);
    }
    Qnn_ContextBinarySize_t size = 0;
    CALL_QNN(mQnnInterface.contextGetBinarySize(mQnnContextHandle, &size));
    FUNC_PRINT(size);
    if (0 == size) {
        return std::make_pair(nullptr, 0);
    }
    mBinaryBuffer.resize(size);
    Qnn_ContextBinarySize_t writesize = 0;
    CALL_QNN(mQnnInterface.contextGetBinary(mQnnContextHandle, mBinaryBuffer.data(), size, &writesize));
    return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
}

Backend* QnnRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new QnnBackend(this);
}

QnnRuntime* QnnRuntime::create(const Backend::Info& info) {
    static std::mutex runtimeCreateMutex;
    std::lock_guard<std::mutex> runtimeCreateLock(runtimeCreateMutex);
    const bool tblive = info.type == MNN_FORWARD_QNN;
    QnnBackendKind requestedBackend = tblive ? QnnBackendKind::None : QnnBackendKind::Htp;
    std::string manifestPath;
    std::string resourceDirectory = environmentValue("ADSP_LIBRARY_PATH");
    std::string runtimeLibraryDirectory;
    std::string opPackageInterfaceProvider;
    std::string opPackageName;
    const auto options = copyQnnOptions(info);
    if (options == nullptr) {
        MNN_ERROR("MNN_QNN: invalid BackendConfig::sharedContext configuration.\n");
        return nullptr;
    }
    options->report(MNN_QNN_STAGE_RUNTIME, NO_EXECUTION, "QNN runtime initialization failed");
    const auto* qnnOptions = options.get();
    const bool qnnOfflineContextModel = tblive && (options->flags & MNN_QNN_CONFIG_OFFLINE_CONTEXT) != 0;
    const bool explicitQnn = tblive && qnnOptions != nullptr && qnnOptions->explicitConfig;
    if (explicitQnn) {
        manifestPath = qnnOptions->runtimeManifestPath;
        resourceDirectory = qnnOptions->acceleratorResourceDirectory;
        runtimeLibraryDirectory = qnnOptions->runtimeLibraryDirectory;
        opPackageInterfaceProvider = qnnOptions->opPackageInterfaceProvider;
        opPackageName = qnnOptions->opPackageName;
        switch (qnnOptions->runtime) {
            case MNN_QNN_RUNTIME_HTP:
                requestedBackend = QnnBackendKind::Htp;
                break;
            case MNN_QNN_RUNTIME_DSP:
                requestedBackend = QnnBackendKind::Dsp;
                break;
            default:
                break;
        }
    }

#ifndef ENABLE_QNN_CONVERT_MODE
    const bool hasLiveBackendHandles =
        QNN::gContext.backendHandle != nullptr || QNN::gIndependentContextCount.load(std::memory_order_acquire) != 0;
    if (hasLiveBackendHandles && requestedBackend != QnnBackendKind::None &&
        requestedBackend != QNN::getLoadedQNNBackend()) {
        MNN_ERROR(
            "MNN_QNN: requested %s but the process already owns a live %s backend; refusing to unload the active host "
            "backend.\n",
            requestedBackend == QnnBackendKind::Htp ? "HTP" : "DSP V66", QNN::getLoadedQNNBackendName());
        return nullptr;
    }
    if (hasLiveBackendHandles && !QNN::isLoadedQNNLibraryCompatible(requestedBackend, runtimeLibraryDirectory)) {
        MNN_ERROR(
            "MNN_QNN: the requested runtime library differs from the one used by a live backend; refusing to unload "
            "active handles.\n");
        return nullptr;
    }
    if (QNN::gContext.backendHandle == nullptr) {
        const bool symbolsLoaded = requestedBackend == QnnBackendKind::None
                                       ? QNN::loadQNNSymbol()
                                       : QNN::loadQNNSymbol(requestedBackend, runtimeLibraryDirectory);
        if (!symbolsLoaded) {
            options->report(MNN_QNN_STAGE_RUNTIME, NO_EXECUTION, "QNN runtime library could not be loaded");
            return nullptr;
        }
    }
    if (qnnOfflineContextModel && !QNN::loadQNNSystemSymbol(runtimeLibraryDirectory)) {
        options->report(MNN_QNN_STAGE_RUNTIME, NO_EXECUTION, "QNN offline System library could not be loaded");
        return nullptr;
    }
#endif
#ifdef ENABLE_QNN_CONVERT_MODE
    const bool hasLiveBackendHandles = QNN::gContext.backendHandle != nullptr;
#endif
    std::string resolvedOpPackageName;
    std::shared_ptr<QNN::IndependentQnnContextOwner> independentContext;
    QNN::QnnContext* selectedContext = &QNN::gContext;
    bool useIndependentContext = false;
#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR <= 27
    useIndependentContext = QNN::getLoadedQNNBackend() == QnnBackendKind::Dsp &&
                            qnnV66LayerNormOpPackageRequested(manifestPath, opPackageInterfaceProvider, opPackageName);
#endif
    if (useIndependentContext) {
        independentContext = QNN::createIndependentQnnContext(
            manifestPath, resourceDirectory, opPackageInterfaceProvider, opPackageName, &resolvedOpPackageName);
        if (independentContext == nullptr) {
            return nullptr;
        }
        selectedContext = &independentContext->context;
    } else if (!QNN::createQnnContext(manifestPath, resourceDirectory, opPackageInterfaceProvider, opPackageName,
                                      requestedBackend == QnnBackendKind::None && !hasLiveBackendHandles,
                                      &resolvedOpPackageName)) {
        return nullptr;
    }
    if (qnnOfflineContextModel &&
        (useIndependentContext ? selectedContext->systemInterface.systemContextCreate == nullptr
                               : !QNN::ensureQnnSystemInterface())) {
        return nullptr;
    }
    // Create Interface.
    options->report(MNN_QNN_STAGE_RUNTIME, NO_ERROR, "QNN Runtime ready");
    return new QnnRuntime(info, selectedContext->QnnInterface, selectedContext->logHandle,
                          selectedContext->backendHandle, selectedContext->deviceHandle, resolvedOpPackageName,
                          selectedContext, independentContext);
}

// Do nothing
void QnnRuntime::onGabageCollect(int level) {}

Runtime::CompilerType QnnRuntime::onGetCompilerType() const {
    return Compiler_Origin;
}

bool QnnRuntime::onSetCachePath(const char* path, int mode) {
#ifdef ENABLE_QNN_CONVERT_MODE
    MNN_ASSERT(path != nullptr);
    QNNConvertor::OutputDir = std::string(path);
    MNNCreateDir(path);
#endif
    return true;
}

bool QnnRuntime::registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target) {
    if (QNN_GET_ERROR_CODE(qnnInterface.backendRegisterOpPackage(backendHandle, path.c_str(), interfaceProvider.c_str(), target.c_str())) != QNN_SUCCESS) {
        MNN_PRINT("MNN_QNN: Failed to register the Op Package: %s.\n", path.c_str());
        return false;
    }
    return true;
}

static bool supportQnnQuant(const Op* op, const std::vector<Tensor*>& inputs,
                            const std::vector<Tensor*>& outputs, bool tblive) {
    if (op == nullptr) {
        return true;
    }
    if (inputs.empty() || outputs.empty()) return false;
    const auto opType = op->type();
    if (tblive && (opType == OpType_Scale || opType == OpType_LayerNorm || opType == OpType_ReLU)) {
        for (auto* tensor : inputs) if (!TensorUtils::getDescribe(tensor)->quantAttr) return false;
        for (auto* tensor : outputs) if (!TensorUtils::getDescribe(tensor)->quantAttr) return false;
        // Capability is checked against the selected SDK in the backend's
        // graph builder; no global or Core-side backend mode is involved.
        return true;
    }
    const std::set<OpType> oneInputQuantizedOps = {OpType_Slice,   OpType_StridedSlice, OpType_GatherV2,
                                                   OpType_Reshape, OpType_Unsqueeze,    OpType_Flatten,
                                                   OpType_Squeeze};
    if (oneInputQuantizedOps.find(opType) != oneInputQuantizedOps.end()) {
        if (TensorUtils::getDescribe(inputs[0])->quantAttr == nullptr) {
            return false;
        }
    } else {
        for (auto tensor : inputs) {
            if (TensorUtils::getDescribe(tensor)->quantAttr == nullptr) {
                return false;
            }
        }
    }
    const auto quantType = TensorUtils::getDescribe(inputs[0])->quantAttr->type;
    bool supported = true;
    switch (opType) {
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
            supported = inputs.size() <= 1 && TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr &&
                        !(op->main_as_Convolution2D() != nullptr && op->main_as_Convolution2D()->weight() != nullptr);
            break;
        case OpType_ReLU:
            supported = op->main_as_Relu() == nullptr || op->main_as_Relu()->slope() == 0.0f;
            break;
        case OpType_LayerNorm:
            supported = quantType == DataType_DT_INT16;
            break;
        case OpType_Scale:
        case OpType_Attention:
            supported = false;
            break;
        default:
            break;
    }
    return supported;
}

class QnnRuntimeCreator : public RuntimeCreator {
public:
    explicit QnnRuntimeCreator(bool tblive) : mTBLive(tblive) {}
    virtual Runtime* onCreate(const Backend::Info& info) const override { return QnnRuntime::create(info); }
    virtual bool onSetQuantInfo(const Op* op, const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) const override {
        if (op == nullptr) {
            return true;
        }
        const bool supported = supportQnnQuant(op, inputs, outputs, mTBLive);
        for (auto tensor : outputs) {
            TensorUtils::getDescribe(tensor)->applyQuant = supported;
        }
        return supported;
    }
    virtual bool onValid(Backend::Info& info) const override {
        return true;
    }
    virtual bool onGetDeviceInfo(const std::string& deviceKey, std::string& deviceValue) const override {
        if(deviceKey == "soc_id" && gContext.soc_id != 0) {
            deviceValue = std::to_string(gContext.soc_id);
            return true;
        }
        if(deviceKey == "dsp_arch" && gContext.dsp_arch != 0) {
            deviceValue = "v" + std::to_string(gContext.dsp_arch);
            return true;
        }
        return false;
    }
private:
    const bool mTBLive;
};
#endif
} // end namespace QNN

bool registerQNNRuntimeCreator(bool registerLegacyAlias, bool enableOfflineContext, MNNForwardType* registeredType) {
#ifdef ENABLE_QNN_CONVERT_MODE
    const MNNForwardType runtimeType = MNN_CONVERT_QNN;
#else
    const MNNForwardType runtimeType = MNN_FORWARD_QNN;
#endif
    if (registeredType != nullptr) {
        *registeredType = runtimeType;
    }

    // The zero-argument/automatic path keeps the 3.6.1 eager HTP readiness
    // contract, including QnnSystem for Plugin models. The additive manual
    // registration path stays lazy so an explicitly selected type-16 backend
    // can resolve its own runtime at session creation.
    const bool explicitRegistration = registeredType != nullptr;
    bool legacyRuntimeAvailable = true;
#if !defined(ENABLE_QNN_CONVERT_MODE)
    if (!explicitRegistration) {
        legacyRuntimeAvailable = QNN::loadQNNSymbol(QNN::QnnBackendKind::Htp);
#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
        if (legacyRuntimeAvailable && enableOfflineContext) {
            legacyRuntimeAvailable = QNN::loadQNNSystemSymbol({});
        }
#endif
    }
#endif

    bool available = false;
#ifdef ENABLE_QNN_ONLINE_FINALIZE
    static std::mutex registrationMutex;
    static bool runtimeRegistered = false;
    static QNN::QnnRuntimeCreator* creator = nullptr;
    std::lock_guard<std::mutex> registrationLock(registrationMutex);
    if (!runtimeRegistered) {
        static std::once_flag opRegisterFlag;
        std::call_once(opRegisterFlag, []() { QNN::registerQNNOps(); });
        creator = new QNN::QnnRuntimeCreator(runtimeType == MNN_FORWARD_QNN);
        runtimeRegistered = MNNInsertExtraRuntimeCreator(runtimeType, creator, false);
    }
#ifdef ENABLE_QNN_CONVERT_MODE
    static bool tbliveConverterRegistered = false;
    if (!tbliveConverterRegistered) {
        static QNN::QnnRuntimeCreator tbliveCreator(true);
        tbliveConverterRegistered = MNNInsertExtraRuntimeCreator(MNN_FORWARD_QNN, &tbliveCreator, false);
    }
#else
    static bool legacyAliasRegistered = false;
    if (registerLegacyAlias && legacyRuntimeAvailable && !legacyAliasRegistered) {
        static QNN::QnnRuntimeCreator legacyCreator(false);
        legacyAliasRegistered = MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, &legacyCreator, false);
    }
#endif
    available = runtimeRegistered;
#else
    (void)registerLegacyAlias;
#endif

#if defined(MNN_WITH_PLUGIN) || defined(MNN_QNN_OFFLINE_CONTEXT)
    if (enableOfflineContext && (explicitRegistration || legacyRuntimeAvailable)) {
        static std::once_flag pluginRegisterFlag;
        std::call_once(pluginRegisterFlag, []() {
            rpcmem_init();
            plugin::InferShapeKernelRegister::add("QNN", []() {        // NOLINT
                return new plugin::shape_inference::QNNPluginShapeRaw; // NOLINT
            });
            plugin::ComputeKernelRegistry<plugin::backend::QNNPluginExecuteRaw::KernelT>::add(
                "QNN", []() { return new plugin::backend::QNNPluginExecuteRaw; });
        });
    }
#else
    (void)enableOfflineContext;
#endif
    return available;
}

// Preserve the source and mangled-symbol entry point used by existing QNN
// integrations. The backend-specific registration API above is used by the
// dedicated QNN plugin.
void registerQNNRuntimeCreator() {
#if defined(MNN_WITH_PLUGIN)
    constexpr bool enableLegacyOfflineContext = true;
#else
    constexpr bool enableLegacyOfflineContext = false;
#endif
    registerQNNRuntimeCreator(true, enableLegacyOfflineContext, nullptr);
}

} // end namespace MNN
