// Optional QNN backend plugin API. Core MNN ABI is unchanged.
#ifndef MNN_QNN_BACKEND_API_H
#define MNN_QNN_BACKEND_API_H

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MNN_QNN_CONFIG_VERSION 1u
#define MNN_QNN_PLUGIN_ABI_VERSION 1u

typedef enum {
    MNN_QNN_STATUS_SUCCESS = 0,
    MNN_QNN_STATUS_INVALID_ARGUMENT = 1,
    MNN_QNN_STATUS_NOT_COMPILED = 2,
    MNN_QNN_STATUS_UNAVAILABLE = 3,
    MNN_QNN_STATUS_MODE_CONFLICT = 4,
    MNN_QNN_STATUS_UNSUPPORTED_OPTION = 6,
} MNNQnnStatus;

typedef enum {
    MNN_QNN_CONFIG_NONE = 0,
    MNN_QNN_CONFIG_OFFLINE_CONTEXT = 1u << 0,
    MNN_QNN_CONFIG_DUMP_OUTPUTS = 1u << 16,
} MNNQnnConfigFlags;

typedef enum {
    MNN_QNN_RUNTIME_AUTO = 0,
    MNN_QNN_RUNTIME_HTP = 1,
    MNN_QNN_RUNTIME_DSP = 2,
} MNNQnnRuntimeKind;

typedef enum {
    MNN_QNN_STAGE_CONFIG = 0,
    MNN_QNN_STAGE_RUNTIME = 1,
    MNN_QNN_STAGE_RESIZE = 2,
    MNN_QNN_STAGE_EXECUTE = 3,
    MNN_QNN_STAGE_COPY = 4,
} MNNQnnStage;

/** Caller-owned synchronous diagnostics. Do not share between concurrent sessions. */
typedef struct {
    uint32_t structSize;
    uint32_t version;
    int32_t code;
    MNNQnnStage stage;
    int64_t nativeCode;
    uint32_t readyCount;
    char message[512];
} MNNQnnDiagnosticsV1;

/**
 * Pass through BackendConfig::sharedContext. The QNN Runtime copies all
 * strings and scalar options. diagnostics is borrowed and must outlive every
 * Session and Runtime clone that uses this configuration.
 */
typedef struct {
    uint32_t structSize;
    uint32_t version;
    uint32_t flags;
    MNNQnnRuntimeKind runtime;
    uint32_t directInt8NhwcIo;
    const char* runtimeLibraryDirectory;
    const char* runtimeManifestPath;
    const char* acceleratorResourceDirectory;
    const char* modelDirectory;
    const char* opPackageInterfaceProvider;
    const char* opPackageName;
    MNNQnnDiagnosticsV1* diagnostics;
} MNNQnnBackendConfigV1;

typedef struct {
    uint32_t structSize;
    uint32_t abiVersion;
    MNNForwardType forwardType;
    MNNQnnStatus (*registerRuntime)(void);
    MNNQnnStatus (*validateModel)(const MNNQnnBackendConfigV1*, const void*, size_t);
} MNNQnnBackendApiV1;

typedef const MNNQnnBackendApiV1* (*MNNGetQnnBackendApiV1Fn)(void);

/** Register before concurrent MNN use and keep the plugin loaded afterwards. */
MNN_PUBLIC MNNQnnStatus MNNRegisterQNNRuntime(void);
MNN_PUBLIC const MNNQnnBackendApiV1* MNNGetQnnBackendApiV1(void);

#ifdef __cplusplus
}
#endif
#endif
