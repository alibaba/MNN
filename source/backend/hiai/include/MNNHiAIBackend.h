// Optional HiAI backend plugin API. Core MNN ABI is unchanged.
#ifndef MNN_HIAI_BACKEND_API_H
#define MNN_HIAI_BACKEND_API_H

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MNN_HIAI_CONFIG_VERSION 1u
#define MNN_HIAI_PLUGIN_ABI_VERSION 1u

typedef enum {
    MNN_HIAI_STATUS_SUCCESS = 0,
    MNN_HIAI_STATUS_INVALID_ARGUMENT = 1,
    MNN_HIAI_STATUS_NOT_COMPILED = 2,
    MNN_HIAI_STATUS_UNAVAILABLE = 3,
    MNN_HIAI_STATUS_MODE_CONFLICT = 4,
    MNN_HIAI_STATUS_UNSUPPORTED_OPTION = 6,
} MNNHiAIStatus;

typedef enum {
    MNN_HIAI_STAGE_CONFIG = 0,
    MNN_HIAI_STAGE_RUNTIME = 1,
    MNN_HIAI_STAGE_RESIZE = 2,
    MNN_HIAI_STAGE_EXECUTE = 3,
    MNN_HIAI_STAGE_COPY = 4,
} MNNHiAIStage;

/** Caller-owned synchronous diagnostics. Do not share between concurrent sessions. */
typedef struct {
    uint32_t structSize;
    uint32_t version;
    int32_t code;
    MNNHiAIStage stage;
    int64_t nativeCode;
    uint32_t readyCount;
    char message[512];
} MNNHiAIDiagnosticsV1;

struct MNNHiAINativeHandleIoContext;

/**
 * Pass through BackendConfig::sharedContext. The HiAI Runtime copies strings
 * and scalar options. diagnostics and nativeIoContext are borrowed and must
 * outlive every Session and Runtime clone that uses this configuration.
 */
typedef struct {
    uint32_t structSize;
    uint32_t version;
    const char* runtimeLibraryDirectory;
    const char* acceleratorCacheDirectory;
    uint32_t autoTuning;
    struct MNNHiAINativeHandleIoContext* nativeIoContext;
    MNNHiAIDiagnosticsV1* diagnostics;
} MNNHiAIBackendConfigV1;

typedef struct {
    uint32_t structSize;
    uint32_t abiVersion;
    MNNForwardType forwardType;
    MNNHiAIStatus (*registerRuntime)(void);
    MNNHiAIStatus (*validateModel)(const MNNHiAIBackendConfigV1*, const void*, size_t);
} MNNHiAIBackendApiV1;

typedef const MNNHiAIBackendApiV1* (*MNNGetHiAIBackendApiV1Fn)(void);

/** Register before concurrent MNN use and keep the plugin loaded afterwards. */
MNN_PUBLIC MNNHiAIStatus MNNRegisterHiAIRuntime(void);
MNN_PUBLIC const MNNHiAIBackendApiV1* MNNGetHiAIBackendApiV1(void);

#ifdef __cplusplus
}
#endif
#endif
