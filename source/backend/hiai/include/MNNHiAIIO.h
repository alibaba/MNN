// HiAI-only native I/O contract.
#ifndef MNN_HIAI_IO_H
#define MNN_HIAI_IO_H
#include <stdint.h>

#define MNN_HIAI_NATIVE_HANDLE_IO_MAGIC 0x4D4E4E484941494FULL
#define MNN_HIAI_NATIVE_HANDLE_IO_VERSION 1U

#define MNN_HIAI_SESSION_CONTROL_INDEX 0U
#define MNN_HIAI_SESSION_STATUS_INDEX 1U
#define MNN_HIAI_SESSION_RELEASE_STATUS_INDEX 2U

enum MNNHiAISessionControlFlags {
    MNN_HIAI_SESSION_FORCE_V320 = 1U << 0,
};

enum MNNHiAISessionBuildStatus {
    MNN_HIAI_SESSION_BUILD_PENDING = 0,
    MNN_HIAI_SESSION_HCL_V600_SELECTED = 1,
    MNN_HIAI_SESSION_HCL_V600_FAILED = 2,
    MNN_HIAI_SESSION_V320_SELECTED = 3,
    MNN_HIAI_SESSION_V320_READY = 4,
    MNN_HIAI_SESSION_V320_FAILED = 5,
};

enum MNNHiAISessionReleaseStatus {
    MNN_HIAI_SESSION_RELEASE_PENDING = 0,
    MNN_HIAI_SESSION_RELEASE_SUCCESS = 1,
    MNN_HIAI_SESSION_RELEASE_FAILED = 2,
};

enum MNNHiAINativeHandleIoResult {
    MNN_HIAI_NATIVE_HANDLE_IO_PENDING = -1,
    MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS = 0,
    MNN_HIAI_NATIVE_HANDLE_IO_INVALID_CONTEXT = 1,
    MNN_HIAI_NATIVE_HANDLE_IO_SYMBOL_UNAVAILABLE = 2,
    MNN_HIAI_NATIVE_HANDLE_IO_INVALID_BUFFER_HANDLE = 3,
    MNN_HIAI_NATIVE_HANDLE_IO_INPUT_BIND_FAILED = 4,
    MNN_HIAI_NATIVE_HANDLE_IO_OUTPUT_BIND_FAILED = 5,
};

struct MNNHiAINativeHandleIoContext {
    uint64_t magic;
    uint32_t version;
    uint32_t struct_size;

    /* Borrowed AHardwareBuffer pointers, updated by the caller per frame. */
    void* input_ahardware_buffer;
    void* output_ahardware_buffer;
    uint32_t input_bytes;
    uint32_t output_bytes;

    /* Written synchronously by the HiAI backend. */
    int32_t result_code;
    int32_t process_result;
    int32_t input_native_fd_count;
    int32_t output_native_fd_count;
    uint32_t reserved[8];
};

#endif
