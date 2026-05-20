/**
 * MUSA Runtime API Compatibility Layer (Fixed)
 */

#ifndef MUSA_RUNTIME_COMPAT_H
#define MUSA_RUNTIME_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Stub implementation for compilation only */
typedef int musaError_t;
enum {
    musaSuccess = 0,
    musaErrorMemoryAllocation = 1,
    musaErrorInvalidDevice = 2,
    musaErrorInvalidValue = 3,
    musaErrorNotInitialized = 4,
};

typedef struct _musaStream* musaStream_t;
typedef struct _musaEvent* musaEvent_t;

typedef enum {
    musaMemcpyHostToDevice = 0,
    musaMemcpyDeviceToHost = 1,
    musaMemcpyDeviceToDevice = 2,
    musaMemcpyDefault = 3
} musaMemcpyKind;

typedef struct {
    char name[256];
    size_t totalGlobalMem;
    int major;
    int minor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerMultiProcessor;
    int computeMode;
    int deviceOverlap;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int computePreemption;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int directManagedMemAccessFromHost;
} musaDeviceProp;

/* Stub functions */
static inline musaError_t musaMalloc(void **ptr, size_t size) {
    (void)ptr; (void)size;
    return musaErrorNotInitialized;
}
static inline musaError_t musaFree(void *ptr) {
    (void)ptr;
    return musaErrorNotInitialized;
}
static inline musaError_t musaMemcpy(void *dst, const void *src, size_t count, musaMemcpyKind kind) {
    (void)dst; (void)src; (void)count; (void)kind;
    return musaErrorNotInitialized;
}
static inline musaError_t musaMemset(void *ptr, int value, size_t count) {
    (void)ptr; (void)value; (void)count;
    return musaErrorNotInitialized;
}
static inline musaError_t musaGetDeviceCount(int *count) {
    if (count) *count = 0;
    return musaSuccess;
}
static inline musaError_t musaGetDeviceProperties(musaDeviceProp *prop, int device) {
    (void)prop; (void)device;
    return musaErrorInvalidDevice;
}
static inline musaError_t musaDeviceSynchronize(void) {
    return musaSuccess;
}
static inline musaError_t musaGetLastError(void) {
    return musaSuccess;
}
static inline const char* musaGetErrorString(musaError_t error) {
    (void)error;
    return "MUSA not available (stub)";
}
static inline musaError_t musaSetDevice(int device) {
    (void)device;
    return musaErrorInvalidDevice;
}
static inline musaError_t musaGetDevice(int *device) {
    if (device) *device = 0;
    return musaSuccess;
}
static inline musaError_t musaStreamCreate(musaStream_t *stream) {
    (void)stream;
    return musaSuccess;
}
static inline musaError_t musaStreamDestroy(musaStream_t stream) {
    (void)stream;
    return musaSuccess;
}
static inline musaError_t musaMemGetInfo(size_t *free, size_t *total) {
    if (free) *free = 0;
    if (total) *total = 0;
    return musaSuccess;
}
static inline musaError_t musaMemcpyAsync(void *dst, const void *src, size_t count, musaMemcpyKind kind, musaStream_t stream) {
    (void)dst; (void)src; (void)count; (void)kind; (void)stream;
    return musaErrorNotInitialized;
}

#ifdef __cplusplus
}
#endif

#endif /* MUSA_RUNTIME_COMPAT_H */