//
//  CUDARuntime.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRuntime_hpp
#define OpenCLRuntime_hpp

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cusolverDn.h>
#include <sstream>
#include <string>
#include <vector>
#include "Type_generated.h"
#include "core/Macro.h"
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

typedef enum {
    CUDA_FLOAT32 = 0,
    CUDA_FLOAT16 = 1,
} MNNCUDADataType_t;

typedef enum {
    MNNMemcpyHostToDevice   = 1,
    MNNMemcpyDeviceToHost   = 2,
    MNNMemcpyDeviceToDevice = 3,
} MNNMemcpyKind_t;

#define cuda_check(_x)             \
    do {                           \
        cudaError_t _err = (_x);   \
        if (_err != cudaSuccess) { \
            MNN_CHECK(_err, #_x);  \
        }                          \
    } while (0)

#define cublas_check(_x)                     \
    do {                                     \
        cublasStatus_t _err = (_x);          \
        if (_err != CUBLAS_STATUS_SUCCESS) { \
            MNN_CHECK(_err, #_x);            \
        }                                    \
    } while (0)

#define cudnn_check(_x)                     \
    do {                                    \
        cudnnStatus_t _err = (_x);          \
        if (_err != CUDNN_STATUS_SUCCESS) { \
            MNN_CHECK(_err, #_x);           \
        }                                   \
    } while (0)

#define cusolver_check(_x)                     \
    do {                                       \
        cusolverStatus_t _err = (_x);          \
        if (_err != CUSOLVER_STATUS_SUCCESS) { \
            MNN_CHECK(_err, #_x);              \
        }                                      \
    } while (0)

#define after_kernel_launch()           \
    do {                                \
        cuda_check(cudaGetLastError()); \
    } while (0)

namespace MNN {

class CUDARuntime {
public:
    CUDARuntime(bool permitFloat16, int device_id);
    ~CUDARuntime();
    CUDARuntime(const CUDARuntime &) = delete;
    CUDARuntime &operator=(const CUDARuntime &) = delete;

    bool isSupportedFP16() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const;

    float flops() const {
        return mFlops;
    }
    int device_id() const;
    size_t mem_alignment_in_bytes() const;
    void activate();
    void *alloc(size_t size_in_bytes);
    void free(void *ptr);

    void memcpy(void *dst, const void *src, size_t size_in_bytes, MNNMemcpyKind_t kind, bool sync = false);
    void memset(void *dst, int value, size_t size_in_bytes);
    cublasHandle_t cublas_handle();
    cudnnHandle_t cudnn_handle();

    int threads_num() {
        return mThreadPerBlock;
    }
    int major_sm() const {
        return mProp.major;
    }
    int blocks_num(const int total_threads);
    const cudaDeviceProp& prop() const {
        return mProp;
    }

private:
    cudaDeviceProp mProp;
    int mDeviceId;

    cublasHandle_t mCublasHandle;
    cudnnHandle_t mCudnnHandle;

    bool mIsSupportedFP16   = false;
    bool mSupportDotInt8    = false;
    bool mSupportDotAccInt8 = false;
    float mFlops            = 4.0f;
    bool mIsCreateError{false};
    int mThreadPerBlock = 128;
};

} // namespace MNN
#endif /* CUDARuntime_hpp */
