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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <sstream>
#include <string>
#include <vector>
#include "Type_generated.h"
#include "core/Macro.h"

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

#define after_kernel_launch()           \
    do {                                \
        cuda_check(cudaGetLastError()); \
    } while (0)

#ifdef DEBUG
#define checkKernelErrors\
  do {                                                      \
    cudaDeviceSynchronize();\
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("File:%s Line %d: failed: %s\n", __FILE__, __LINE__,\
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


#define cutlass_check(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
        printf("File:%s Line %d: failed: %s\n", __FILE__, __LINE__,\
            cutlassGetStatusString(error)); \
        abort();                                              \
    }                                                                                            \
  }
#else
#define checkKernelErrors
#define cutlass_check
#endif

namespace MNN {

class CUDARuntime {
public:
    CUDARuntime(int device_id);
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
    void device_sync();
    
    size_t threads_num() {
        return mThreadPerBlock;
    }
    const cudaDeviceProp& prop() const {
        return mProp;
    }
    int major_sm() const {
        return mProp.major;
    }
    int compute_capability() {
        return mProp.major * 10 + mProp.minor;
    }
    size_t blocks_num(const size_t total_threads);
    const int smemPerBlock() {
        return mProp.sharedMemPerBlock;
    }

    std::map<std::pair<std::vector<int32_t>, std::vector<uint32_t>>, std::pair<std::string, uint32_t>> & getTunedBlockWarpShape() {
        return mTunedBlockWarpShape;
    };
    std::pair<const void*, size_t> makeCache();
    bool setCache(std::pair<const void*, size_t> cache);

    int selectDeviceMaxFreeMemory();

private:
    cudaDeviceProp mProp;
    int mDeviceId;
    int mDeviceCount;

    bool mIsSupportedFP16   = false;
    bool mSupportDotInt8    = false;
    bool mSupportDotAccInt8 = false;
    float mFlops            = 4.0f;
    bool mIsCreateError{false};
    size_t mThreadPerBlock = 128;

private:
    std::map<std::pair<std::vector<int32_t>, std::vector<uint32_t>>, std::pair<std::string, uint32_t>> mTunedBlockWarpShape;
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
};

} // namespace MNN
#endif /* CUDARuntime_hpp */
