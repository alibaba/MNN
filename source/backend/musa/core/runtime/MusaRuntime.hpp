//
//  MusaRuntime.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef MusaRuntime_hpp
#define MusaRuntime_hpp

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <musa_runtime.h>
#include <sstream>
#include <string>
#include <vector>
#include "Type_generated.h"
#include "core/Macro.h"

typedef enum {
    MUSA_FLOAT32 = 0,
    MUSA_FLOAT16 = 1,
} MNNMUSADataType_t;

typedef enum {
    MNNMemcpyHostToDevice   = 1,
    MNNMemcpyDeviceToHost   = 2,
    MNNMemcpyDeviceToDevice = 3,
} MNNMemcpyKind_t;

#define musa_check(_x)             \
    do {                           \
        musaError_t _err = (_x);   \
        if (_err != musaSuccess) { \
            MNN_CHECK(_err, #_x);  \
        }                          \
    } while (0)

#define after_kernel_launch()           \
    do {                                \
        musa_check(musaGetLastError()); \
    } while (0)

#ifdef DEBUG
#define checkKernelErrors\
  do {                                                      \
    musaDeviceSynchronize();\
    musaError_t __err = musaGetLastError();                 \
    if (__err != musaSuccess) {                             \
      printf("File:%s Line %d: failed: %s\n", __FILE__, __LINE__,\
             musaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)
#else
#define checkKernelErrors
#endif

namespace MNN {

class MusaRuntime {
public:
    MusaRuntime(int device_id);
    ~MusaRuntime();
    MusaRuntime(const MusaRuntime &) = delete;
    MusaRuntime &operator=(const MusaRuntime &) = delete;

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
    const musaDeviceProp& prop() const {
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
    
    size_t getMemoryUsage(size_t size_in_bytes) const;

private:
    musaDeviceProp mProp;
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
#endif /* MusaRuntime_hpp */
