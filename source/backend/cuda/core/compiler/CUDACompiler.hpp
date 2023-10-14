//
//  CUDACompiler.hpp
//  MNN
//
//  Created by MNN on 2023/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CUDA_COMPILER_H_
#define MNN_CUDA_COMPILER_H_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define MNN_NVRTC_SAFE_CALL(x)                                                                        \
    do {                                                                                              \
        nvrtcResult result = x;                                                                       \
        if (result != NVRTC_SUCCESS) {                                                                \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

#define MNN_CUDA_SAFE_CALL(x)                                                 \
    do {                                                                      \
        CUresult result = x;                                                  \
        if (result != CUDA_SUCCESS) {                                         \
            const char* msg;                                                  \
            cuGetErrorName(result, &msg);                                     \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n'; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define MNN_RUNTIME_SAFE_CALL(x)                                              \
    do {                                                                      \
        cudaError_t result = x;                                               \
        if (result != cudaSuccess) {                                          \
            const char* msg = cudaGetErrorName(result);                       \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n'; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

namespace MNN {
namespace CUDA {
std::string CUDANVRTCCompile(std::pair<string, string> code, std::vector<const char*> compile_params, int device,
                             bool include);
} // namespace CUDA
} // namespace MNN

#endif