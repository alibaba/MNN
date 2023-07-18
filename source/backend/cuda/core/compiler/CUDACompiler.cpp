//
//  CUDACompiler.cpp
//  MNN
//
//  Created by MNN on 2023/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CUDACompiler.hpp"
#if defined(__linux__)
#include <sys/stat.h>
#endif

namespace MNN { 
namespace CUDA {

static std::string findCUDAIncludePath() {
    std::string include_path;
#ifdef MNN_CODEGEN_CUDA
    #if defined(_WIN32) || defined(_WIN64)
    const std::string delimiter = "\\";
    #else
    const std::string delimiter = "/";
    #endif

    const char* cuda_path = "CUDA_PATH";
    const char* cuda_path_env = std::getenv(cuda_path);
    if (cuda_path_env != nullptr) {
        include_path = cuda_path_env + delimiter + "include";
        return include_path;
    }
    #if defined(__linux__)
    struct stat st;
    include_path = "/usr/local/cuda/include";
    if (stat(include_path.c_str(), &st) == 0) {
        return include_path;
    }

    if (stat("/usr/include/cuda.h", &st) == 0) {
        return "/usr/include";
    }
    #endif
#endif
    return include_path;
}

std::string CUDANVRTCCompile(std::pair<string, string> code, std::vector<const char*> compile_params, int device,
                             bool include) {
    std::string ptx_code;

#ifdef MNN_CODEGEN_CUDA
    std::vector<const char*> cuda_compile_params{"-default-device", "--ftz=true"};
    std::vector<std::string> params;
    // printf("cuda device sm_%d\n", device);
    std::string compile_arch = "-arch=compute_" + std::to_string(device);
    params.push_back(compile_arch); 

    if (include) {
        std::string cuda_include = "--include-path=" + findCUDAIncludePath();
        params.push_back(cuda_include);
    }

    for (auto& iter : params) {
        cuda_compile_params.push_back(iter.c_str());
    }
    for (auto& iter : compile_params) {
        cuda_compile_params.push_back(iter);
    }

    nvrtcProgram program;
    MNN_NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, code.second.c_str(), code.first.c_str(), 0, nullptr, nullptr));

    nvrtcResult compile_res = nvrtcCompileProgram(program, cuda_compile_params.size(), cuda_compile_params.data());
    if (compile_res != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log;
        log.resize(log_size);
        nvrtcGetProgramLog(program, &log[0]);
        std::cout << log << '\n';

        //LOG(ERROR) << log;
    }

    size_t ptx_size = 0;
    MNN_NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
    ptx_code.resize(ptx_size);

    MNN_NVRTC_SAFE_CALL(nvrtcGetPTX(program, &ptx_code[0]));
    MNN_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
    cudaDeviceSynchronize();
#endif
    return ptx_code;
}

}} // namespace MNN::CUDA