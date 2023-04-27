//
//  DeconvBaseKernel.cuh
//  MNN
//
//  Created by MNN on 2023/04/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//
 
#ifndef DECONV_BASE_KERNEL_CUH
#define DECONV_BASE_KERNEL_CUH
 
#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"
#include "DeconvSingleInputExecution.hpp"

namespace MNN {
namespace CUDA {
 
void callWeightReorder(const void* input, void* output, const KernelInfo kernel_info, const int icPack, const int precision, CUDARuntime* runtime);
void callCol2ImFunc(const void* input, const void* bias, void* output, const Col2ImParameter* col2im_param, const int precision, CUDARuntime* runtime);
 
} //namespace CUDA
} //namespace MNN
#endif