//
//  ConvBaseKernel.cuh
//  MNN
//
//  Created by MNN on 2023/03/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONV_BASE_KERNEL_CUH
#define CONV_BASE_KERNEL_CUH

#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

void callFloat2Half(const void* input, void* output, const int count, CUDARuntime* runtime);
void callWeightFill(const void* input, void* output, const int l, const int h, const int lp, const int hp, const int precision, CUDARuntime* runtime);
void callIm2ColPack(const void* input, void* output, const ConvolutionCommon::Im2ColParameter* info, const int e, const int l, const int ep, const int lp, const int precision, CUDARuntime* runtime);

ErrorCode callCutlassGemmCudaCoreFloat16(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
ErrorCode callCutlassGemmCudaCoreFloat32(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
ErrorCode callCutlassGemmTensorCore884(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);
ErrorCode callCutlassGemmTensorCore(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs);

} //namespace CUDA
} //namespace MNN
#endif