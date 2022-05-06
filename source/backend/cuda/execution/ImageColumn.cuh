//
//  ImageColumn.cuh
//  MNN
//
//  Created by MNN on 2021/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IMAGE_COLUMN_CUH
#define IMAGE_COLUMN_CUH

#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "TensorCoreGemm.cuh"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

void Im2ColMain(CUDARuntime* runtime, const MatMulParam* cpuMatlMul, const MatMulParam* gpuMatMul, const ConvolutionCommon::Im2ColParameter* cpuIm2Col, const ConvolutionCommon::Im2ColParameter* gpuIm2Col, const Tensor* input, __half* mIm2ColBuffer, int ePack, int eShift, int bytes, int iBlock);

} // namespace CUDA
} // namespace MNN
#endif

