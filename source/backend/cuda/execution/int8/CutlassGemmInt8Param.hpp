//
//  CutlassGemmInt8Param.hpp
//  MNN
//
//  Created by MNN on 2023/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#ifndef CutlassGemmInt8Param_hpp
#define CutlassGemmInt8Param_hpp

#include "../CutlassGemmParam.hpp"
#include "cutlass/epilogue/thread/linear_combination_bias_scale_clamp.h"
#include "cutlass/gemm/device/gemm_bias_scale.h"

namespace MNN {
namespace CUDA {

using SwizzleThreadBlockInt = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueTensorOp_Clamp = cutlass::epilogue::thread::LinearCombinationBiasScaleClamp<
    int32_t,            // bias data type   --> int32_t
    float,        // Scale data type  --> float
    int32_t,  // gemm result accumulator type  --> int32_t
    float,    // compute data type
    int8_t,      // epilogue output  --> int8_t
    8//128 / cutlass::sizeof_bits<float>::value // vector handle size
>;

using GemmInt8Tensor_Clamp_AlignTensor_Normal = cutlass::gemm::device::GemmBiasScale<
    int8_t,
    LayoutInputA,
    int8_t,
    LayoutInputB,
    int8_t,
    LayoutOutput,
    int32_t,//ElementAccumulator
    int32_t,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    EpilogueTensorOp_Clamp,
    SwizzleThreadBlock,
    NumStages, 16, 16>;


using GemmInt8Tensor_Clamp_AlignTensor_Little = cutlass::gemm::device::GemmBiasScale<
    int8_t,
    LayoutInputA,
    int8_t,
    LayoutInputB,
    int8_t,
    LayoutOutput,
    int32_t,//ElementAccumulator
    int32_t,//bias
    float,//scale
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 32, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<8, 8, 16>,

    EpilogueTensorOp_Clamp,
    SwizzleThreadBlockInt,
    NumStages, 16, 16>;

using GemmInt8Tensor_Clamp_AlignTensor_Large = cutlass::gemm::device::GemmBiasScale<
    int8_t,
    LayoutInputA,
    int8_t,
    LayoutInputB,
    int8_t,
    LayoutOutput,
    int32_t,//ElementAccumulator
    int32_t,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    EpilogueTensorOp_Clamp,
    SwizzleThreadBlock,
    NumStages, 16, 16>;

using GemmInt8Tensor_Clamp_AlignTensor_Normal_Sm80 = cutlass::gemm::device::GemmBiasScale<
    int8_t,
    LayoutInputA,
    int8_t,
    LayoutInputB,
    int8_t,
    LayoutOutput,
    int32_t,//ElementAccumulator
    int32_t,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueTensorOp_Clamp,
    SwizzleThreadBlock,
    NumStages, 16, 16>;


}
}
#endif
#endif