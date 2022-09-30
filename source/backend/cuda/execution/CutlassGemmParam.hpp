//
//  ConvCutlassExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CutlassGemmParam_hpp
#define CutlassGemmParam_hpp

#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_relu6.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace MNN {
namespace CUDA {

struct CutlassGemmInfo{
    int elh[3];
    int elhPad[3];
};

using ElementAccumulator = float;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;                       // <- data type of elements in input matrix B
using ElementOutput_F16 = cutlass::half_t;                  // <- data type of elements in output matrix D
using ElementOutput_F32 = float;                  // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch70 = cutlass::arch::Sm70;
using SmArch75 = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<64, 64, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 64>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp1688 = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 16
using ShapeMMAOp884 = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp_F16_Linear = cutlass::epilogue::thread::LinearCombination<
    ElementOutput_F16,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_F16>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using EpilogueOp_F32_Linear = cutlass::epilogue::thread::LinearCombination<
    ElementOutput_F32,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_F32>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using EpilogueOp_F16_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput_F16,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_F16>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue>;                              // <- data type for alpha in linear combination function

using EpilogueOp_F32_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput_F32,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_F32>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue>;                              // <- data type for alpha in linear combination function

using EpilogueOp_F16_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    ElementOutput_F16,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_F16>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue>;                              // <- data type for alpha in linear combination function

using EpilogueOp_F32_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    ElementOutput_F32,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_F32>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue>;                              // <- data type for alpha in linear combination function
// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm_F16_Linear_Sm70 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch70,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp884,
                                         EpilogueOp_F16_Linear,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F16_Linear_Sm75 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F16_Linear,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F32_Linear_Sm70 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch70,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp884,
                                         EpilogueOp_F32_Linear,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F32_Linear_Sm75 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F32_Linear,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F16_Relu_Sm70 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch70,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp884,
                                         EpilogueOp_F16_Relu,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F16_Relu_Sm75 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F16_Relu,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F32_Relu_Sm70 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch70,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp884,
                                         EpilogueOp_F32_Relu,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F32_Relu_Sm75 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F32_Relu,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F16_Relu6_Sm70 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch70,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp884,
                                         EpilogueOp_F16_Relu6,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F16_Relu6_Sm75 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F16_Relu6,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F32_Relu6_Sm70 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch70,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp884,
                                         EpilogueOp_F32_Relu6,
                                         SwizzleThreadBlock,
                                         NumStages>;

using Gemm_F32_Relu6_Sm75 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F32_Relu6,
                                         SwizzleThreadBlock,
                                         NumStages>;

// This code section describes how threadblocks are scheduled on GPU
using BatchedSwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;  // <- ??

using ShapeBatchMMAThreadBlock =
    cutlass::gemm::GemmShape<64, 64, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeBatchMMAWarp = cutlass::gemm::GemmShape<16, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64 

using GemmBatched_F16_Linear_Sm75 = cutlass::gemm::device::GemmBatched<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F16,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeBatchMMAThreadBlock,
                                         ShapeBatchMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F16_Linear,
                                         BatchedSwizzleThreadBlock,
                                         NumStages>;

using GemmBatched_F32_Linear_Sm75 = cutlass::gemm::device::GemmBatched<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput_F32,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch75,
                                         ShapeBatchMMAThreadBlock,
                                         ShapeBatchMMAWarp,
                                         ShapeMMAOp1688,
                                         EpilogueOp_F32_Linear,
                                         BatchedSwizzleThreadBlock,
                                         NumStages>;

} // namespace CUDA
} // namespace MNN

#endif /* CutlassGemmParam_hpp */