#ifdef ENABLE_CUDA_BF16

#ifndef CutlassGemmBF16Param_hpp
#define CutlassGemmBF16Param_hpp

#include "../CutlassGemmParam.hpp"

namespace MNN {
namespace CUDA {

using ElementInput_BF16 = cutlass::bfloat16_t;
using ElementOutput_BF16 = cutlass::bfloat16_t;

using EpilogueTensorOp_BF16_Linear = cutlass::epilogue::thread::LinearCombination<
    cutlass::bfloat16_t,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_BF16_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::bfloat16_t,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_BF16_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    cutlass::bfloat16_t,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using GemmTensor_BF16_BF16_Linear_AlignTensor_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    LayoutInputA,
    cutlass::bfloat16_t,
    LayoutInputB,
    cutlass::bfloat16_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_BF16_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, true>;

using GemmTensor_BF16_BF16_Relu_AlignTensor_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    LayoutInputA,
    cutlass::bfloat16_t,
    LayoutInputB,
    cutlass::bfloat16_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_BF16_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, true>;

using GemmTensor_BF16_BF16_Relu6_AlignTensor_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    LayoutInputA,
    cutlass::bfloat16_t,
    LayoutInputB,
    cutlass::bfloat16_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_BF16_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, true>;

}
}
#endif
#endif
