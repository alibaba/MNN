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

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInput_F16 = cutlass::half_t;
using ElementInput_F32 = float;
using ElementOutput_F16 = cutlass::half_t;
using ElementOutput_F32 = float;

using EpilogueCudaOp_F16_Linear = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_F16_Linear = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueCudaOp_F32_Linear = cutlass::epilogue::thread::LinearCombination<
    float,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_F32_Linear = cutlass::epilogue::thread::LinearCombination<
    float,
    128 / cutlass::sizeof_bits<float>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueCudaOp_F16_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_F16_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueCudaOp_F32_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    float,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_F32_Relu = cutlass::epilogue::thread::LinearCombinationRelu<
    float,
    128 / cutlass::sizeof_bits<float>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueCudaOp_F16_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    cutlass::half_t,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_F16_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueCudaOp_F32_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    float,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueTensorOp_F32_Relu6 = cutlass::epilogue::thread::LinearCombinationRelu6<
    float,
    128 / cutlass::sizeof_bits<float>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 2;
using GemmCuda_F16_F16_Linear_AlignCuda = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F16_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmCuda_F16_F32_Linear_AlignCuda = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Linear_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F16_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Linear_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F16_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F32_Linear_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F32_Linear_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Linear_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F16_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F16_Linear_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F16_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F32_Linear_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F32_Linear_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmCuda_F32_F32_Linear_AlignCuda = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Linear_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Linear_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Linear_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<float>::value, 128 / cutlass::sizeof_bits<float>::value, true>;

using GemmTensor_F32_F32_Linear_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Linear,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<float>::value, 128 / cutlass::sizeof_bits<float>::value, true>;

using GemmCuda_F16_F16_Relu_AlignCuda = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F16_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmCuda_F16_F32_Relu_AlignCuda = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Relu_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F16_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Relu_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F16_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F32_Relu_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F32_Relu_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Relu_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F16_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F16_Relu_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F16_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F32_Relu_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F32_Relu_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmCuda_F32_F32_Relu_AlignCuda = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Relu_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Relu_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Relu_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<float>::value, 128 / cutlass::sizeof_bits<float>::value, true>;

using GemmTensor_F32_F32_Relu_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Relu,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<float>::value, 128 / cutlass::sizeof_bits<float>::value, true>;

using GemmCuda_F16_F16_Relu6_AlignCuda = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F16_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmCuda_F16_F32_Relu6_AlignCuda = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Relu6_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F16_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Relu6_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F16_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F32_Relu6_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F32_Relu6_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F16_F16_Relu6_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F16_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F16_Relu6_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F16_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F32_Relu6_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmTensor_F16_F32_Relu6_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    LayoutInputA,
    cutlass::half_t,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, true>;

using GemmCuda_F32_F32_Relu6_AlignCuda = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Relu6_AlignCuda_Sm70 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Relu6_AlignTensor_Sm70 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages>;

using GemmTensor_F32_F32_Relu6_AlignCuda_Sm75 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<float>::value, 128 / cutlass::sizeof_bits<float>::value, true>;

using GemmTensor_F32_F32_Relu6_AlignTensor_Sm75 = cutlass::gemm::device::Gemm<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Relu6,
    SwizzleThreadBlock,
    NumStages,
    128 / cutlass::sizeof_bits<float>::value, 128 / cutlass::sizeof_bits<float>::value, true>;

}
}
#endif
