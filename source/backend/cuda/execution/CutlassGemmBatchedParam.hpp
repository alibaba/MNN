#ifndef CutlassGemmBatchedParam_hpp
#define CutlassGemmBatchedParam_hpp

#include "CutlassGemmParam.hpp"
#include "cutlass/gemm/device/gemm_batched.h"

namespace MNN {
namespace CUDA {
using BatchedSwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
using ShapeBatchMMAThreadBlock = cutlass::gemm::GemmShape<32, 64, 32>;
using ShapeBatchMMAWarp = cutlass::gemm::GemmShape<16, 32, 32>;
using ShapeBatchCudaThreadBlock = cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeBatchCudaWarp = cutlass::gemm::GemmShape<32, 32, 64>;

using GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    ShapeBatchCudaThreadBlock,
    ShapeBatchCudaWarp,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    ShapeBatchCudaThreadBlock,
    ShapeBatchCudaWarp,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    ShapeBatchCudaThreadBlock,
    ShapeBatchCudaWarp,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    ShapeBatchCudaThreadBlock,
    ShapeBatchCudaWarp,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Row_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm70 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Row_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F16_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Row_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    ShapeBatchCudaThreadBlock,
    ShapeBatchCudaWarp,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    ShapeBatchCudaThreadBlock,
    ShapeBatchCudaWarp,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Row_Sm70 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm70 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm70 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm70 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Row_Sm75 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueCudaOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

using GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75 = cutlass::gemm::device::GemmBatched<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ShapeBatchMMAThreadBlock,
    ShapeBatchMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueTensorOp_F32_Linear,
    BatchedSwizzleThreadBlock,
    NumStages>;

}
}
#endif