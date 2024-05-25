/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/sparse_gemm.h"

#include "cutlass/gemm/kernel/default_gemm_sparse_with_visitor.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Sparse GEMM with visitor
 */
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm80,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename FusionCallbacks_ =
        typename cutlass::epilogue::threadblock::detail::EmptyCallbacks,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Number of stages used in the pipelined epilogue
    int EpilogueStages = 1>
class SparseGemmWithVisitor {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using FusionCallbacks = FusionCallbacks_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  using MathOperator = Operator;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;

  /// Define the kernel
  using GemmKernel = typename kernel::DefaultSparseGemmWithVisitor<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    FusionCallbacks,
    ThreadblockSwizzle,
    kStages,
    Operator,
    EpilogueStages
  >::GemmKernel;

  using ElementE = typename GemmKernel::ElementE;

  using LayoutE = typename GemmKernel::LayoutE;

  static int const kAlignmentE = 128 / sizeof_bits<ElementE>::value;

  static int const kSparse = GemmKernel::kSparse;
  static int const kMetaSizeInBits = GemmKernel::kMetaSizeInBits;
  static int const kElementsPerElementE = GemmKernel::kElementsPerElementE;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementE const, LayoutE> ref_E;
    typename FusionCallbacks::Arguments epilogue;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0) {

    }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementE, LayoutE> ref_E_,
      typename FusionCallbacks::Arguments epilogue_ = 
        typename FusionCallbacks::Arguments()
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_E(ref_E_),
      epilogue(epilogue_) {

    }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  SparseGemmWithVisitor() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    Status status = GemmKernel::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      cutlass::TensorRef<ElementC, LayoutC>(), // It only matters that it's empty.
      cutlass::TensorRef<ElementC, LayoutC>(), // Same as above.
      args.ref_E.non_const_ref()
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    size_t bytes = 0;

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    constexpr int SplitKSlices = 1;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      SplitKSlices);

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_E.non_const_ref(),
      args.epilogue
    };

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_E.reset(args.ref_E.non_const_ref().data());
    params_.output_op = args.epilogue;

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {

    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
