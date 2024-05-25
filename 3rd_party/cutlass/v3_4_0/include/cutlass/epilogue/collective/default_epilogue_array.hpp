/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/int.hpp"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Applies an element wise operation to all elements within the fragment
// and writes them out to destination storage.
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_
>
class DefaultEpilogueArray {
public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;
  
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;
  using StridesC = cute::conditional_t<cute::is_same_v<EpilogueSchedule, NoSmemWarpSpecializedGroup>,
    StrideC const*, StrideC>;
  using StridesD = cute::conditional_t<cute::is_same_v<EpilogueSchedule, NoSmemWarpSpecializedGroup>,
    StrideD const*, StrideD>;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(cute::is_same_v<EpilogueSchedule, NoSmemWarpSpecializedGroup> ||
    cute::is_same_v<EpilogueSchedule, NoSmemWarpSpecializedArray>, "Incompatible epilogue schedule.");
  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage { };

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const** ptr_C = nullptr;
    StridesC dC{};
    ElementD** ptr_D = nullptr;
    StridesD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const&,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  DefaultEpilogueArray(Params const& params_)
      : params(params_), epilogue_op(params_.thread) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      [[maybe_unused]] char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    // Batches are managed by using appropriate pointers to C and D matrices
    const int32_t mock_L = 1;
    const int32_t mock_l_coord = 0;
    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

    StrideC stride_c;
    StrideD stride_d;
    if constexpr (cute::is_same_v<EpilogueSchedule, NoSmemWarpSpecializedGroup>) {
      stride_c = detail::get_epilogue_stride<EpilogueSchedule>(params.dC[l_coord]);
      stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD[l_coord]);
    }
    else {
      stride_c = detail::get_epilogue_stride<EpilogueSchedule>(params.dC);
      stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD);
    }

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C[l_coord]), make_shape(M,N,mock_L), stride_c);      // (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D[l_coord]), make_shape(M,N,mock_L), stride_d);      // (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)

    Tensor gC = gC_mnl(_,_,m_coord,n_coord, mock_l_coord);                                                 // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord, mock_l_coord);                                                 // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                       // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC);                                       // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    // Make an identity coordinate tensor for predicating our output MN tile
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
        }
      }
    }
    // source is not needed, avoid load
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          tCgD(i) = epilogue_op(accumulators(i));
        }
      }
    }
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
