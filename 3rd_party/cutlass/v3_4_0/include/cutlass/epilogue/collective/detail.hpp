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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/int.hpp"
#include "cute/util/type_traits.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Stride>
constexpr bool
is_m_major() {
  return cutlass::gemm::detail::is_major<0,Stride>();
}

template <class Stride>
constexpr bool
is_n_major() {
  return cutlass::gemm::detail::is_major<1,Stride>();
}

using cutlass::atomic_maximum;

template <class T>
static constexpr int elements_per_access_v = cutlass::sizeof_bits<uint32_t>::value / cutlass::sizeof_bits<T>::value;

template <class EpilogueSchedule>
static constexpr bool sm90_is_cooperative_v =
  cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueSchedule>;

template <class EpilogueSchedule>
static constexpr bool sm90_is_warp_specialized_v =
  cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecialized, EpilogueSchedule>;

template <class T>
struct EmptyStorage {
  CUTLASS_HOST_DEVICE
  T* data() { return nullptr; }
};

template<class EpilogueSchedule, class Stride>
CUTLASS_HOST_DEVICE
auto get_epilogue_stride(Stride stride){
  if constexpr (cute::is_base_of_v<cutlass::gemm::EpilogueTransposed, EpilogueSchedule>) {
    return cute::make_stride(cute::get<1>(stride), cute::get<0>(stride), cute::get<2>(stride));
  }
  else {
    return stride;
  }
}

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithBias { 
  static constexpr bool value = false; 
  using type = typename ThreadEpilogueOp::ElementCompute; 
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithBias <ThreadEpilogueOp, cute::void_t<typename ThreadEpilogueOp::ElementBias>> { 
  static constexpr bool value = true; 
  using type = typename ThreadEpilogueOp::ElementBias; 
};

// Wrapper class to use operator-style epilogues in sm90 TMA warp-specialized kernels
template <class EpilogueOp>
class Sm90TmaWarpSpecializedAdapter : public EpilogueOp {
public:
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  using LoadPipeline = cutlass::PipelineTransactionAsync<0>;
  using LoadPipelineState = cutlass::PipelineState<0>;
  constexpr static uint32_t TmaTransactionBytes = 0;

  using StorePipeline = cutlass::PipelineTmaStore<0>;
  using StorePipelineState = cutlass::PipelineState<0>;

  using TensorStorage = typename EpilogueOp::SharedStorage;
  using PipelineStorage = typename LoadPipeline::SharedStorage;

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_load_pipe_increment([[maybe_unused]] TileShapeMNK) {
    return 1;
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_store_pipe_increment([[maybe_unused]] TileShapeMNK) {
    return 1;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors([[maybe_unused]] typename EpilogueOp::Params const&) {
  }

  // ctor inheritance
  using EpilogueOp::EpilogueOp;

  CUTLASS_HOST_DEVICE
  Sm90TmaWarpSpecializedAdapter(
      typename EpilogueOp::Params const& params,
      [[maybe_unused]] TensorStorage& shared_tensors)
    : EpilogueOp(params) { }

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return false;
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  load(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      [[maybe_unused]] ProblemShapeMNKL problem_shape_mnkl,
      [[maybe_unused]] TileShapeMNK tile_shape_MNK,
      [[maybe_unused]] TileCoordMNKL tile_coord_mnkl,
      [[maybe_unused]] TiledMma tiled_mma,
      [[maybe_unused]] int thread_idx,
      [[maybe_unused]] TensorStorage& shared_tensors,
      [[maybe_unused]] int subtile_idx=-1)
  {
    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto
  load_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state)
  {
    return load_pipe_producer_state;
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  store(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      int subtile_index = -1)
  {
    constexpr int BLK_M_RANK = cute::rank<0>(tile_shape_MNK);
    auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
        return get<0,i>(problem_shape_mnkl) - get<0,i>(tile_shape_MNK) * get<0,i>(tile_coord_mnkl);
      }));

    constexpr int BLK_N_RANK = cute::rank<1>(tile_shape_MNK);
    auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
        return get<1,i>(problem_shape_mnkl) - get<1,i>(tile_shape_MNK) * get<1,i>(tile_coord_mnkl);
      }));

    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

    (*this)(
        problem_shape_mnkl,
        tile_shape_MNK,
        tile_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        reinterpret_cast<char*>(&shared_tensors));

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  store_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state) {
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

};

} // namespace detail
} // namespace collective
} // namespace epilogue
} // namespace cutlass
