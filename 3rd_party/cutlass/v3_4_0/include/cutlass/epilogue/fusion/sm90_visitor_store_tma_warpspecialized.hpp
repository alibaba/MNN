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
  \brief Visitor tree store operations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"

#include "cute/tensor.hpp"
#include "sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class EpilogueTile,
  class Element,
  FloatRoundStyle RoundStyle,
  class StrideMNL,
  class SmemLayoutAtom,
  class CopyOpR2S,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90AuxStore {
  static_assert(Alignment * sizeof_bits_v<Element> % 128 == 0, "sub-16B alignment not supported yet");

  constexpr static bool is_m_major = epilogue::collective::detail::is_m_major<StrideMNL>();
  // Find the max contiguous layout usable by TMA (if EpilogueTile is a non-compact tiler)
  using SmemShapeTma = decltype(make_shape(
      max_common_vector(make_layout(get<0>(EpilogueTile{})),make_layout(get<0>(EpilogueTile{}))),
      max_common_vector(make_layout(get<1>(EpilogueTile{})),make_layout(get<1>(EpilogueTile{})))));
  using SmemLayoutTma = decltype(tile_to_shape(
      SmemLayoutAtom{}, SmemShapeTma{},
      cute::conditional_t<is_m_major, Step<_2,_1>, Step<_1,_2>>{} ));
  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutTma{},
      make_shape(size<0>(shape(EpilogueTile{})), size<1>(shape(EpilogueTile{})), Int<Stages>{}),
      cute::conditional_t<is_m_major, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));

  struct SharedStorage {
    alignas(cutlass::detail::alignment_for_swizzle(SmemLayout{}))
    array_aligned<Element, size(SmemLayout{})> smem_aux;
  };

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  struct Params {
    using TMA_Aux = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(static_cast<Element*>(nullptr), repeat_like(StrideMNL{}, int32_t(0)), StrideMNL{}),
        SmemLayoutTma{}));
    TMA_Aux tma_store_aux;
    bool is_nullptr = false;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;

    bool is_nullptr = false;
    if constexpr (EnableNullptr) {
      is_nullptr = args.ptr_aux == nullptr;
    }

    typename Params::TMA_Aux tma_store_aux;
    if (not is_nullptr) {
      Tensor tensor_aux = make_tensor(args.ptr_aux, make_layout(make_shape(M,N,L), args.dAux));
      tma_store_aux = make_tma_copy(SM90_TMA_STORE{}, tensor_aux, SmemLayoutTma{});
    }

    return {tma_store_aux, is_nullptr};
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

  CUTLASS_HOST_DEVICE
  Sm90AuxStore() { }

  CUTLASS_HOST_DEVICE
  Sm90AuxStore(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_aux(const_cast<Element*>(shared_storage.smem_aux.data())) { }

  Params const* params_ptr;
  Element* smem_aux;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <
    class RTensor,
    class TiledR2S,
    class STensorR2S,
    class STensorS2G,
    class GTensorS2G
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
          RTensor&& tC_rAux,
          TiledR2S tiled_r2s,
          STensorR2S&& tRS_sAux,
          STensorS2G&& bSG_sAux,
          GTensorS2G&& bSG_gAux,
          Params const* params_ptr)
      : tiled_r2s(tiled_r2s),
        tC_rAux(cute::forward<RTensor>(tC_rAux)),
        tRS_sAux(cute::forward<STensorR2S>(tRS_sAux)),
        bSG_sAux(cute::forward<STensorS2G>(bSG_sAux)),
        bSG_gAux(cute::forward<GTensorS2G>(bSG_gAux)),
        params_ptr(params_ptr) {}

    TiledR2S tiled_r2s;
    RTensor tC_rAux;                                                                   // (CPY,CPY_M,CPY_N)
    STensorR2S tRS_sAux;                                                               // (R2S,R2S_M,R2S_N,PIPE)
    STensorS2G bSG_sAux;                                                               // (S2G,S2G_M,S2G_N,PIPE)
    GTensorS2G bSG_gAux;                                                               // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
    Params const* params_ptr;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));                          // (EPI_V)
      tC_rAux_frg(epi_v) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    postvisit(int epi_m, int epi_n, int store_iteration, bool issue_smem_store) {
      if constexpr (EnableNullptr) {
        if (params_ptr->is_nullptr) {
          return;
        }
      }

      using RLayoutR2S = decltype(cute::layout(TiledR2S{}.get_slice(0).retile_S(RTensor{})));
      Tensor tRS_rAux = make_tensor(tC_rAux.data(), RLayoutR2S{});                                 // (R2S,R2S_M,R2S_N)

      if (issue_smem_store) {
        int store_pipe_index = store_iteration % Stages;
        copy(tiled_r2s, tRS_rAux, tRS_sAux(_,_,_,store_pipe_index));
      }
    }

    CUTLASS_DEVICE void
    step(int epi_m, int epi_n, int store_iteration, bool issue_tma_store) {
      if constexpr (EnableNullptr) {
        if (params_ptr->is_nullptr) {
          return;
        }
      }

      if (issue_tma_store) {
        // Issue the TMA store
        int store_pipe_index = store_iteration % Stages;
        copy(params_ptr->tma_store_aux, bSG_sAux(_,_,_,store_pipe_index), bSG_gAux(_,_,_,epi_m,epi_n));
      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    Tensor mAux = params_ptr->tma_store_aux.get_tma_tensor(make_shape(M,N,L));                               // (M,N,L)
    Tensor gAux = local_tile(mAux, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));                 // (CTA_M,CTA_N)

    Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(                        // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gAux, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tC_rAux = make_tensor<Element>(take<0,3>(shape(tC_gAux)));                  // (CPY,CPY_M,CPY_N)

    Tensor sAux_epi = cute::as_position_independent_swizzle_tensor(
                        make_tensor(make_smem_ptr(smem_aux), SmemLayout{}));     // (EPI_TILE_M,EPI_TILE_N,PIPE)
    Tensor gAux_epi = flat_divide(gAux, args.epi_tile);                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    auto tiled_r2s = conditional_return<ReferenceSrc>(
      make_tiled_copy_S(Copy_Atom<CopyOpR2S,Element>{}, args.tiled_copy),
      make_tiled_copy_D(Copy_Atom<CopyOpR2S,Element>{}, args.tiled_copy)
    );
    auto tRS_sAux = tiled_r2s.get_slice(args.thread_idx).partition_D(sAux_epi);               // (R2S,R2S_M,R2S_N,PIPE)

    ThrCopy thrblk_s2g = params_ptr->tma_store_aux.get_slice(_0{});
    Tensor bSG_sAux = thrblk_s2g.partition_S(sAux_epi);                                // (TMA,TMA_M,TMA_N,PIPE)
    Tensor bSG_gAux = thrblk_s2g.partition_D(gAux_epi);                                // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)

    return ConsumerStoreCallbacks<decltype(tC_rAux), decltype(tiled_r2s), decltype(tRS_sAux), decltype(bSG_sAux), decltype(bSG_gAux)>(
            cute::move(tC_rAux),
            tiled_r2s,
            cute::move(tRS_sAux),
            cute::move(bSG_sAux),
            cute::move(bSG_gAux),
            params_ptr);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Reduction Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Scalar reduction
template <
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_0,_0>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90ScalarReduction {
private:
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_0, _0>>) || // scalar reduction, e.g. tensor max element
    (cute::is_same_v<StrideMNL, Stride<_0,_0, _1>>) || // batched scalar reduction, e.g. per-batch max element
    (cute::is_same_v<StrideMNL, Stride<_0,_0,int>>));
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(IsAtomic, "non-atomic scalar reduction not supported yet");

public:
  struct SharedStorage { };

  struct Arguments {
    ElementOutput* ptr_scalar = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dScalar = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
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
    if constexpr (IsAtomic) {
      auto [M, N, K, L] = problem_shape;
      Layout mScalar_layout = make_layout(make_shape(M,N,L), args.dScalar);
      if (args.ptr_scalar != nullptr) {
        return fill_workspace(args.ptr_scalar, ElementOutput(args.reduction_identity), cosize(mScalar_layout), stream);
      }
    }

    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90ScalarReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90ScalarReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params const params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class CTensor, class ResidueMN>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        int l_coord,
        CTensor tCcScalar,
        ResidueMN residue_mn,
        Params const& params)
      : scalar(params.reduction_identity),
        l_coord(l_coord),
        tCcScalar(tCcScalar),
        residue_mn(residue_mn),
        params(params) {}

    ElementCompute scalar;
    int l_coord;
    CTensor tCcScalar;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ResidueMN residue_mn;
    Params params;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      if constexpr (EnableNullptr) {
        if (params.ptr_scalar == nullptr) {
          return frg_input;
        }
      }

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      Tensor tCcScalar_mn = tCcScalar(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (elem_less(tCcScalar_mn(epi_v * FragmentSize + i), residue_mn)) {
          scalar = reduce_input(scalar, frg_I[i]);
        }
      }

      return frg_input;
    }

    CUTLASS_DEVICE void
    end() {
      if constexpr (EnableNullptr) {
        if (params.ptr_scalar == nullptr) {
          return;
        }
      }

      using ConvertI = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
      using ReduceInput = GmemReduceFn<ElementOutput>;

      ConvertI convert_I{};
      ReduceInput reduce_input{};

      ElementOutput* ptr_scalar = params.ptr_scalar + l_coord * get<2>(params.dScalar);
      reduce_input(ptr_scalar, convert_I(scalar));
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks<decltype(args.tCcD), decltype(args.residue_mn)>(
      get<3>(args.tile_coord_mnkl), args.tCcD, args.residue_mn, params);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  int Stages,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90RowReduction {
private:
  static_assert(Stages == 0, "Smem usage not supported yet");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_1, _0>>) || // row vector reduction, e.g. per-col sum over all batches
    (cute::is_same_v<StrideMNL, Stride<_0,_1,int>>));  // batched row vector reduction, e.g. per-col sum per batch
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(IsAtomic, "non-atomic row reduction not supported yet");

public:
  struct SharedStorage { };

  struct Arguments {
    ElementOutput* ptr_row = nullptr;
    ElementCompute reduction_identity = 0;
    StrideMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
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
    if constexpr (IsAtomic) {
      auto [M, N, K, L] = problem_shape;
      Layout mRow_layout = make_layout(make_shape(M,N,L), args.dRow);
      if (args.ptr_row != nullptr) {
        return fill_workspace(args.ptr_row, ElementOutput(args.reduction_identity), cosize(mRow_layout), stream);
      }
    }

    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90RowReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90RowReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class RTensor, class GTensor, class CTensor, class ResidueMN>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        RTensor&& tCrRow,
        GTensor&& tCgRow,
        CTensor   tCcRow,
        ResidueMN residue_mn,
        Params const& params)
      : tCrRow(cute::forward<RTensor>(tCrRow)),
        tCgRow(cute::forward<GTensor>(tCgRow)),
        tCcRow(tCcRow),
        residue_mn(residue_mn),
        params(params) {}

    // gmem store after every column of subtiles, assuming M-major loop
    // needed to reduce reg pressure, otherwise each thread stores up to a full row in RF
    // since row-elements aren't evenly distributed amongst threads
    RTensor tCrRow;                                                                    // (CPY,CPY_M,CPY_N,EPI_M)
    GTensor tCgRow;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    CTensor tCcRow;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ResidueMN residue_mn;
    Params const& params;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {

      if constexpr (EnableNullptr) {
        if (params.ptr_row == nullptr) {
          return frg_input;
        }
      }

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      Tensor tCrRow_mn = tCrRow(_,_,_,epi_m);
      Tensor tCcRow_mn = tCcRow(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (elem_less(tCcRow_mn(epi_v * FragmentSize + i), residue_mn)) {
          ElementCompute& tCrRow_vmn = tCrRow_mn(epi_v * FragmentSize + i);
          tCrRow_vmn = reduce_input(tCrRow_vmn, frg_I[i]);
        }
      }

      return frg_input;
    }

    CUTLASS_DEVICE void
    step(int epi_m, int epi_n, int store_iteration, bool issue_tma_store) {
      if constexpr (EnableNullptr) {
        if (params.ptr_row == nullptr) {
          return;
        }
      }

      if (epi_m == size<3>(tCrRow)-1) { // assumes M-major subtile loop
        using ConvertI = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        using ReduceInput = GmemReduceFn<ElementOutput>;

        ConvertI convert_I{};
        ReduceInput reduce_input{};

        // Filter so we don't issue redunant copies over stride-0 modes
        Tensor tCrRow_flt = filter_zeros(tCrRow(_,_,_,epi_m));
        Tensor tCgRow_flt = filter_zeros(tCgRow(_,_,_,epi_m,epi_n));
        Tensor tCcRow_mn  = tCcRow(_,_,_,epi_m,epi_n);
        Tensor tCcRow_flt = make_tensor(tCcRow_mn.data(), make_layout(tCgRow_flt.shape(), tCcRow_mn.stride()));


        auto [residue_m, residue_n] = residue_mn;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrRow_flt); ++i) {
          // partially OOB in M must still issue gmem reduction, so only consider residue_n
          // in case last epi tile in column is fully OOB in M and CTA tile is partially OOB in M
          if (residue_n > get<1>(tCcRow_flt(i)) &&
              // fully OOB in M does not need to issue gmem reduction, skip
              residue_m > 0) {
            reduce_input(&tCgRow_flt(i), convert_I(tCrRow_flt(i)));
          }
        }

        // Reset the registers to the reduction identity
        fill(tCrRow, params.reduction_identity);
      }
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    Tensor mRow = make_tensor(make_gmem_ptr(params.ptr_row), make_shape(M,N,L), params.dRow);
    Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mRow, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrRow = make_tensor_like<ElementCompute>(tCgRow(_,_,_,_,_0{}));            // (CPY,CPY_M,CPY_N,EPI_M)
    fill(tCrRow, params.reduction_identity);

    return ConsumerStoreCallbacks<decltype(tCrRow),decltype(tCgRow),decltype(args.tCcD),decltype(args.residue_mn)>(
      cute::move(tCrRow), cute::move(tCgRow), args.tCcD, args.residue_mn, params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Col vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  int Stages,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool EnableNullptr = true, // Noop on nullptr params
  // If this is false, ptr_col is assumed to point to a compact m-major (round_nearest(M,CTA_M), ceil_div(N,CTA_N), L)
  // tensor of ElementCompute. It is the user's responsibility to reduce this to a (M, L) tensor of ElementOutput
  bool FinalReduction = true
>
struct Sm90ColReduction {
private:
  static_assert(Stages == 0, "Smem usage not supported yet");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_1,_0, _0>>) || // col vector reduction, e.g. per-row sum over all batches
    (cute::is_same_v<StrideMNL, Stride<_1,_0,int>>));  // batched col vector reduction, e.g. per-row sum per batch
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(not (IsAtomic && not FinalReduction), "atomic reduction must be final");

public:
  struct SharedStorage { };

  struct Arguments {
    void* ptr_col = nullptr; // ElementOutput* if FinalReduction, else ElementCompute*
    ElementCompute reduction_identity = 0;
    StrideMNL dCol = {};
  };

  struct Params {
    void* ptr_col = nullptr;
    ElementCompute reduction_identity = 0;
    StrideMNL dCol = {};
    ElementCompute* reduction_buffer = nullptr;
    int* tile_counters = nullptr;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    ElementCompute* reduction_buffer;
    int* tile_counters;
    if constexpr (IsAtomic) {
      reduction_buffer = nullptr;
      tile_counters = nullptr;
    }
    else if constexpr (not FinalReduction) {
      reduction_buffer = reinterpret_cast<ElementCompute*>(args.ptr_col);
      tile_counters = nullptr;
    }
    else {
      auto [M, N, K, L] = problem_shape;
      auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
      size_t tile_counters_offset = product(ceil_div(make_shape(M,N,L), make_shape(tile_M, tile_N))) * tile_M * sizeof(ElementCompute);
      tile_counters_offset = round_nearest(tile_counters_offset, sizeof(int));

      reduction_buffer = reinterpret_cast<ElementCompute*>(workspace);
      tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
    }

    return {
      args.ptr_col,
      args.reduction_identity,
      args.dCol,
      reduction_buffer,
      tile_counters
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    if constexpr (IsAtomic || not FinalReduction) {
      return 0;
    }

    size_t workspace_size = 0;
    auto [M, N, K, L] = problem_shape;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};

    // Increment by size of reduction buffer
    workspace_size += product(ceil_div(make_shape(M,N,L), make_shape(tile_M, tile_N))) * tile_M * sizeof(ElementCompute);
    // Align and increment by size of tile counters
    workspace_size = round_nearest(workspace_size, sizeof(int));
    workspace_size += cute::ceil_div(M, tile_M) * sizeof(int);

    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream) {
    if constexpr (IsAtomic) {
      auto [M, N, K, L] = problem_shape;
      Layout mCol_layout = make_layout(make_shape(M,N,L), args.dCol);
      if (args.ptr_col != nullptr) {
        return fill_workspace(args.ptr_col, ElementOutput(args.reduction_identity), cosize(mCol_layout), stream);
      }
      return Status::kSuccess;
    }

    auto [M, N, K, L] = problem_shape;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    size_t tile_counters_offset = product(ceil_div(make_shape(M,N,L), make_shape(tile_M, tile_N))) * tile_M * sizeof(ElementCompute);
    tile_counters_offset = round_nearest(tile_counters_offset, sizeof(int));

    int* tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
    size_t tile_counters_size = cute::ceil_div(M, tile_M) * sizeof(int);
    return zero_workspace(tile_counters, tile_counters_size, stream);
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90ColReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90ColReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
      : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
        params(params) {}

    ArgsTuple args_tuple;
    Params const& params;
    bool do_final_reduction = false;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          return frg_input;
        }
      }

      auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
              lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
              tile_coord_mnkl, residue_mn, epi_tile, tiled_copy, thread_idx] = args_tuple;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);
      Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (elem_less(tCcCol_mn(epi_v * FragmentSize + i), residue_mn)) {
          ElementCompute& tCrCol_vmn = tCrCol_mn(epi_v * FragmentSize + i);
          tCrCol_vmn = reduce_input(tCrCol_vmn, frg_I[i]);
        }
      }

      return frg_input;
    }

    template <class STensor, class SyncFn>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration) {
      if (not is_last_iteration) {
        return;
      }

      auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
              lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
              tile_coord_mnkl, residue_mn, epi_tile, tiled_copy, thread_idx] = args_tuple;
      auto [m, n, k, l] = tile_coord_mnkl;
      constexpr bool ReferenceSrc = decltype(ref_src)::value;

      // Runtime nullptr is noop
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          return;
        }
      }

      // fully OOB CTA in partially OOB cluster
      if (not elem_less(cCol(_0{},_0{}), residue_mn)) {
        return;
      }

      //
      // 1. Warp shuffle reduction
      //
      using FragmentShuffle = Array<ElementCompute, sizeof(uint64_t) / sizeof(ElementCompute)>;
      using ReduceShuffle = RegReduceFn<FragmentShuffle>;
      ReduceShuffle reduce_shuffle{};
      Tensor tCrCol_frg = recast<FragmentShuffle>(filter(tCrCol));
      CUTLASS_PRAGMA_UNROLL
      for (int reduction_cols = size<1>(lane_layout_MN) / 2; reduction_cols > 0; reduction_cols /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int frg_idx = 0; frg_idx < size(tCrCol_frg); ++frg_idx) {
          uint64_t frg_shfl = reinterpret_cast<uint64_t&>(tCrCol_frg(frg_idx));
          frg_shfl = __shfl_down_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(_0{},reduction_cols));
          tCrCol_frg(frg_idx) = reduce_shuffle(tCrCol_frg(frg_idx), reinterpret_cast<FragmentShuffle&>(frg_shfl));
        }
      }
      bool is_reduced_lane = get<1>(lane_mn) == 0;

      //
      // 2. Atomic reduction
      //
      if constexpr (IsAtomic) {
        // Filter so we don't issue redunant copies over stride-0 modes
        Tensor tCrCol_flt = filter_zeros(tCrCol);
        Tensor tCcCol_flt = make_tensor(tCcCol.data(), make_layout(tCrCol_flt.shape(), tCcCol.stride()));

        Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(gCol_l(_,_,l), epi_tile, tiled_copy, thread_idx);
        Tensor tCgCol_flt = filter_zeros(tCgCol);

        // NOTE: atomic reduction is performed in the output type
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        using ReduceOutput = GmemReduceFn<ElementOutput>;
        ConvertOutput convert_output{};
        ReduceOutput reduce_output{};

        if (is_reduced_lane) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrCol_flt); ++i) {
            if (elem_less(tCcCol_flt(i), residue_mn)) {
              reduce_output(&tCgCol_flt(i), convert_output(tCrCol_flt(i)));
            }
          }
        }
        sync_fn();
      }

      //
      // 2. One warp in N, skip threadblock smem reduction
      //
      else if constexpr (decltype(size<1>(warp_layout_MN))::value <= 1) {
        // Dump warp reduction to gmem workspace
        using ElementGmem = conditional_t<FinalReduction, ElementCompute volatile, ElementCompute>;
        Tensor tCgBuf = sm90_partition_for_epilogue<ReferenceSrc>(gBuf_nl(_,_,n,l), epi_tile, tiled_copy, thread_idx);
        if (is_reduced_lane) {
          // Filter so we don't issue redundant copies over stride-0 modes
          // (only works if 0-strides are in same location, which is by construction)
          copy_aligned(filter(tCrCol), recast<ElementGmem>(filter(tCgBuf)));
        }
        sync_fn();
      }

      //
      // 2. Multiple warps in N, do threadblock smem reduction
      //
      else {
        Tensor sBuf = make_tensor(make_smem_ptr<ElementCompute>(raw_pointer_cast(smem_buffer.data())), sBuf_layout);
        static_assert(decltype(cosize(sBuf.layout()))::value * sizeof(ElementCompute) <=
                      decltype(cosize(smem_buffer.layout()))::value * sizeof(typename remove_cvref_t<STensor>::value_type),
                      "smem reduction buffer not large enough, use a larger epilogue tile");

        // Dump warp reduction to smem workspace
        Tensor tCsBuf = sm90_partition_for_epilogue<ReferenceSrc>(sBuf(_,_,get<1>(warp_mn)), epi_tile, tiled_copy, thread_idx);
        if (is_reduced_lane) {
          // Filter so we don't issue redunant copies over stride-0 modes
          // (only works if 0-strides are in same location, which is by construction)
          copy_aligned(filter(tCrCol), filter(tCsBuf));
        }
        sync_fn();

        constexpr int SmemFragSize = cute::max(1, sizeof(uint32_t) / sizeof(ElementCompute));
        using FragmentSmem = Array<ElementCompute, SmemFragSize>;
        using VectorSmem = uint_bit_t<sizeof_bits_v<FragmentSmem>>;
        using ReduceSmem = GmemReduceFn<FragmentSmem>;
        ReduceSmem reduce_smem{};

        Tensor sBuf_frg = recast<FragmentSmem>(filter_zeros(sBuf));
        Tensor sBuf_vec = recast<VectorSmem>(filter_zeros(sBuf));
        constexpr int FragsPerCol = decltype(size<0>(sBuf_frg))::value;

        // Do the threadblock smem reduction
        CUTLASS_PRAGMA_UNROLL
        for (int reduction_cols = size<1>(warp_layout_MN) / 2; reduction_cols > 1; reduction_cols /= 2) {
          int FragsPerReduction = reduction_cols * FragsPerCol;
          CUTLASS_PRAGMA_NO_UNROLL
          for (int frg_idx = thread_idx; frg_idx < FragsPerReduction; frg_idx += size(tiled_copy)) {
            FragmentSmem frg_smem = reduce_smem(sBuf_frg(frg_idx), sBuf_frg(frg_idx + FragsPerReduction));
            sBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem);
          }
          sync_fn();
        }

        // Do final smem reduction and dump to gmem workspace
        using VectorGmem = conditional_t<FinalReduction, VectorSmem volatile, VectorSmem>;
        Tensor gBuf_vec = recast<VectorGmem>(filter(gBuf_nl(_,_,n,l)));
        CUTLASS_PRAGMA_NO_UNROLL
        for (int frg_idx = thread_idx; frg_idx < FragsPerCol; frg_idx += size(tiled_copy)) {
          FragmentSmem frg_smem = reduce_smem(sBuf_frg(frg_idx), sBuf_frg(frg_idx + FragsPerCol));
          gBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem);
        }
        sync_fn();
      }

      //
      // 3. Increment atomic counters to signal final gmem reduction
      //
      if constexpr (not IsAtomic && FinalReduction) {
        // Ensure gmem writes are visible to other threads before incrementing counter
        __threadfence();
        sync_fn();
        // Collective thread 0 increments atomic tile counter and copies value to smem
        int* prev_tile_count = reinterpret_cast<int*>(raw_pointer_cast(smem_buffer.data()));
        if (thread_idx == 0) {
          *prev_tile_count = atomicAdd(&params.tile_counters[m], 1);
        }
        sync_fn();
        // Broadcast tile count to other threads in CTA and determine final reduction status
        do_final_reduction = *prev_tile_count == size<2>(gBuf_nl) * size<3>(gBuf_nl) - 1;
        sync_fn();
      }
    }

    CUTLASS_DEVICE void
    end() {
      //
      // 4. Do final gmem reduction if necessary
      //
      if constexpr (not IsAtomic && FinalReduction) {
        if (not do_final_reduction) {
          return;
        }

        auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
                lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
                tile_coord_mnkl, residue_mn, epi_tile, tiled_copy, thread_idx] = args_tuple;

        using ReduceOutput = GmemReduceFn<ElementCompute>;
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        ReduceOutput reduce_output{};
        ConvertOutput convert_output{};

        // Reduction over batches
        if (size<2>(stride(gCol_l)) == 0) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int m = thread_idx; m < size<0>(gBuf_nl); m += size(tiled_copy)) {
            Tensor tRgBuf_nl = gBuf_nl(m,_0{},_,_);
            ElementCompute output = tRgBuf_nl(_0{});
            CUTLASS_PRAGMA_NO_UNROLL
            for (int nl = 1; nl < size(tRgBuf_nl); ++nl) {
              output = reduce_output(output, tRgBuf_nl(nl));
            }
            if (elem_less(cCol(m,_0{}), residue_mn)) {
              gCol_l(m,_0{},_0{}) = convert_output(output);
            }
          }
        }
        // No reduction over batches
        else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int m = thread_idx; m < size<0>(gBuf_nl); m += size(tiled_copy)) {
            bool do_store = elem_less(cCol(m,_0{}), residue_mn);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int l = 0; l < size<3>(gBuf_nl); ++l) {
              Tensor tRgBuf_n = gBuf_nl(m,_0{},_,l);
              ElementCompute output = tRgBuf_n(_0{});
              CUTLASS_PRAGMA_NO_UNROLL
              for (int n = 1; n < size(tRgBuf_n); ++n) {
                output = reduce_output(output, tRgBuf_n(n));
              }
              if (do_store) {
                gCol_l(m,_0{},l) = convert_output(output);
              }
            }
          }
        }

      }
    }

    CUTLASS_DEVICE bool
    is_reduction_buffer_needed(int epi_m, int epi_n, bool is_last_iteration) const {
      auto const& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
                    lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
                    tile_coord_mnkl, residue_mn, epi_tile, tiled_copy, thread_idx] = args_tuple;

      return (not IsAtomic &&                                  // atomic reduction doesn't use smem
              is_last_iteration &&                             // smem reduction happens after epilogue loop
              (decltype(size<1>(warp_layout_MN))::value > 1 || // smem reduction happens when multiple warps are in N
               FinalReduction));                               // smem is used to broadcast tile counters for final reduction
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Layout ref_layout_MN = [&] () {
      if constexpr (ReferenceSrc) { return get<0>(args.tiled_copy.get_layoutS_MN()); }
      else                        { return get<0>(args.tiled_copy.get_layoutD_MN()); }
    }();                                                                                         // tile_mn -> tv_idx

    // Get the MN layout + coord of lanes to determine shuffle reduction iterations
    using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
    Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};            //   tv_idx -> lane_idx
    Layout ref2lane = composition(tv2lane, ref_layout_MN);                                      //  tile_mn -> lane_idx
    Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));    //  lane_mn -> lane_idx
    Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);                                  // lane_idx -> lane_mn
    int lane_idx = canonical_lane_idx();
    auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

    // Get the MN layout + coord of warps to determine smem reduction iterations
    Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};            //   tv_idx -> warp_idx
    Layout ref2warp = composition(tv2warp, ref_layout_MN);                                      //  tile_mn -> warp_idx
    Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));    //  warp_mn -> warp_idx
    Layout inv_warp_layout_MN = right_inverse(warp_layout_MN);                                  // warp_idx -> warp_mn
    int warp_idx = args.thread_idx / NumThreadsPerWarp;
    auto warp_mn = idx2crd(inv_warp_layout_MN(warp_idx), shape(warp_layout_MN));

    // Partition output gmem and register tensors
    auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    Tensor mCol = make_tensor(make_gmem_ptr<ElementOutput>(params.ptr_col), make_shape(M,N,L), params.dCol); // (M,N,L)
    Tensor gCol_l = local_tile(mCol, take<0,2>(args.tile_shape_mnk), make_coord(m,n,_));             // (CTA_M,CTA_N,L)
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gCol_l(_,_,l), args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like<ElementCompute>(tCgCol);                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    fill(tCrCol, params.reduction_identity);

    // Partition gmem+smem reduction buffer tensors
    Layout gBuf_layout = make_layout(take<0,2>(args.tile_shape_mnk), make_stride(_1{}, _0{}));
    Layout mBuf_layout = blocked_product(gBuf_layout, make_layout(ceil_div(make_shape(M,N,L), shape(gBuf_layout))));
    Tensor mBuf = make_tensor(make_gmem_ptr(params.reduction_buffer), mBuf_layout);                // (ceil_M,ceil_N,L)
    Tensor gBuf_nl = local_tile(mBuf, take<0,2>(args.tile_shape_mnk), make_coord(m,_,_));     // (CTA_M,CTA_N,REST_N,L)
    Layout sBuf_layout = blocked_product(gBuf_layout,make_layout(make_shape(_1{},_1{},size<1>(warp_layout_MN)))); // (CTA_M,CTA_N,WARPS_N)

    auto args_tuple = make_tuple(
        bool_constant<ReferenceSrc>{}, cute::move(tCrCol), args.tCcD, gCol_l, args.cD, gBuf_nl, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        args.tile_coord_mnkl, args.residue_mn, args.epi_tile, args.tiled_copy, args.thread_idx);
    return ConsumerStoreCallbacks<decltype(args_tuple)>(std::move(args_tuple), params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Batch matrix reduction
template <
  int Stages,
  class EpilogueTile,
  class Element,
  class StrideMNL,
  class CopyOpR2S,
  class SmemLayoutAtom,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90MatrixReduction;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
