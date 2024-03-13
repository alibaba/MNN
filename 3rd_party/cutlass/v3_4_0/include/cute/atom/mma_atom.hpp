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

#include <cute/config.hpp>

#include <cute/arch/mma.hpp>

#include <cute/atom/mma_traits.hpp>

#include <cute/tensor.hpp>
#include <cute/util/type_traits.hpp>

namespace cute {

template <class... Args>
struct MMA_Atom;

template <class MMAOperation>
struct MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>
{};

template <class... Args>
struct MMA_Atom<MMA_Traits<Args...>>
  : MMA_Traits<Args...>
{
  using Traits = MMA_Traits<Args...>;

  // Element value types from the MMA_Traits
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  // Thr-Val layouts from the MMA_Traits
  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // Fragment value types from the MMA_Traits (optional, defaults to Val type)
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(std::forward<TraitsArgs>(args)...);
    return MMA_Atom<decltype(traits)>{traits};
  }

  //
  // Tensor call interfaces
  //

  // Cast, check, and call fma
  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C) const
  {
    static_assert(DLayout::rank == 1, "Expected rank-1 D tensor");
    static_assert(ALayout::rank == 1, "Expected rank-1 A tensor");
    static_assert(BLayout::rank == 1, "Expected rank-1 B tensor");
    static_assert(CLayout::rank == 1, "Expected rank-1 C tensor");

    return mma_unpack(*this, D, A, B, C);
  }

  // Three arguments reproduces C
  template <class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout>      & C) const
  {
    return call(C, A, B, C);
  }

  //
  // make_fragment_A|B|C
  //   These functions are awkward as they expect already-partitioned tensors
  //     resulting from a previous call to partition_A|B|C
  //   The reasoning is that we can inspect the layout of the partitioned data
  //     and attempt to match it in generated fragment to promote vectorization
  //     when copying from partition to fragment.
  //

  template <class CTensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_C(CTensor&& ctensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<3>{});  // VMN
    CUTE_STATIC_ASSERT_V(size<0>(ctensor) == size<1>(LayoutC_TV{}));
    // C is a bit special because we are after accumulators here
    // The input/output type doesn't have to match the accumulator type
    //static_assert(std::is_same<ValTypeC, typename remove_cvref_t<CTensor>::value_type>::value, "Expecting ValTypeC type");

    // We'll never base the accumulator layout on the input tensor layout, so just return a FrgTypeC tensor
    return make_tensor<FrgTypeC>(shape(ctensor));
  }

  template <class ATensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_A(ATensor&& atensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<3>{});  // VMK
    CUTE_STATIC_ASSERT_V(size<0>(atensor) == size<1>(LayoutA_TV{}));

    if constexpr (has_dereference<FrgTypeA>::value) {
      // If the intended FrgTypeA is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeA, typename remove_cvref_t<ATensor>::value_type>::value
                      , "Expecting ValTypeA type");
      return make_tensor<FrgTypeA>(std::forward<ATensor>(atensor));
    } else {
      // Else, the intended FrgTypeA is a value type, construct a new tensor with a fragment layout
      return make_fragment_like<FrgTypeA>(atensor);
    }

    CUTE_GCC_UNREACHABLE;
  }

  template <class BTensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_B(BTensor&& btensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(btensor) >= Int<3>{});  // VNK
    CUTE_STATIC_ASSERT_V(size<0>(btensor) == size<1>(LayoutB_TV{}));

    if constexpr (has_dereference<FrgTypeB>::value) {
      // If the intended FrgTypeB is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeB, typename remove_cvref_t<BTensor>::value_type>::value
                      , "Expecting ValTypeB type");
      return make_tensor<FrgTypeB>(std::forward<BTensor>(btensor));
    } else {
      // Else, the intended FrgTypeB is a value type, construct a new tensor with a fragment layout
      return make_fragment_like<FrgTypeB>(btensor);
    }

    CUTE_GCC_UNREACHABLE;
  }
};

//
// A tiling of mma atoms
//

template <class TiledMMA, class ThrCoord>
struct ThrMMA;

// @tparam MMA_Atom The MMA_Atom to use in the TiledMMA
// @tparam AtomLayoutMNK The MNK-tiling of the Atom to be performed.
// @tparam PermuationsMNK Permutations to apply to each MNK-mode before tiling for the Atom.
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom
{
  using Atom           = MMA_Atom;
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;
  using AtomThrID      = typename MMA_Atom::ThrID;
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
  using AtomLayoutA_TV = typename MMA_Atom::LayoutA_TV;
  using AtomLayoutB_TV = typename MMA_Atom::LayoutB_TV;

  static_assert(   rank_v<AtomLayoutMNK>  == 3,   "TiledMMA requires rank-3 AtomLayoutMNK");
  static_assert(   rank_v<PermutationMNK> == 3,   "TiledMMA requires rank-3 PermutationMNK");
  static_assert(  is_tile<PermutationMNK>::value, "TiledMMA requires independent permutations of MNK.");
  static_assert(is_static<PermutationMNK>::value, "TiledMMA requires static permutations of MNK.");

  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_;

  CUTE_HOST_DEVICE constexpr
  TiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
    : MMA_Atom(mma_atom),
      thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}

  CUTE_HOST_DEVICE constexpr auto
  get_thr_layout_vmnk() const {
    return thr_layout_vmnk_;
  }

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))
  // where
  //   ThrV:  The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrM:  The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
  //   ThrN:  The threads tiled in N.      layout<2>(ThrLayoutVMNK): ThrN -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_C(CTensor&& ctensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<2>{});
    //CUTE_STATIC_ASSERT_V(size<0>(ctensor) % size<0>(TiledShape_MNK{}) == Int<0>{});
    //CUTE_STATIC_ASSERT_V(size<1>(ctensor) % size<1>(TiledShape_MNK{}) == Int<0>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}),
                            get<1>(PermutationMNK{}));
    auto t_tensor = logical_divide(ctensor, t_tile);                 // (PermM,PermN)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<1>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomN),(RestM,RestN))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutC_TV{},_);           // ((ThrV,FrgV),(RestM,RestN))

    // Tile the tensor for the C-threads
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                        make_layout(size<2>(thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))

    return thr_tensor;
  }

  // Tile a tensor or a layout from shape
  //   (M,K,...)
  // to shape
  //   ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
  // where
  //   ThrV: The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrM: The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
  //   ThrK: The threads tiled in K.      layout<3>(ThrLayoutVMNK): ThrK -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestM: The values tiled in M.
  //   RestK: The values tiled in K.
  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_A(ATensor&& atensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<2>{});
    //CUTE_STATIC_ASSERT_V(size<0>(atensor) % size<0>(TiledShape_MNK{}) == Int<0>{});
    //CUTE_STATIC_ASSERT_V(size<1>(atensor) % size<2>(TiledShape_MNK{}) == Int<0>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}),
                            get<2>(PermutationMNK{}));
    auto t_tensor = logical_divide(atensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                        make_layout(size<3>(thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  // Tile a tensor or a layout from shape
  //   (N,K,...)
  // to shape
  //   ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...)))
  // where
  //   ThrV: The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrN: The threads tiled in N.      layout<2>(ThrLayoutVMNK): ThrN -> thread_idx
  //   ThrK: The threads tiled in K.      layout<3>(ThrLayoutVMNK): ThrK -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestN: The values tiled in N.
  //   RestK: The values tiled in K.
  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_B(BTensor&& btensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(btensor) >= Int<2>{});
    //CUTE_STATIC_ASSERT_V(size<0>(btensor) % size<1>(TiledShape_MNK{}) == Int<0>{});
    //CUTE_STATIC_ASSERT_V(size<1>(btensor) % size<2>(TiledShape_MNK{}) == Int<0>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(PermutationMNK{}),
                            get<2>(PermutationMNK{}));
    auto t_tensor = logical_divide(btensor, t_tile);                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk_)),
                                        make_layout(size<3>(thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))

    return thr_tensor;
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_slice(ThrIdx const& thr_idx) const
  {
    auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
    return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_thread_slice(ThrIdx const& thr_idx) const
  {
    return get_slice(thr_idx);
  }

  //
  // Utility for printing and visualization
  //

  // The size of the MNK-mode
  template <int I>
  CUTE_HOST_DEVICE constexpr
  auto
  tile_size_mnk() const {
    static_assert(0 <= I && I < 3);
    auto core_size = size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk());
    [[maybe_unused]] auto perm_size = size<I>(PermutationMNK{});
    if constexpr (is_underscore<decltype(perm_size)>::value) {
      return core_size;
    } else {
      return cute::max(core_size, perm_size);
    }
  }

  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutC_MN() const
  {
    // (M,N) -> (M,N)
    auto ref_C = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<1>()));
    // (cthrid,val) -> (M,N)
    auto layoutC_TV = thrfrg_C(ref_C);
    // (M,N) -> (cthrid,frg)
    auto layoutC_MN = right_inverse(layoutC_TV).with_shape(shape(ref_C));

    // cthrid = (v,m,n) -> thr_idx
    auto thrID_C = thr_layout_vmnk_(_,_,_,Int<0>{});

    return cute::make_tuple(layoutC_MN, thrID_C);
  }

  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutC_TV() const
  {
    // (M,N) -> (M,N)
    auto ref_C = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<1>()));
    // (cthrid,val) -> (M,N)
    auto layoutC_TV = thrfrg_C(ref_C);

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk_);

    // (thr_idx,val) -> (M,N)
    return layoutC_TV.compose(thridx_2_thrid, _);
  }

  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutA_MK() const
  {
    // (M,K) -> (M,K)
    auto ref_A = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<2>()));
    // (athrid,val) -> (M,K)
    auto layoutA_TV = thrfrg_A(ref_A);
    // (M,K) -> (athrid,frg)
    auto layoutA_MK = right_inverse(layoutA_TV).with_shape(shape(ref_A));

    // athrid = (v,m,k) -> thr_idx
    auto thrID_A = thr_layout_vmnk_(_,_,Int<0>{},_);

    return cute::make_tuple(layoutA_MK, thrID_A);
  }

  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutA_TV() const
  {
    // (M,K) -> (M,K)
    auto ref_A = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<2>()));
    // (athrid,val) -> (M,K)
    auto layoutA_TV = thrfrg_A(ref_A);

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(_,
                           make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk_), size<2>(thr_layout_vmnk_)),
                                                 make_stride(               Int<1>{} ,                Int<0>{} )),
                                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk_);

    // (thr_idx,val) -> (M,K)
    return thrfrg_A(ref_A).compose(atile, _).compose(thridx_2_thrid, _);
  }

  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutB_NK() const
  {
    // (N,K) -> (N,K)
    auto ref_B = make_layout(make_shape(tile_size_mnk<1>(), tile_size_mnk<2>()));
    // (bthrid,val) -> (N,K)
    auto layoutB_TV = thrfrg_B(ref_B);
    // (N,K) -> (bthrid,frg)
    auto layoutB_NK = right_inverse(layoutB_TV).with_shape(shape(ref_B));

    // bthrid = (v,n,k) -> thr_idx
    auto thrID_B = thr_layout_vmnk_(_,Int<0>{},_,_);

    return cute::make_tuple(layoutB_NK, thrID_B);
  }

  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutB_TV() const
  {
    // (N,K) -> (N,K)
    auto ref_B = make_layout(make_shape(tile_size_mnk<1>(), tile_size_mnk<2>()));
    // (bthrid,val) -> (N,K)
    auto layoutB_TV = thrfrg_B(ref_B);

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(_,
                           make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk_), size<2>(thr_layout_vmnk_)),
                                                 make_stride(               Int<0>{} ,                Int<1>{} )),
                                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk_);

    // (thr_idx,val) -> (N,K)
    return thrfrg_B(ref_B).compose(btile, _).compose(thridx_2_thrid, _);
  }
};

template <class TiledMMA, class ThrVMNK>
struct ThrMMA : TiledMMA
{
  ThrVMNK thr_vmnk_;

  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_C(CTensor&& ctensor) const
  {
    auto thr_tensor = make_tensor(std::forward<CTensor>(ctensor).data(), this->thrfrg_C(ctensor.layout()));

    auto thr_vmn = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
    return thr_tensor(thr_vmn, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_A(ATensor&& atensor) const
  {
    auto thr_tensor = make_tensor(std::forward<ATensor>(atensor).data(), this->thrfrg_A(atensor.layout()));

    auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));
    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_B(BTensor&& btensor) const
  {
    auto thr_tensor = make_tensor(std::forward<BTensor>(btensor).data(), this->thrfrg_B(btensor.layout()));

    auto thr_vnk = make_coord(get<0>(thr_vmnk_), make_coord(get<2>(thr_vmnk_), get<3>(thr_vmnk_)));
    return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_C(CTensor&& ctensor) const
  {
    return TiledMMA::make_fragment_C(partition_C(ctensor));
  }

  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_A(ATensor&& atensor) const
  {
    return TiledMMA::make_fragment_A(partition_A(atensor));
  }

  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_B(BTensor&& btensor) const
  {
    return TiledMMA::make_fragment_B(partition_B(btensor));
  }
};

//
// These tile the MMA_Atom as a whole
//

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
{
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});
  auto permutation_mnk = append<3>(permutations, _);

  return TiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}

//
// partition_fragment_C -- static context
//

template <class... Args, class Shape_MN>
CUTE_HOST_DEVICE constexpr
auto
partition_shape_C(TiledMMA<Args...> const& mma, Shape_MN const& shape_MN)
{
  constexpr int R = rank_v<Shape_MN>;
  static_assert(R >= 2, "Must have at least rank-2");
  auto atomMNK = typename TiledMMA<Args...>::AtomShape_MNK{};
  auto thrVMNK = typename TiledMMA<Args...>::ThrLayoutVMNK{};
  auto V = shape<1>(typename TiledMMA<Args...>::AtomLayoutC_TV{});
  auto M = shape_div(size<0>(shape_MN), size<0>(atomMNK) * size<1>(thrVMNK));
  auto N = shape_div(size<1>(shape_MN), size<1>(atomMNK) * size<2>(thrVMNK));
  return cute::tuple_cat(make_shape(V,M,N), take<2,R>(shape_MN));
}

template <class... Args, class Shape_MN>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_C(TiledMMA<Args...> const& mma, Shape_MN const& shapeMN)
{
  return make_tensor<typename TiledMMA<Args...>::FrgTypeC>(partition_shape_C(mma, shapeMN));
}

// partition_fragment_A and partition_fragment_B often depend on the
//   layout of A and B and/or the thread_idx that is requesting the partition.
// For these reasons, they should not be used in a static context.
// See TiledMMA::get_slice(thr_idx).partition_fragment_A(tensorA) instead.

template <class... Args, class Shape_MK>
CUTE_HOST_DEVICE constexpr
auto
partition_shape_A(TiledMMA<Args...> const& mma, Shape_MK const& shape_MK)
{
  constexpr int R = rank_v<Shape_MK>;
  static_assert(R >= 2, "Must have at least rank-2");
  auto atomMNK = typename TiledMMA<Args...>::AtomShape_MNK{};
  auto thrVMNK = typename TiledMMA<Args...>::ThrLayoutVMNK{};
  auto V = shape<1>(typename TiledMMA<Args...>::AtomLayoutA_TV{});
  auto M = shape_div(size<0>(shape_MK), size<0>(atomMNK) * size<1>(thrVMNK));
  auto K = shape_div(size<1>(shape_MK), size<2>(atomMNK) * size<3>(thrVMNK));
  return cute::tuple_cat(make_shape(V,M,K), take<2,R>(shape_MK));
}

template <class... Args, class Shape_NK>
CUTE_HOST_DEVICE constexpr
auto
partition_shape_B(TiledMMA<Args...> const& mma, Shape_NK const& shape_NK)
{
  constexpr int R = rank_v<Shape_NK>;
  static_assert(R >= 2, "Must have at least rank-2");
  auto atomMNK = typename TiledMMA<Args...>::AtomShape_MNK{};
  auto thrVMNK = typename TiledMMA<Args...>::ThrLayoutVMNK{};
  auto V = shape<1>(typename TiledMMA<Args...>::AtomLayoutB_TV{});
  auto N = shape_div(size<0>(shape_NK), size<1>(atomMNK) * size<2>(thrVMNK));
  auto K = shape_div(size<1>(shape_NK), size<2>(atomMNK) * size<3>(thrVMNK));
  return cute::tuple_cat(make_shape(V,N,K), take<2,R>(shape_NK));
}

//
// Size
//

template <int I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_size(TiledMMA<Args...> const& mma)
{
  return mma.template tile_size_mnk<I>();
}

template <class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_shape(TiledMMA<Args...> const& mma)
{
  return make_shape(tile_size<0>(mma), tile_size<1>(mma), tile_size<2>(mma));
}

// Deprecate?
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
size(TiledMMA<Args...> const& mma)
{
  return size<I...>(mma.get_thr_layout_vmnk());
}

// Alias
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
thr_size(TiledMMA<Args...> const& mma)
{
  return size<I...>(mma.get_thr_layout_vmnk());
}

//
// Display utilities
//

template <class... Args>
CUTE_HOST_DEVICE
void
print(MMA_Atom<MMA_Traits<Args...>> const&)
{
  using Atom = MMA_Atom<MMA_Traits<Args...>>;
  print("MMA_Atom\n");
  print("  ThrID:      "); print(typename Atom::ThrID{});      print("\n");
  print("  LayoutA_TV: "); print(typename Atom::LayoutA_TV{}); print("\n");
  print("  LayoutB_TV: "); print(typename Atom::LayoutB_TV{}); print("\n");
  print("  LayoutC_TV: "); print(typename Atom::LayoutC_TV{}); print("\n");
}

template <class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE
void
print(TiledMMA<Atom, TiledThr, TiledPerm> const& mma)
{
  print("TiledMMA\n");
  print("  ThrLayoutVMNK:  "); print(mma.get_thr_layout_vmnk());  print("\n");
  print("  PermutationMNK: "); print(TiledPerm{}); print("\n");
  print(static_cast<Atom const&>(mma));
}

template <class TiledMMA, class ThrVMNK>
CUTE_HOST_DEVICE
void
print(ThrMMA<TiledMMA, ThrVMNK> const& thr_mma)
{
  print("ThrMMA\n");
  print("  Thr VMNK: "); print(thr_mma.thr_vmnk_); print("\n");
  print(static_cast<TiledMMA>(thr_mma));
}

template <class... Args>
CUTE_HOST_DEVICE
auto
print_latex(TiledMMA<Args...> const& mma)
{
  auto layout_and_thrid_C = mma.get_layoutC_MN();
  auto layoutC_MN = get<0>(layout_and_thrid_C);
  auto thrID_C    = get<1>(layout_and_thrid_C);

  auto layout_and_thrid_A = mma.get_layoutA_MK();
  auto layoutA_MK = get<0>(layout_and_thrid_A);
  auto thrID_A    = get<1>(layout_and_thrid_A);

  auto layout_and_thrid_B = mma.get_layoutB_NK();
  auto layoutB_NK = get<0>(layout_and_thrid_B);
  auto thrID_B    = get<1>(layout_and_thrid_B);

  print_latex_mma(layoutC_MN, thrID_C,
                  layoutA_MK, thrID_A,
                  layoutB_NK, thrID_B);
}

// MNK MMA Layout to console printer -- 8-value color coded by thread
template <class LayoutC, class ThrIDC,
          class LayoutA, class ThrIDA,
          class LayoutB, class ThrIDB>
CUTE_HOST_DEVICE
void
print_layout_mma(LayoutC const& C, ThrIDC const& TC,  // (m,n) -> (tid,vid)  and  tid -> thr_idx
                 LayoutA const& A, ThrIDA const& TA,  // (m,k) -> (tid,vid)  and  tid -> thr_idx
                 LayoutB const& B, ThrIDB const& TB)  // (n,k) -> (tid,vid)  and  tid -> thr_idx
{
  CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

  assert(size<0>(A) == size<0>(C));
  assert(size<0>(B) == size<1>(C));
  assert(size<1>(A) == size<1>(B));

  int a_width = size<1>(A) * 6 + 4;

  // Print out B (white-shifted) k-by-n
  for (int k = 0; k < size<1>(B); ++k) {
    // Header
    printf("%*s", a_width, "");
    for (int n = 0; n < size<0>(B); ++n) printf("+-----");
    printf("+\n");
    // Values
    printf("%*s", a_width, "");
    for (int n = 0; n < size<0>(B); ++n) printf("|T%02dV%1d", int(TB(B(n,k) % size(TB))), int(B(n,k) / size(TB)));
    printf("|\n");
  }
  // Footer
  printf("%*s", a_width, "");
  for (int n = 0; n < size<0>(B); ++n) printf("+-----");
  printf("+\n\n");

  // Print out A m-by-k and C m-by-n
  for (int m = 0; m < size<0>(A); ++m) {
    // Header
    for (int k = 0; k < size<1>(A); ++k) printf("+-----");
    printf("+   ");
    for (int n = 0; n < size<1>(C); ++n) printf("+-----");
    printf("+\n");
    // Values
    for (int k = 0; k < size<1>(A); ++k) printf("|T%02dV%1d", int(TA(A(m,k) % size(TA))), int(A(m,k) / size(TA)));
    printf("|   ");
    for (int n = 0; n < size<1>(C); ++n) printf("|T%02dV%1d", int(TC(C(m,n) % size(TC))), int(C(m,n) / size(TC)));
    printf("|\n");
  }
  // Footer
  for (int k = 0; k < size<1>(A); ++k) printf("+-----");
  printf("+   ");
  for (int n = 0; n < size<1>(C); ++n) printf("+-----");
  printf("+\n");
}

// MNK MMA Layout to Latex TIKZ -- 8-value color coded by thread
template <class LayoutC, class ThrIDC,
          class LayoutA, class ThrIDA,
          class LayoutB, class ThrIDB>
CUTE_HOST_DEVICE
void
print_latex_mma(LayoutC const& C, ThrIDC const& TC,  // (m,n) -> (tid,vid)  and  tid -> thr_idx
                LayoutA const& A, ThrIDA const& TA,  // (m,k) -> (tid,vid)  and  tid -> thr_idx
                LayoutB const& B, ThrIDB const& TB)  // (n,k) -> (tid,vid)  and  tid -> thr_idx
{
  CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

  assert(size<0>(A) == size<0>(C));
  assert(size<0>(B) == size<1>(C));
  assert(size<1>(A) == size<1>(B));

  char const* latex_header =
      "\\documentclass{standalone}\n"
      "\\usepackage{tikz}\n"
      "\\usetikzlibrary{external}\n"
      "\\tikzexternalize\n"
      "\\begin{document}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/.style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n";
  char const* latex_footer =
      "\\end{tikzpicture}\n"
      "\\end{document}\n";

  char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                              "{rgb,255:red,175;green,255;blue,175}",
                              "{rgb,255:red,255;green,255;blue,175}",
                              "{rgb,255:red,255;green,175;blue,175}",
                              "{rgb,255:red,210;green,210;blue,255}",
                              "{rgb,255:red,210;green,255;blue,210}",
                              "{rgb,255:red,255;green,255;blue,210}",
                              "{rgb,255:red,255;green,210;blue,210}"};

  // Header
  printf("%% LayoutC: "); print(C);  printf("\n");
  printf("%% ThrIDC : "); print(TC); printf("\n");
  printf("%% LayoutA: "); print(A);  printf("\n");
  printf("%% ThrIDA : "); print(TA); printf("\n");
  printf("%% LayoutB: "); print(B);  printf("\n");
  printf("%% ThrIDB : "); print(TB); printf("\n\n");

  printf(latex_header);

  // C starting at 0,0
  for (int m = 0; m < size<0>(C); ++m) {
    for (int n = 0; n < size<1>(C); ++n) {
      int thrid   = C(m,n) % size(TC);
      int val_idx = C(m,n) / size(TC);
      int thr_idx = TC(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             m, n,
             thr_idx, val_idx);
    }
  }

  // A starting at 0,-size<1>(A)-1
  for (int m = 0; m < size<0>(A); ++m) {
    for (int k = 0; k < size<1>(A); ++k) {
      int thrid   = A(m,k) % size(TA);
      int val_idx = A(m,k) / size(TA);
      int thr_idx = TA(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             m, k-1-size<1>(A),
             thr_idx, val_idx);
    }
  }

  // B starting at -size<1>(B)-1,0
  for (int n = 0; n < size<0>(B); ++n) {
    for (int k = 0; k < size<1>(B); ++k) {
      int thrid   = B(n,k) % size(TB);
      int val_idx = B(n,k) / size(TB);
      int thr_idx = TB(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             k-1-size<1>(B), n,
             thr_idx, val_idx);
    }
  }

  // A labels
  for (int m = 0, k = -1; m < size<0>(A); ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), m);
  }
  for (int k = 0, m = -1; k < size<1>(A); ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), k);
  }
  // B labels
  for (int n = 0, k = -1; n < size<0>(B); ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, n);
  }
  for (int k = 0, n = -1; k < size<1>(B); ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, k);
  }

  // Footer
  printf(latex_footer);
}

} // namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cute/atom/mma_traits_sm61.hpp>
#include <cute/atom/mma_traits_sm70.hpp>
#include <cute/atom/mma_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm90.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
////////////////////////////////////////////////////////////////////////////////////////////////////
