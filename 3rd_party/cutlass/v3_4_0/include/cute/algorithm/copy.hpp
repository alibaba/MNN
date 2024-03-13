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

#include <cute/container/alignment.hpp>

#include <cute/tensor.hpp>
#include <cute/tensor_predicate.hpp>

#include <cute/atom/copy_atom.hpp>

namespace cute
{

//
// Accept mutable temporaries
//

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy(src, dst);
}

template <class VecType,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
         Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_vec<VecType>(src, dst);
}

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_aligned(Tensor<SrcEngine, SrcLayout> const& src,
             Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_aligned(src, dst);
}

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_if(pred, src, dst);
}

template <class CopyPolicy,
          class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(CopyPolicy                   const& copy_policy,
        PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_if(copy_policy, pred, src, dst);
}

template <class CopyPolicy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(CopyPolicy                   const& copy_policy,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy(copy_policy, src, dst);
}

//
// copy_if -- Predicated Copy
//

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst)
{
  auto copy_op = select_elementwise_copy(src, dst);

  CUTE_UNROLL
  for (int i = 0; i < size(src); ++i) {
    if (pred(i)) {
      copy_op.copy(src(i), dst(i));
    }
  }
}

//
// copy_if -- Predicated CopyAtom
//

namespace detail {

// Trait that detects if atom's traits has a member function with(bool)
template<typename, typename Enable = void>
constexpr bool has_with_bool = false;

template<typename T>
constexpr bool has_with_bool<T, cute::void_t<decltype(declval<typename T::Traits>().with(declval<bool>()))>> = true;

} // end namespace detail

template <class... CopyArgs,
          class PredTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(Copy_Atom<CopyArgs...>       const& copy_atom,
        PredTensor                   const& pred,      // (Rest...)
        Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
        Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");
  if constexpr (SrcLayout::rank == 1) {   // Dispatch the copy
    copy_atom.call(src, dst);
  } else {                                // Loop over all but the first mode
    constexpr int R = SrcLayout::rank;
    Tensor src_v = group_modes<1,R>(src);
    Tensor dst_v = group_modes<1,R>(dst);
    CUTE_UNROLL
    for (int i = 0; i < size<1>(src_v); ++i) {
      // If copy traits can be transformed with a predicate value, do it, otherwise branch here
      if constexpr (detail::has_with_bool<Copy_Atom<CopyArgs...>>) {
        copy_atom.with(pred(i)).call(src_v(_,i), dst_v(_,i));
      } else {
        if (pred(i)) {
          copy_atom.call(src_v(_,i), dst_v(_,i));
        }
      }
    }
  }
}

//
// copy_vec -- attempt vectorized copy with VecType
//

template <class VecType,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
         Tensor<DstEngine, DstLayout>      & dst)
{
  static_assert(sizeof_bits_v<VecType> >= 8 && sizeof_bits_v<VecType> % 8 == 0,
                "Expected a vectorization type of at least a byte.");
  using SrcType = typename SrcEngine::element_type;
  using DstType = typename DstEngine::element_type;
  if constexpr (sizeof_bits_v<SrcType> == sizeof_bits_v<DstType> &&
                sizeof_bits_v<VecType>  > sizeof_bits_v<DstType>)
  {
    // Preserve volatility of Src/Dst types.
    using SrcVecType = conditional_t<is_volatile_v<SrcType>, VecType const volatile, VecType const>;
    using DstVecType = conditional_t<is_volatile_v<DstType>, VecType       volatile, VecType      >;
    Tensor src_v = recast<SrcVecType>(src);
    Tensor dst_v = recast<DstVecType>(dst);

#if 0
    if (thread0()) {
      print("copy_vec<%db> -- vectorizing copy:\n", int(sizeof_bits_v<VecType>));
      print("   "); print(src); print(" => "); print(src_v); print("\n");
      print("   "); print(dst); print(" => "); print(dst_v); print("\n");
    }
#endif

    return copy_if(TrivialPredTensor{}, src_v, dst_v);
  } else {
#if 0
  if (thread0()) {
    print("copy_vec<%db> -- NOT vectorizing copy:\n", int(sizeof_bits_v<VecType>));
    print("   "); print(src); print("\n");
    print("   "); print(dst); print("\n");
  }
#endif

    return copy_if(TrivialPredTensor{}, src, dst);
  }
}

//
// copy -- CopyAtom
//

template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  return copy_if(copy_atom, TrivialPredTensor{}, src, dst);
}

//////////////////////////////////////////
// Special Auto-Vectorizing Overloads
//////////////////////////////////////////

// Specialization for AutoVectorizingCopyAssumedAlignment<MaxVecBits>
template <int MaxVecBits, class... Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(AutoVectorizingCopyWithAssumedAlignment<MaxVecBits> const&,
     Tensor<SrcEngine, SrcLayout>                        const& src,
     Tensor<DstEngine, DstLayout>                             & dst)
{
  constexpr int vec_elem = decltype(max_common_vector(src, dst))::value;

  constexpr int src_bits = sizeof_bits<typename SrcEngine::value_type>::value;
  // When layouts are static,  accept vec_bits up to 128
  // When layouts are dynamic, accept vec_bits up to MaxVecBits
  constexpr int vec_bits = (is_static<SrcLayout>::value && is_static<DstLayout>::value) ?
                            cute::min(vec_elem * src_bits, 128) :
                            cute::min(vec_elem * src_bits, MaxVecBits);

#if 0
  if (thread0()) {
    print("copy -- found max_common_vector of %d elems and vectorization to %d bits\n", vec_elem, vec_bits);
    print("   "); print(src); print("\n");
    print("   "); print(dst); print("\n");
  }
#endif

  if constexpr (vec_elem > 1 && vec_bits >= 8) {
    return copy_vec<uint_bit_t<vec_bits>>(src, dst);
  } else {
    return copy_if(TrivialPredTensor{}, src, dst);
  }
}

// Auto-vectorizing copy for static layouts
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  return copy(AutoVectorizingCopy{}, src, dst);
}

// Auto-vectorizing copy with assumed alignment of dynamic layout strides up to 128bit.
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_aligned(Tensor<SrcEngine, SrcLayout> const& src,
             Tensor<DstEngine, DstLayout>      & dst)
{
  return copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
}

// Specializaton for Atom AutoVectorizingCopy
template <class... Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<AutoVectorizingCopy, Args...> const&,
     Tensor<SrcEngine, SrcLayout>            const& src,
     Tensor<DstEngine, DstLayout>                 & dst)
{
  return copy(AutoVectorizingCopy{}, src, dst);
}

// Specializaton for Atom AutoVectorizingCopyAssumedAlignment
template <int MaxVecBits, class... Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>, Args...> const&,
     Tensor<SrcEngine, SrcLayout>                                            const& src,
     Tensor<DstEngine, DstLayout>                                                 & dst)
{
  return copy(AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>{}, src, dst);
}

#if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)
template <class... CT_Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const& atom,  // Copy_Traits may or may not have the memory barrier in it already
     Tensor<SrcEngine, SrcLayout>                 const& src,
     Tensor<DstEngine, DstLayout>                      & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;
  static_assert(sizeof_bits<SrcType>::value == sizeof_bits<DstType>::value);
  static_assert((is_gmem<SrcEngine>::value && is_smem<DstEngine>::value) ||
                (is_smem<SrcEngine>::value && is_gmem<DstEngine>::value),
                "Bulk Copy only supports gmem -> smem or smem -> gmem movement.");
  // G2S or S2G dispatch
  using BULK_COPY_OP = conditional_t<is_gmem<SrcEngine>::value,
                                     SM90_BULK_COPY_G2S,
                                     SM90_BULK_COPY_S2G>;

  // Find the common subtensor of src and dst
  auto tiler = max_common_layout(src, dst);
  constexpr int vec_elem = decltype(size(tiler))::value;
  constexpr int vec_bits = vec_elem * sizeof_bits_v<SrcType>;
  static_assert(vec_bits >= 128, "Expected at least 128-bits for BLKCP");

  // Construct a new concrete Atom of the vector size
  using BulkAtom = Copy_Atom<Copy_Traits<BULK_COPY_OP, Int<vec_bits>, CT_Args...>, SrcType>;
  auto bulk_atom = apply(atom.opargs_, [](auto const&... args) { return BulkAtom{args...}; });

#if 0
  if (thread0()) {
    print("copy blkcp -- found a max_common_layout of "); print(tiler); print("\n");
    print("   "); print(src); print("\n");
    print("   "); print(dst); print("\n");
  }
#endif

  return copy(bulk_atom, logical_divide(src, tiler), logical_divide(dst, tiler));
}

// Backwards-compat. Throw out any extra Copy_Atom args.
template <class... CT_Args, class... CA_Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...>, CA_Args...> const& atom,
     Tensor<SrcEngine, SrcLayout>                const& src,
     Tensor<DstEngine, DstLayout>                     & dst)
{
  return copy(static_cast<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const&>(atom), src, dst);
}
#endif // #if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)

} // end namespace cute
