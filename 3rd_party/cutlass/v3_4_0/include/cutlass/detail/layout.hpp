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

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"

#include "cute/layout.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
// For each cutlass::layout, provides its corresponding cute stride types, 64b by default

template <class L>
struct TagToStrideA {
  using type = L;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::RowMajor> {
  using type = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using tag = layout::RowMajor;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::ColumnMajor> {
  using type = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using tag = layout::ColumnMajor;
};

template <class L>
struct TagToStrideB {
  using type = L;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::RowMajor> {
  using type = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using tag = layout::RowMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::ColumnMajor> {
  using type = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using tag = layout::ColumnMajor;
};

// Maps to modes [M, N, L]
template <class LayoutTag>
struct TagToStrideC : TagToStrideA<LayoutTag> { };

// Convenience aliases
template<class LayoutTag>
using TagToStrideA_t = typename TagToStrideA<LayoutTag>::type;

template<class LayoutTag>
using TagToStrideB_t = typename TagToStrideB<LayoutTag>::type;

template<class LayoutTag>
using TagToStrideC_t = typename TagToStrideC<LayoutTag>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////
// For 2.x compatibility APIs, provide stride->layout tag mappers

template<int ModeIndex, class Stride>
constexpr bool
is_major(Stride = {}) {
  // Account for stride types with and without batch mode and batch modes with static zero stride
  return cute::is_constant<1, decltype(cute::front(cute::get<ModeIndex>(Stride{})))>::value;
}

// Note : This method can be used for deducing the Layout Tag of A, C, D Matrices
template<class StrideA>
constexpr
auto
stride_to_layout_tag_A() {
  if constexpr (is_major<0, StrideA>()) { // M major
    return layout::ColumnMajor{};
  }
  else { // K major
    return layout::RowMajor{};
  }

  CUTE_GCC_UNREACHABLE;
}

template<class StrideB>
constexpr
auto
stride_to_layout_tag_B() {
  if constexpr (is_major<0, StrideB>()) { // N major
    return layout::RowMajor{};
  }
  else { // K major
    return layout::ColumnMajor{};
  }

  CUTE_GCC_UNREACHABLE;
}

template<class StrideC>
constexpr
auto
stride_to_layout_tag_C() {
  if constexpr (is_major<0, StrideC>()) { // M major
    return layout::ColumnMajor{};
  }
  else { // N major
    return layout::RowMajor{};
  }

  CUTE_GCC_UNREACHABLE;
}

// Utilities to map Stride back on to their corresponding layout tags
template <class S>
struct StrideToLayoutTagA {
  using type = decltype(detail::stride_to_layout_tag_A<S>());
};

template <class S>
struct StrideToLayoutTagB {
  using type = decltype(detail::stride_to_layout_tag_B<S>());
};

template <class S>
struct StrideToLayoutTagC {
  using type = decltype(detail::stride_to_layout_tag_C<S>());
};

// Convenience aliases
template<class S>
using StrideToLayoutTagA_t = typename StrideToLayoutTagA<S>::type;

template<class S>
using StrideToLayoutTagB_t = typename StrideToLayoutTagB<S>::type;

template<class S>
using StrideToLayoutTagC_t = typename StrideToLayoutTagC<S>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Inspects a tiled copy and whether its copy engine is TMA or not
template<class GmemTiledCopy>
constexpr bool is_tma_copy_engine() {
  if constexpr (cute::is_void_v<GmemTiledCopy>) {
    return false;
  }
  else {
   if constexpr (   cute::is_base_of_v<cute::SM90_TMA_LOAD,                         GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_LOAD_MULTICAST,              GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_LOAD_IM2COL,                 GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_LOAD_IM2COL_MULTICAST,       GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_STORE,                       GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_STORE_IM2COL,                GmemTiledCopy>
                  ) {
      return true;
    }
  }
  return false;
}

// Inspects a TiledCopy and returns its alignment in terms of element count
template <class GmemTiledCopy, class Element>
constexpr int
get_alignment_count_from_gmem_tiled_copy() {

  if constexpr (cute::is_void_v<GmemTiledCopy>) {
    return 1;
  }

  // Account for ElementC = void kernels
  else if constexpr (cute::is_void_v<Element>) {
    return 0;
  }

  else {
    // For TMA tiled copies, we know the alignment has to be 128 bits
    if constexpr (is_tma_copy_engine<GmemTiledCopy>()) {
      return 128 / sizeof_bits<Element>::value;
    }
    else {
      // For non-TMA tiled copies, TiledCopy holds the alignment count directly in its TiledShape_MN
      return GmemTiledCopy::NumValSrc;
    }
  }
}

// Return the shape that is associated with stride-1 mode, or 1 if not found
template<typename Shape, typename Stride>
CUTLASS_HOST_DEVICE constexpr
auto
get_contiguous_shape(Shape const & shape, Stride const & stride) {
  using namespace cute;
  auto idx = find_if(append(flatten(stride), _1{}), [](auto s){ return is_constant<1,decltype(s)>{}; });
  return get<decltype(idx)::value>(append(flatten(shape), _1{}));
}

// Check if tensor shape satisfies a given major alignment
template<int Alignment, class Shape, class Stride>
CUTLASS_HOST_DEVICE constexpr
bool
check_alignment(Shape const & shape, Stride const & stride) {
  return is_major<0>(stride)
    ? get_contiguous_shape(cute::get<0>(shape), cute::get<0>(stride)) % Alignment == 0
    : get_contiguous_shape(cute::get<1>(shape), cute::get<1>(stride)) % Alignment == 0;
}

// Check if tensor shape satisfies a given major alignment

template<int B, int M, int S>
CUTLASS_HOST_DEVICE constexpr
size_t
alignment_for_swizzle(cute::Swizzle<B, M, S>) {
  static_assert(B >= 0 and M >= 0);
  return size_t(1) << size_t(B + M + cute::abs(S));
}

template<class Layout>
CUTLASS_HOST_DEVICE constexpr
size_t
alignment_for_swizzle(Layout layout) {
  return alignment_for_swizzle(cute::detail::get_swizzle_portion(layout));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
