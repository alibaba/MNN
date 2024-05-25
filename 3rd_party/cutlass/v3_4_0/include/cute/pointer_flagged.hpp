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

#include <cute/arch/util.hpp>        // cast_smem_ptr_to_uint

#include <cute/pointer.hpp>
#include <cute/pointer_swizzle.hpp>
#include <cute/swizzle_layout.hpp>

#include <cute/tensor.hpp>

namespace cute
{

//
// Stand-in Swizzle Layout
//   A model of a nullptr smem_ptr<T> with B == sizeof_bits<T>::value
//   That represents an unset pointer. This is a placeholder type that is waiting for an smem_ptr
//

template <int Bits>
struct smem_ptr_flag_bits : Int<0> {};

using smem_ptr_flag = smem_ptr_flag_bits<1>;

// A flagged construction method to transform ComposedLayout
// Make a swizzle pointer tensor and check that the intended type size matches
template <class Iterator, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& ptr,
            ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  static_assert(is_smem<Iterator>::value, "Expected smem.");
  static_assert(B == sizeof_bits<iter_value_t<Iterator>>::value, "Expected a B-bit pointer type.");
  return make_tensor(make_smem_ptr(ptr.get(), layout.layout_a()),
                     layout.layout_b());
}

// NOTE: To preserve smem_ptr_flag_bits under recast ops
template <int N, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
upcast(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  return composition(layout.layout_a(), smem_ptr_flag_bits<B*N>{}, upcast<N>(layout.layout_b()));
}

template <int N, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
downcast(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  return composition(layout.layout_a(), smem_ptr_flag_bits<B/N>{}, downcast<N>(layout.layout_b()));
}

//
// Conversion with swizzle_layout
//

template <class T, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE
auto
as_position_independent_swizzle_layout(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  return composition(recast_layout<uint8_t,uint_bit_t<B>>(layout.layout_a()), Int<0>{}, layout.layout_b());
}

template <class Tensor>
CUTE_HOST_DEVICE
auto
as_position_independent_swizzle_tensor(Tensor&& tensor)
{
  static_assert(is_smem<remove_cvref_t<Tensor>>::value, "Expected smem tensor.");
  using SwizzleFn = get_swizzle_t<remove_cvref_t<Tensor>>;
  if constexpr (SwizzleFn::num_bits == 0) {
    return tensor;
  } else {
#if !defined(NDEBUG)
    {
    uint32_t address = cast_smem_ptr_to_uint(raw_pointer_cast(std::forward<Tensor>(tensor).data()));
    uint32_t mask    = ((uint32_t(1) << SwizzleFn::num_base) - 1) | SwizzleFn::swizzle_code;
    assert((address & mask) == 0);  // Alignment to the Base, Z, and Y of Swizzle
    }
#endif
    using T = typename remove_cvref_t<Tensor>::value_type;
    // Recast swizzle from acting on byte-addressed pointers to elements of type-T
    auto new_swizzle = recast_layout<uint8_t, T>(SwizzleFn{});
    // Strip off everything and create a new smem_ptr for type-T
    auto new_ptr = make_smem_ptr<T>(raw_pointer_cast(std::forward<Tensor>(tensor).data()));
    return make_tensor(new_ptr, composition(new_swizzle, Int<0>{}, tensor.layout()));
  }
  CUTE_GCC_UNREACHABLE;
}

//
// Display utilities
//

// Capture and cast smem_ptr_flag Layouts to offset-0 layouts
template <class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE
void
print_latex(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  print_latex(as_position_independent_swizzle_layout(layout));
}

template <int B>
CUTE_HOST_DEVICE void print(smem_ptr_flag_bits<B> ptr)
{
  printf("smem_ptr[%db](unset)", B);
}

} // end namespace cute
