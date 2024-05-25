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

/// @file copy_traits_sm90_tma_swizzle.hpp
/// @brief Functions for converting swizzle layout to TMA descriptor

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/swizzle_layout.hpp>

namespace cute::detail {

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
TMA::SmemSwizzleBits
get_tma_swizzle_bits(Swizzle<B,M,S>)
{
  if constexpr (M == 4) {
    switch (B) {
      default:  static_assert(0 <= B && B <= 3, "Expected B = 0,1,2, or 3 when M == 4. Unsupported layout swizzle.");
      case 3:   return TMA::SmemSwizzleBits::B128;
      case 2:   return TMA::SmemSwizzleBits::B64;
      case 1:   return TMA::SmemSwizzleBits::B32;
      case 0:   return TMA::SmemSwizzleBits::DISABLE;
    }
  } else
  {
    static_assert(M < 0, "Unsupported layout swizzle.");
  }
}

template <class Layout>
TMA::SmemSwizzleBits
get_tma_swizzle_bits(Layout const& layout)
{
  return get_tma_swizzle_bits(get_swizzle_portion(layout));
}

} // namespace cute::detail
