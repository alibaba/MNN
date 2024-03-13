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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#include <cute/numeric/integer_subbyte.hpp>
#include <cute/numeric/uint128.hpp>

namespace cute
{

//
// Signed integers
//

using int2_t  = cute::int2b_t;
using int4_t  = cute::int4b_t;
using int8_t  = CUTE_STL_NAMESPACE::int8_t;
using int16_t = CUTE_STL_NAMESPACE::int16_t;
using int32_t = CUTE_STL_NAMESPACE::int32_t;
using int64_t = CUTE_STL_NAMESPACE::int64_t;

template <int N> struct int_bit;
template <> struct int_bit<  2>  { using type = cute::int2b_t; };
template <> struct int_bit<  4>  { using type = cute::int4b_t; };
template <> struct int_bit<  8>  { using type = int8_t;  };
template <> struct int_bit< 16>  { using type = int16_t; };
template <> struct int_bit< 32>  { using type = int32_t; };
template <> struct int_bit< 64>  { using type = int64_t; };

template <int N>
using int_bit_t = typename int_bit<N>::type;

template <int N>
using int_byte = int_bit<8*N>;

template <int N>
using int_byte_t = typename int_byte<N>::type;

//
// Unsigned integers
//

using uint1_t   = cute::uint1b_t;
using uint2_t   = cute::uint2b_t;
using uint4_t   = cute::uint4b_t;
using uint8_t   = CUTE_STL_NAMESPACE::uint8_t;
using uint16_t  = CUTE_STL_NAMESPACE::uint16_t;
using uint32_t  = CUTE_STL_NAMESPACE::uint32_t;
using uint64_t  = CUTE_STL_NAMESPACE::uint64_t;
using uint128_t = cute::uint128_t;

template <int N> struct uint_bit;
template <> struct uint_bit<  1> { using type = cute::uint1b_t; };
template <> struct uint_bit<  2> { using type = cute::uint2b_t; };
template <> struct uint_bit<  4> { using type = cute::uint4b_t; };
template <> struct uint_bit<  8> { using type = uint8_t;  };
template <> struct uint_bit< 16> { using type = uint16_t; };
template <> struct uint_bit< 32> { using type = uint32_t; };
template <> struct uint_bit< 64> { using type = uint64_t; };
template <> struct uint_bit<128> { using type = cute::uint128_t; };

template <int N>
using uint_bit_t = typename uint_bit<N>::type;

template <int N>
using uint_byte = uint_bit<8*N>;

template <int N>
using uint_byte_t = typename uint_byte<N>::type;

//
// sizeof_bytes
//

template <class T>
struct sizeof_bytes {
  static constexpr size_t value = sizeof(T);
};
template <class T>
static constexpr int sizeof_bytes_v = sizeof_bytes<T>::value;

//
// sizeof_bits
//

template <class T>
struct sizeof_bits {
  static constexpr size_t value = sizeof(T) * 8;
};

template <class T>
struct sizeof_bits<T const>: sizeof_bits<T> {};

template <>
struct sizeof_bits<void> {
  static constexpr size_t value = 0;
};

template <>
struct sizeof_bits<bool> {
  static constexpr size_t value = 1;
};

template <int Bits, bool Signed>
struct sizeof_bits<integer_subbyte<Bits,Signed>> {
  static constexpr size_t value = Bits;
};

template <int Bits, bool Signed>
struct sizeof_bits<cutlass::integer_subbyte<Bits,Signed>> {
  static constexpr size_t value = Bits;
};

template <class T>
static constexpr int sizeof_bits_v = sizeof_bits<T>::value;

} // namespace cute
