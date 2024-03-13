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
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#else
#include <type_traits>
#include <utility>      // tuple_size, tuple_element
#include <cstddef>      // ptrdiff_t
#include <cstdint>      // uintptr_t
#include <limits>       // numeric_limits
#endif

#include <cute/config.hpp>

namespace cute
{
  using CUTE_STL_NAMESPACE::enable_if;
  using CUTE_STL_NAMESPACE::enable_if_t;
}

#define __CUTE_REQUIRES(...)   typename cute::enable_if<(__VA_ARGS__)>::type* = nullptr
#define __CUTE_REQUIRES_V(...) typename cute::enable_if<decltype((__VA_ARGS__))::value>::type* = nullptr

namespace cute
{

// <type_traits>
using CUTE_STL_NAMESPACE::conjunction;
using CUTE_STL_NAMESPACE::conjunction_v;

using CUTE_STL_NAMESPACE::disjunction;
using CUTE_STL_NAMESPACE::disjunction_v;

using CUTE_STL_NAMESPACE::negation;
using CUTE_STL_NAMESPACE::negation_v;

using CUTE_STL_NAMESPACE::void_t;
using CUTE_STL_NAMESPACE::is_void_v;

using CUTE_STL_NAMESPACE::is_base_of;
using CUTE_STL_NAMESPACE::is_base_of_v;

using CUTE_STL_NAMESPACE::is_const;
using CUTE_STL_NAMESPACE::is_const_v;
using CUTE_STL_NAMESPACE::is_volatile;
using CUTE_STL_NAMESPACE::is_volatile_v;

// using CUTE_STL_NAMESPACE::true_type;
// using CUTE_STL_NAMESPACE::false_type;

using CUTE_STL_NAMESPACE::conditional;
using CUTE_STL_NAMESPACE::conditional_t;

using CUTE_STL_NAMESPACE::remove_const_t;
using CUTE_STL_NAMESPACE::remove_cv_t;
using CUTE_STL_NAMESPACE::remove_reference_t;

using CUTE_STL_NAMESPACE::extent;
using CUTE_STL_NAMESPACE::remove_extent;

using CUTE_STL_NAMESPACE::decay;
using CUTE_STL_NAMESPACE::decay_t;

using CUTE_STL_NAMESPACE::is_lvalue_reference;
using CUTE_STL_NAMESPACE::is_lvalue_reference_v;

using CUTE_STL_NAMESPACE::is_reference;
using CUTE_STL_NAMESPACE::is_trivially_copyable;

using CUTE_STL_NAMESPACE::is_same;
using CUTE_STL_NAMESPACE::is_same_v;

using CUTE_STL_NAMESPACE::is_arithmetic;
using CUTE_STL_NAMESPACE::is_unsigned;
using CUTE_STL_NAMESPACE::is_unsigned_v;
using CUTE_STL_NAMESPACE::is_signed;
using CUTE_STL_NAMESPACE::is_signed_v;

using CUTE_STL_NAMESPACE::make_signed;
using CUTE_STL_NAMESPACE::make_signed_t;

// using CUTE_STL_NAMESPACE::is_integral;
template <class T>
using is_std_integral = CUTE_STL_NAMESPACE::is_integral<T>;

using CUTE_STL_NAMESPACE::is_empty;
using CUTE_STL_NAMESPACE::is_empty_v;

using CUTE_STL_NAMESPACE::invoke_result_t;

using CUTE_STL_NAMESPACE::common_type;
using CUTE_STL_NAMESPACE::common_type_t;

// <utility>
using CUTE_STL_NAMESPACE::declval;

template< class T >
constexpr T&& forward(remove_reference_t<T>& t) noexcept
{
  return static_cast<T&&>(t);
}

template< class T >
constexpr T&& forward(remove_reference_t<T>&& t) noexcept
{
  static_assert(! is_lvalue_reference_v<T>,
    "T cannot be an lvalue reference (e.g., U&).");
  return static_cast<T&&>(t);
}

template< class T >
constexpr remove_reference_t<T>&& move( T&& t ) noexcept
{
  return static_cast<remove_reference_t<T>&&>(t);
}

// <limits>
using CUTE_STL_NAMESPACE::numeric_limits;

// <cstddef>
using CUTE_STL_NAMESPACE::ptrdiff_t;

// <cstdint>
using CUTE_STL_NAMESPACE::uintptr_t;

// C++20
// using std::remove_cvref;
template <class T>
struct remove_cvref {
  using type = remove_cv_t<remove_reference_t<T>>;
};

// C++20
// using std::remove_cvref_t;
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

//
// dependent_false
//
// @brief An always-false value that depends on one or more template parameters.
// See
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1830r1.pdf
// https://github.com/cplusplus/papers/issues/572
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
template <class... Args>
inline constexpr bool dependent_false = false;

//
// tuple_size, tuple_element
//
// @brief CuTe-local tuple-traits to prevent conflicts with other libraries.
// For cute:: types, we specialize std::tuple-traits, which is explicitly allowed.
//   cute::tuple, cute::array, cute::array_subbyte, etc
// But CuTe wants to treat some external types as tuples as well. For those,
// we specialize cute::tuple-traits to avoid polluting external traits.
//   dim3, uint3, etc

template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T,void_t<typename CUTE_STL_NAMESPACE::tuple_size<T>::type>> : CUTE_STL_NAMESPACE::integral_constant<size_t, CUTE_STL_NAMESPACE::tuple_size<T>::value> {};

// S =  : std::integral_constant<std::size_t, std::tuple_size<T>::value> {};

template <class T>
constexpr size_t tuple_size_v = tuple_size<T>::value;

template <size_t I, class T, class = void>
struct tuple_element;

template <size_t I, class T>
struct tuple_element<I,T,void_t<typename CUTE_STL_NAMESPACE::tuple_element<I,T>::type>> : CUTE_STL_NAMESPACE::tuple_element<I,T> {};

template <size_t I, class T>
using tuple_element_t = typename tuple_element<I,T>::type;

//
// is_valid
//

namespace detail {

template <class F, class... Args, class = decltype(declval<F&&>()(declval<Args&&>()...))>
CUTE_HOST_DEVICE constexpr auto
is_valid_impl(int) { return CUTE_STL_NAMESPACE::true_type{}; }

template <class F, class... Args>
CUTE_HOST_DEVICE constexpr auto
is_valid_impl(...) { return CUTE_STL_NAMESPACE::false_type{}; }

template <class F>
struct is_valid_fn {
  template <class... Args>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Args&&...) const { return is_valid_impl<F, Args&&...>(int{}); }
};

} // end namespace detail

template <class F>
CUTE_HOST_DEVICE constexpr auto
is_valid(F&&) {
  return detail::is_valid_fn<F&&>{};
}

template <class F, class... Args>
CUTE_HOST_DEVICE constexpr auto
is_valid(F&&, Args&&...) {
  return detail::is_valid_impl<F&&, Args&&...>(int{});
}

} // end namespace cute
