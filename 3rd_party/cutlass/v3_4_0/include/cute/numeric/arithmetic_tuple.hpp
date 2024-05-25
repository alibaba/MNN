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

#include <cute/container/tuple.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/util/type_traits.hpp>

namespace cute
{

template <class... T>
struct ArithmeticTuple : tuple<T...>
{
  template <class... U>
  CUTE_HOST_DEVICE constexpr
  ArithmeticTuple(ArithmeticTuple<U...> const& u)
    : tuple<T...>(static_cast<tuple<U...> const&>(u)) {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr
  ArithmeticTuple(tuple<U...> const& u)
    : tuple<T...>(u) {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr
  ArithmeticTuple(U const&... u)
    : tuple<T...>(u...) {}
};

template <class... T>
struct is_tuple<ArithmeticTuple<T...>> : true_type {};

template <class... T>
CUTE_HOST_DEVICE constexpr
auto
make_arithmetic_tuple(T const&... t) {
  return ArithmeticTuple<T...>(t...);
}

template <class... T>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(tuple<T...> const& t) {
  return ArithmeticTuple<T...>(t);
}

template <class T, __CUTE_REQUIRES(is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr
T const&
as_arithmetic_tuple(T const& t) {
  return t;
}

template <class... T>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ArithmeticTuple<T...> const& t) {
  return t;
}

//
// Numeric operators
//

// Addition
template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), plus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, tuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), plus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(tuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), plus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

//
// Special cases
//

template <auto t, class... U>
CUTE_HOST_DEVICE constexpr
ArithmeticTuple<U...> const&
operator+(C<t>, ArithmeticTuple<U...> const& u) {
  static_assert(t == 0, "Artihmetic tuple op+ error!");
  return u;
}

template <class... T, auto u>
CUTE_HOST_DEVICE constexpr
ArithmeticTuple<T...> const&
operator+(ArithmeticTuple<T...> const& t, C<u>) {
  static_assert(u == 0, "Artihmetic tuple op+ error!");
  return t;
}

//
// ArithmeticTupleIterator
//

template <class ArithTuple>
struct ArithmeticTupleIterator
{
  using value_type   = ArithTuple;
  using element_type = ArithTuple;
  using reference    = ArithTuple;

  ArithTuple coord_;

  CUTE_HOST_DEVICE constexpr
  ArithmeticTupleIterator(ArithTuple const& coord = {}) : coord_(coord) {}

  CUTE_HOST_DEVICE constexpr
  ArithTuple const& operator*() const { return coord_; }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto operator[](Coord const& c) const { return *(*this + c); }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto operator+(Coord const& c) const {
    return ArithmeticTupleIterator<decltype(coord_ + c)>(coord_ + c);
  }
};

template <class Tuple>
CUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(Tuple const& t) {
  return ArithmeticTupleIterator(as_arithmetic_tuple(t));
}

template <class T0, class T1, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(T0 const& t0, T1 const& t1, Ts const&... ts) {
  return make_inttuple_iter(cute::make_tuple(t0, t1, ts...));
}

//
// ArithmeticTuple "basis" elements
//   A ScaledBasis<T,N> is a (at least) rank-N+1 ArithmeticTuple:
//      (_0,_0,...,T,_0,...)
//   with value T in the Nth mode

template <class T, int N>
struct ScaledBasis : private tuple<T>
{
  CUTE_HOST_DEVICE constexpr
  ScaledBasis(T const& t = {}) : tuple<T>(t) {}

  CUTE_HOST_DEVICE constexpr
  decltype(auto) value()       { return get<0>(static_cast<tuple<T>      &>(*this)); }
  CUTE_HOST_DEVICE constexpr
  decltype(auto) value() const { return get<0>(static_cast<tuple<T> const&>(*this)); }

  CUTE_HOST_DEVICE static constexpr
  auto mode() { return Int<N>{}; }
};

template <class T>
struct is_scaled_basis : false_type {};
template <class T, int N>
struct is_scaled_basis<ScaledBasis<T,N>> : true_type {};

template <class T, int N>
struct is_integral<ScaledBasis<T,N>> : true_type {};

// Get the scalar T out of a ScaledBasis
template <class SB>
CUTE_HOST_DEVICE constexpr auto
basis_value(SB const& e)
{
  if constexpr (is_scaled_basis<SB>::value) {
    return basis_value(e.value());
  } else {
    return e;
  }
  CUTE_GCC_UNREACHABLE;
}

// Apply the N... pack to another Tuple
template <class SB, class Tuple>
CUTE_HOST_DEVICE constexpr auto
basis_get(SB const& e, Tuple const& t)
{
  if constexpr (is_scaled_basis<SB>::value) {
    return basis_get(e.value(), get<SB::mode()>(t));
  } else {
    return t;
  }
  CUTE_GCC_UNREACHABLE;
}

namespace detail {

template <int... Ns>
struct Basis;

template <>
struct Basis<> {
  using type = Int<1>;
};

template <int N, int... Ns>
struct Basis<N,Ns...> {
  using type = ScaledBasis<typename Basis<Ns...>::type, N>;
};

} // end namespace detail

// Shortcut for writing ScaledBasis<ScaledBasis<ScaledBasis<Int<1>, N0>, N1>, ...>
// E<>    := _1
// E<0>   := (_1,_0,_0,...)
// E<1>   := (_0,_1,_0,...)
// E<0,0> := ((_1,_0,_0,...),_0,_0,...)
// E<0,1> := ((_0,_1,_0,...),_0,_0,...)
// E<1,0> := (_0,(_1,_0,_0,...),_0,...)
// E<1,1> := (_0,(_0,_1,_0,...),_0,...)
template <int... N>
using E = typename detail::Basis<N...>::type;

namespace detail {

template <class T, int... I, int... J>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(T const& t, seq<I...>, seq<J...>) {
  return make_arithmetic_tuple((void(I),Int<0>{})..., t, (void(J),Int<0>{})...);
}

template <class... T, int... I, int... J>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ArithmeticTuple<T...> const& t, seq<I...>, seq<J...>) {
  return make_arithmetic_tuple(get<I>(t)..., (void(J),Int<0>{})...);
}

} // end namespace detail

// Turn a ScaledBases<T,N> into a rank-M ArithmeticTuple
//    with N prefix 0s:  (_0,_0,...N...,_0,T,_0,...,_0,_0)
template <int M, class T, int N>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ScaledBasis<T,N> const& t) {
  static_assert(M > N, "Mismatched ranks");
  return detail::as_arithmetic_tuple(t.value(), make_seq<N>{}, make_seq<M-N-1>{});
}

// Turn a ScaledBases<T,N> into a rank-N ArithmeticTuple
//    with N prefix 0s:  (_0,_0,...N...,_0,T)
template <class T, int N>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ScaledBasis<T,N> const& t) {
  return as_arithmetic_tuple<N+1>(t);
}

// Turn an ArithmeticTuple into a rank-M ArithmeticTuple
//    with postfix 0s:  (t0,t1,t2,...,_0,...,_0,_0)
template <int M, class... T>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ArithmeticTuple<T...> const& t) {
  static_assert(M >= sizeof...(T), "Mismatched ranks");
  return detail::as_arithmetic_tuple(t, make_seq<int(sizeof...(T))>{}, make_seq<M-int(sizeof...(T))>{});
}

template <class T, int M, class U>
CUTE_HOST_DEVICE constexpr
auto
safe_div(ScaledBasis<T,M> const& b, U const& u)
{
  auto t = safe_div(b.value(), u);
  return ScaledBasis<decltype(t),M>{t};
}

template <class T, int M, class U>
CUTE_HOST_DEVICE constexpr
auto
shape_div(ScaledBasis<T,M> const& b, U const& u)
{
  auto t = shape_div(b.value(), u);
  return ScaledBasis<decltype(t),M>{t};
}

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_basis_like(Shape const& shape)
{
  if constexpr (is_integral<Shape>::value) {
    return Int<1>{};
  }
  else {
    // Generate bases for each rank of shape
    return transform(tuple_seq<Shape>{}, shape, [](auto I, auto si) {
      // Generate bases for each rank of si and add an i on front
      using I_type = decltype(I);
      return transform_leaf(make_basis_like(si), [](auto e) {
        // MSVC has trouble capturing variables as constexpr,
        // so that they can be used as template arguments.
        // This is exactly what the code needs to do with i, unfortunately.
        // The work-around is to define i inside the inner lambda,
        // by using just the type from the enclosing scope.
        constexpr int i = I_type::value;
        return ScaledBasis<decltype(e), i>{};
      });
    });
  }

  CUTE_GCC_UNREACHABLE;
}

// Equality
template <class T, int N, class U, int M>
CUTE_HOST_DEVICE constexpr
auto
operator==(ScaledBasis<T,N> const& t, ScaledBasis<U,M> const& u) {
  return bool_constant<M == N>{} && t.value() == u.value();
}

// Not equal to anything else
template <class T, int N, class U>
CUTE_HOST_DEVICE constexpr
false_type
operator==(ScaledBasis<T,N> const&, U const&) {
  return {};
}

template <class T, class U, int M>
CUTE_HOST_DEVICE constexpr
false_type
operator==(T const&, ScaledBasis<U,M> const&) {
  return {};
}

// Abs
template <int N, class T>
CUTE_HOST_DEVICE constexpr
auto
abs(ScaledBasis<T,N> const& e) {
  return ScaledBasis<decltype(abs(e.value())),N>{abs(e.value())};
}

// Multiplication
template <class A, int N, class T>
CUTE_HOST_DEVICE constexpr
auto
operator*(A const& a, ScaledBasis<T,N> const& e) {
  auto r = a * e.value();
  return ScaledBasis<decltype(r),N>{r};
}

template <int N, class T, class B>
CUTE_HOST_DEVICE constexpr
auto
operator*(ScaledBasis<T,N> const& e, B const& b) {
  auto r = e.value() * b;
  return ScaledBasis<decltype(r),N>{r};
}

// Addition
template <int N, class T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(N+1, int(sizeof...(U)));
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple<R>(u);
}

template <class... T, int M, class U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, ScaledBasis<U,M> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), M+1);
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple<R>(u);
}

template <int N, class T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, tuple<U...> const& u) {
  constexpr int R = cute::max(N+1, int(sizeof...(U)));
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple(u);
}

template <class... T, int M, class U>
CUTE_HOST_DEVICE constexpr
auto
operator+(tuple<T...> const& t, ScaledBasis<U,M> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), M+1);
  return as_arithmetic_tuple(t) + as_arithmetic_tuple<R>(u);
}

template <int N, class T, int M, class U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, ScaledBasis<U,M> const& u) {
  constexpr int R = cute::max(N+1,M+1);
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple<R>(u);
}

template <auto t, class U, int M>
CUTE_HOST_DEVICE constexpr
auto
operator+(C<t>, ScaledBasis<U,M> const& u) {
  static_assert(t == 0, "ScaledBasis op+ error!");
  return u;
}

template <class T, int N, auto u>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,N> const& t, C<u>) {
  static_assert(u == 0, "ScaledBasis op+ error!");
  return t;
}

//
// Display utilities
//

template <class ArithTuple>
CUTE_HOST_DEVICE void print(ArithmeticTupleIterator<ArithTuple> const& iter)
{
  printf("ArithTuple"); print(iter.coord_);
}

template <class T, int N>
CUTE_HOST_DEVICE void print(ScaledBasis<T,N> const& e)
{
  print(e.value()); printf("@%d", N);
}

#if !defined(__CUDACC_RTC__)
template <class ArithTuple>
CUTE_HOST std::ostream& operator<<(std::ostream& os, ArithmeticTupleIterator<ArithTuple> const& iter)
{
  return os << "ArithTuple" << iter.coord_;
}

template <class T, int N>
CUTE_HOST std::ostream& operator<<(std::ostream& os, ScaledBasis<T,N> const& e)
{
  return os << e.value() << "@" << N;
}
#endif

} // end namespace cute


namespace CUTE_STL_NAMESPACE
{

template <class... T>
struct tuple_size<cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>
{};

template <class... T>
struct tuple_size<const cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, const cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::tuple_element<I, const CUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace CUTE_STL_NAMESPACE

#ifdef CUTE_STL_NAMESPACE_IS_CUDA_STD
namespace std
{

#if defined(__CUDACC_RTC__)
template <class... _Tp>
struct tuple_size;

template<size_t _Ip, class... _Tp>
struct tuple_element;
#endif

template <class... T>
struct tuple_size<cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>
{};

template <class... T>
struct tuple_size<const cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, const cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::tuple_element<I, const CUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace std
#endif // CUTE_STL_NAMESPACE_IS_CUDA_STD
