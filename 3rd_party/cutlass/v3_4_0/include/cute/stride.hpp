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

#include <cute/int_tuple.hpp>

namespace cute
{

/** crd2idx(c,s,d) maps a coordinate within <Shape,Stride> to an index
 *
 * This is computed as follows:
 *  [coord, shape, and stride are all integers => step forward by stride]
 * op(c, s, d)             => c * d
 *  [coord is integer, shape and stride are tuple => divmod coord for each mode]
 * op(c, (s,S), (d,D))     => op(c % prod(s), s, d) + op(c / prod(s), (S), (D))
 *  [coord, shape, and stride are all tuples => consider each mode independently]
 * op((c,C), (s,S), (d,D)) => op(c, s, d) + op((C), (S), (D))
 */
template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
crd2idx(Coord  const& coord,
        Shape  const& shape,
        Stride const& stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
CUTE_HOST_DEVICE constexpr
auto
crd2idx_ttt(Coord  const& coord,
            Shape  const& shape,
            Stride const& stride, seq<Is...>)
{
  return (... + crd2idx(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
CUTE_HOST_DEVICE constexpr
auto
crd2idx_itt(CInt   const& coord,
            STuple const& shape,
            DTuple const& stride, seq<I0,Is...>)
{
  if constexpr (sizeof...(Is) == 0) {  // Avoid recursion and mod on single/last iter
    return crd2idx(coord, get<I0>(shape), get<I0>(stride));
  } else if constexpr (is_constant<0, CInt>::value) {
    return crd2idx(_0{}, get<I0>(shape), get<I0>(stride))
         + (_0{} + ... + crd2idx(_0{}, get<Is>(shape), get<Is>(stride)));
  } else {                             // General case
    return crd2idx(coord % product(get<I0>(shape)), get<I0>(shape), get<I0>(stride))
         + crd2idx_itt(coord / product(get<I0>(shape)), shape, stride, seq<Is...>{});
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
crd2idx(Coord  const& coord,
        Shape  const& shape,
        Stride const& stride)
{
  if constexpr (is_tuple<Coord>::value) {
    if constexpr (is_tuple<Shape>::value) {      // tuple tuple tuple
      static_assert(tuple_size<Coord>::value == tuple_size< Shape>::value, "Mismatched Ranks");
      static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
      return detail::crd2idx_ttt(coord, shape, stride, tuple_seq<Coord>{});
    } else {                                     // tuple "int" "int"
      static_assert(sizeof(Coord) == 0, "Invalid parameters");
    }
  } else {
    if constexpr (is_tuple<Shape>::value) {      // "int" tuple tuple
      static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
      return detail::crd2idx_itt(coord, shape, stride, tuple_seq<Shape>{});
    } else {                                     // "int" "int" "int"
      return coord * stride;
    }
  }

  CUTE_GCC_UNREACHABLE;
}

namespace detail {

template <class CTuple, class STuple, int I0, int... Is>
CUTE_HOST_DEVICE constexpr
auto
crd2idx_horner(CTuple const& coord,
               STuple const& shape, seq<I0,Is...>)
{
  if constexpr (sizeof...(Is) == 0) {  // No recursion on single/last iter
    return get<I0>(coord);
  } else {                             // General case
    return get<I0>(coord) + get<I0>(shape) * crd2idx_horner(coord, shape, seq<Is...>{});
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

/** crd2idx(c,s) maps a coordinate within Shape to an index
 * via a colexicographical enumeration of coordinates in Shape.
 * i = c0 + s0 * (c1 + s1 * (c2 + s2 * ...))
 */
template <class Coord, class Shape>
CUTE_HOST_DEVICE constexpr
auto
crd2idx(Coord const& coord,
        Shape const& shape)
{
  if constexpr (is_integral<Coord>::value) {  // Coord is already an index
    return coord;
  } else if constexpr (is_integral<Shape>::value) {
    static_assert(dependent_false<Shape>, "Invalid parameters");
  } else {                                    // Make congruent, flatten, and apply Horner's method
    static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
    auto flat_coord = flatten(coord);
    auto flat_shape = flatten(product_like(shape, coord));
    return detail::crd2idx_horner(flat_coord, flat_shape, tuple_seq<decltype(flat_shape)>{});
  }

  CUTE_GCC_UNREACHABLE;
}

/** idx2crd(i,s,d) splits an index into a coordinate within <Shape,Stride>.
 *
 * This is computed as follows:
 *  [index, shape, and stride are all integers => determine 1D coord]
 * op(i, s, d)             => (i / d) % s
 *  [index is integer, shape and stride are tuple => determine component for each mode]
 * op(i, (s,S), (d,D))     => (op(i, s, d), op(i, S, D)...)
 *  [index, shape, and stride are all tuples => consider each mode independently]
 * op((i,I), (s,S), (d,D)) => (op(i, s, d), op((I), (S), (D)))
 *
 * NOTE: This only works for compact shape+stride layouts. A more general version would
 *       apply to all surjective layouts
 */
template <class Index, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
idx2crd(Index  const& idx,
        Shape  const& shape,
        Stride const& stride)
{
  if constexpr (is_tuple<Index>::value) {
    if constexpr (is_tuple<Shape>::value) {      // tuple tuple tuple
      static_assert(tuple_size<Index>::value == tuple_size< Shape>::value, "Mismatched Ranks");
      static_assert(tuple_size<Index>::value == tuple_size<Stride>::value, "Mismatched Ranks");
      return transform(idx, shape, stride, [](auto const& i, auto const& s, auto const& d){ return idx2crd(i,s,d); });
    } else {                                     // tuple "int" "int"
      static_assert(sizeof(Index) == 0, "Invalid parameters");
    }
  } else {
    if constexpr (is_tuple<Shape>::value) {
      if constexpr (is_tuple<Stride>::value) {   // "int" tuple tuple
        static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
        return transform(shape, stride, [&](auto const& s, auto const& d){ return idx2crd(idx,s,d); });
      } else {                                   // "int" tuple "int"
        return transform(shape, compact_col_major(shape, stride), [&](auto const& s, auto const& d){ return idx2crd(idx,s,d); });
      }
    } else {                                     // "int" "int" "int"
      if constexpr (is_constant<1, Shape>::value) {
        // Skip potential stride-0 division
        return Int<0>{};
      } else {
        return (idx / stride) % shape;
      }
    }
  }

  CUTE_GCC_UNREACHABLE;
}

/** idx2crd(i,s) splits an index into a coordinate within Shape
 * via a colexicographical enumeration of coordinates in Shape.
 * c0 = (idx / 1) % s0
 * c1 = (idx / s0) % s1
 * c2 = (idx / (s0 * s1)) % s2
 * ...
 */
template <class Index, class Shape>
CUTE_HOST_DEVICE constexpr
auto
idx2crd(Index const& idx,
        Shape const& shape)
{
  if constexpr (is_tuple<Index>::value) {
    if constexpr (is_tuple<Shape>::value) {      // tuple tuple
      static_assert(tuple_size<Index>::value == tuple_size<Shape>::value, "Mismatched Ranks");
      return transform(idx, shape, [](auto const& i, auto const& s) { return idx2crd(i,s); });
    } else {                                     // tuple "int"
      static_assert(sizeof(Index) == 0, "Invalid parameters");
    }
  } else {
    if constexpr (is_tuple<Shape>::value) {      // "int" tuple
      return idx2crd(idx, shape, compact_col_major(shape));
    } else {                                     // "int" "int"
      return idx;
    }
  }

  CUTE_GCC_UNREACHABLE;
}

//
// crd2crd
//

template <class Coord, class SShape, class DShape>
CUTE_HOST_DEVICE constexpr
auto
crd2crd(Coord  const& coord,
        SShape const& src_shape,
        DShape const& dst_shape)
{
  if constexpr (is_tuple<Coord>::value && is_tuple<SShape>::value && is_tuple<DShape>::value) {
    static_assert(tuple_size<Coord>::value == tuple_size<SShape>::value, "Mismatched Ranks");
    static_assert(tuple_size<Coord>::value == tuple_size<DShape>::value, "Mismatched Ranks");
    return transform(coord, src_shape, dst_shape, [](auto const& c, auto const& s, auto const& d) { return crd2crd(c,s,d); });
  } else {
    // assert(size(src_shape) == size(dst_shape))
    return idx2crd(crd2idx(coord, src_shape), dst_shape);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Compact Major
//

// Tags for common layouts and dispatching
struct LayoutLeft;               // Col-major layout mapping; leftmost extent has stride 1
using GenColMajor = LayoutLeft;  // Alias

struct LayoutRight;              // Row-major layout mapping; rightmost extent has stride 1
using GenRowMajor = LayoutRight; // Alias

namespace detail {

// For GCC8.5 -- Use of lambdas in unevaluated contexts. Instead use function objects.
template <class Major>
struct CompactLambda;

// @pre is_integral<Current>
// Return (result, current * product(shape)) to enable recurrence
template <class Major, class Shape, class Current>
CUTE_HOST_DEVICE constexpr
auto
compact(Shape   const& shape,
        Current const& current)
{
  if constexpr (is_tuple<Shape>::value) { // Shape::tuple Current::int
    using Lambda = CompactLambda<Major>;                  // Append or Prepend
    using Seq    = typename Lambda::template seq<Shape>;  // Seq or RSeq
    return cute::detail::fold(shape, cute::make_tuple(cute::make_tuple(), current), Lambda{}, Seq{});
  } else {                                // Shape::int Current::int
    if constexpr (is_constant<1, Shape>::value) {
      return cute::make_tuple(Int<0>{}, current); // If current is dynamic, this could save a reg
    } else {
      return cute::make_tuple(current, current * shape);
    }
  }

  CUTE_GCC_UNREACHABLE;
}

// For GCC8.5 -- Specialization LayoutLeft
template <>
struct CompactLambda<LayoutLeft>
{
  template <class Init, class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Init const& init, Shape const& si) {
    auto result = detail::compact<LayoutLeft>(si, get<1>(init));
    return cute::make_tuple(append(get<0>(init), get<0>(result)), get<1>(result));  // Append
  }

  template <class Shape>
  using seq = tuple_seq<Shape>;                                                     // Seq
};

// For GCC8.5 -- Specialization LayoutRight
template <>
struct CompactLambda<LayoutRight>
{
  template <class Init, class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Init const& init, Shape const& si) {
    auto result = detail::compact<LayoutRight>(si, get<1>(init));
    return cute::make_tuple(prepend(get<0>(init), get<0>(result)), get<1>(result));  // Prepend
  }

  template <class Shape>
  using seq = tuple_rseq<Shape>;                                                     // RSeq
};

} // end namespace detail

template <class Major, class Shape, class Current = Int<1>,
          __CUTE_REQUIRES(is_tuple<Shape>::value || is_integral<Shape>::value)>
CUTE_HOST_DEVICE constexpr
auto
compact_major(Shape   const& shape,
              Current const& current = {})
{
  if constexpr (is_tuple<Current>::value) {    // Shape::tuple Current::tuple
    static_assert(is_tuple<Shape>::value, "Invalid parameters");
    static_assert(tuple_size<Shape>::value == tuple_size<Current>::value, "Mismatched Ranks");
    // Recurse to apply to the terminals of current
    return transform(shape, current, [&](auto const& s, auto const& c){ return compact_major<Major>(s,c); });
  } else {
    return get<0>(detail::compact<Major>(shape, current));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Compact Col Major
//

struct LayoutLeft {
  template <class Shape>
  using Apply = decltype(compact_major<LayoutLeft>(declval<Shape>()));
};

template <class Shape, class Current = Int<1>>
CUTE_HOST_DEVICE constexpr
auto
compact_col_major(Shape   const& shape,
                  Current const& current = {})
{
  return compact_major<LayoutLeft>(shape, current);
}

//
// Compact Row Major
//

struct LayoutRight {
  template <class Shape>
  using Apply = decltype(compact_major<LayoutRight>(declval<Shape>()));
};

template <class Shape, class Current = Int<1>>
CUTE_HOST_DEVICE constexpr
auto
compact_row_major(Shape   const& shape,
                  Current const& current = {})
{
  return compact_major<LayoutRight>(shape, current);
}

//
// Compact Order -- compute a compact stride based on an ordering of the modes
//

namespace detail {

template <class Shape, class Order, class OrigShape, class OrigOrder>
CUTE_HOST_DEVICE constexpr
auto
compact_order(Shape const& shape, Order const& order,
              OrigShape const& orig_shape, OrigOrder const& orig_order)
{
  if constexpr (is_tuple<Order>::value) {
    return transform(shape, order, [&](auto const& x, auto const& y) { return compact_order(x, y, orig_shape, orig_order); });
  } else {
    auto d = product(transform(orig_shape, orig_order,
                               [&](auto const& s, auto const& o) {
                                  return conditional_return(o < order, product(s), Int<1>{});
                                }));
    return compact_col_major(shape, d);
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class Shape, class Order>
CUTE_HOST_DEVICE constexpr
auto
compact_order(Shape const& shape, Order const& order)
{
  if constexpr(is_congruent<Shape,Order>::value) {
    return detail::compact_order(shape, order, flatten_to_tuple(shape), flatten_to_tuple(order));
  }
  else
  {
    // Here we only want to apply order to top-level subshapes and default (col-major) order on other levels
    static_assert(rank(Shape{}) == rank(Order{}), "Need equal rank of shape and order");
    return detail::compact_order(shape, order, shape, order);
  }
}

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
compact_order(Shape const& shape, GenColMajor const& major)
{
  return compact_major<LayoutLeft>(shape);
}

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
compact_order(Shape const& shape, GenRowMajor const& major)
{
  return compact_major<LayoutRight>(shape);
}

} // end namespace cute
