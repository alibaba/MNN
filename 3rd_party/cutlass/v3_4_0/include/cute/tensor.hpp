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

#include <cute/util/type_traits.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/integer_sequence.hpp>

#include <cute/container/tuple.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/container/array_subbyte.hpp>

#include <cute/pointer.hpp>
#include <cute/layout.hpp>
#include <cute/tile.hpp>

namespace cute
{

//
// Engine -- owning or non-owning data store
//

// concept Engine {
//   using iterator     = ;
//   using value_type   = ;
//   using element_type = ;
//   using reference    = ;
//   iterator begin();
// };

template <class T, int N>
struct ArrayEngine
{
  using Storage = typename conditional<(sizeof_bits<T>::value % 8 == 0),
                                       array_aligned<T,N>,
                                       array_subbyte<T,N>>::type;
  using iterator     = typename Storage::iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  Storage storage_;

  CUTE_HOST_DEVICE constexpr auto begin() const { return storage_.begin(); }
  CUTE_HOST_DEVICE constexpr auto begin()       { return storage_.begin(); }
};

template <class Iterator>
struct ViewEngine
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }
  CUTE_HOST_DEVICE constexpr iterator      & begin()       { return storage_; }
};

template <class Iterator>
struct ConstViewEngine
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }
};

//
// Tensor
//

template <class Engine, class Layout>
struct Tensor
{
  using iterator     = typename Engine::iterator;
  using value_type   = typename Engine::value_type;
  using element_type = typename Engine::element_type;
  using reference    = typename Engine::reference;

  using engine_type  = Engine;
  using layout_type  = Layout;

  CUTE_HOST_DEVICE constexpr
  Tensor() {}

  template <class Ptr>
  CUTE_HOST_DEVICE constexpr
  Tensor(Ptr const& ptr, Layout const& layout)
      : rep_(layout, ptr) {
  }

  //
  // Accessors
  //

  static constexpr int rank  = Layout::rank;

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  tensor() const {
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout() const {
    return get<0>(rep_);
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  engine() const {
    return get<1>(rep_);
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  engine() {
    return get<1>(rep_);
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  data() const {
    return engine().begin();
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  data() {
    return engine().begin();
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  shape() const {
    return layout().shape();
  }

  CUTE_HOST_DEVICE constexpr
  auto
  size() const {
    return cute::size(shape());
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  stride() const {
    return layout().stride();
  }

  //
  // Indexing op() and op[]
  //

  // Index into this tensor like an array by computing the offset via layout()
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator[](Coord const& coord) {
    return data()[layout()(coord)];
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator[](Coord const& coord) const {
    return data()[layout()(coord)];
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord const& coord) {
    if constexpr (has_underscore<Coord>::value) {
      auto const& [sliced_layout,offset] = slice_and_offset(coord, layout());
      return make_tensor(data() + offset, sliced_layout);
    } else {
      return data()[layout()(coord)];
    }

    CUTE_GCC_UNREACHABLE;
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      auto const& [sliced_layout,offset] = slice_and_offset(coord, layout());
      return make_tensor(data() + offset, sliced_layout);
    } else {
      return data()[layout()(coord)];
    }

    CUTE_GCC_UNREACHABLE;
  }

  // op() convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) {
    return operator()(make_coord(c0,c1,cs...));
  }

  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0,c1,cs...));
  }

  //
  // Compose
  //

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(Layouts const&... layouts) {
    return make_tensor(data(), layout().compose(layouts...));
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(Layouts const&... layouts) const {
    return make_tensor(data(), layout().compose(layouts...));
  }

  //
  // Tile
  //

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(Layouts const&... layouts) {
    return make_tensor(data(), layout().tile(layouts...));
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(Layouts const&... layouts) const {
    return make_tensor(data(), layout().tile(layouts...));
  }

  //
  // Utility
  //

  template <class Int,
            __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_1d_coord(Int const& linear_idx) const {
    return layout().get_1d_coord(linear_idx);
  }

  template <class Int,
            __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_hier_coord(Int const& linear_idx) const {
    return layout().get_hier_coord(linear_idx);
  }

  template <class Int,
            __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_flat_coord(Int const& linear_idx) const {
    return layout().get_flat_coord(linear_idx);
  }

  cute::tuple<layout_type, engine_type> rep_;
};

template <class T>
struct is_tensor : false_type {};
template <class Engine, class Layout>
struct is_tensor<Tensor<Engine,Layout>> : true_type {};

// Customization point for creation of owning and non-owning Tensors
template <class T>
struct MakeTensor
{
  template <class Layout,
            __CUTE_REQUIRES(not has_dereference<T>::value &&
                            is_layout<Layout>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Layout const& layout) const
  {
    static_assert(is_static<Layout>::value, "Dynamic owning tensors not supported");
    using Engine = ArrayEngine<T, cosize_v<Layout>>;
    return Tensor<Engine,Layout>();
  }

  template <class Layout,
            __CUTE_REQUIRES(has_dereference<T>::value &&
                            is_layout<Layout>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(T const& iter, Layout const& layout)
  {
    using Engine = ViewEngine<T>;
    return Tensor<Engine,Layout>(iter, layout);
  }

  template <class LayoutArg, class... LayoutArgs,
            __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(LayoutArg const& arg, LayoutArgs const&... args) const
  {
    return operator()(make_layout(arg, args...));
  }

  template <class LayoutArg, class... LayoutArgs,
            __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(T const& iter, LayoutArg const& arg, LayoutArgs const&... args)
  {
    return operator()(iter, make_layout(arg, args...));
  }
};

//
// make_tensor
//

// Make an owning Tensor that will allocate a static array
// e.g. make_tensor<float>(Int<12>{})
template <class T, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Args const&... args)
{
  return MakeTensor<T>{}(args...);
}

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& iter, Args const&... args)
{
  return MakeTensor<Iterator>{}(iter, args...);
}

//
// make_tensor_like
//   Make a register tensor the same type and shape and (if possible) order as another tensor
//

template <class NewT, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor_like(Layout const& layout)
{
  return make_tensor<NewT>(make_layout_like(layout));
}

template <class NewT, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor_like(Tensor<Engine,Layout> const& tensor)
{
  return make_tensor_like<NewT>(tensor.layout());
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor_like(Tensor<Engine,Layout> const& tensor)
{
  return make_tensor_like<typename Engine::value_type>(tensor.layout());
}

//
// make_fragment_like --
//   Make a tensor the same shape and (if possible) order as another tensor, with special
//   consideration of the 0th mode. The 0th mode is commonly used for MMA_Atoms or Copy_Atoms
//   so this allocates the 0th mode with LayoutLeft regardless of the reference layout.
//

template <class NewT, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Layout const& layout)
{
  return make_tensor<NewT>(make_fragment_like(layout));
}

template <class NewT, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Tensor<Engine,Layout> const& tensor)
{
  return make_fragment_like<NewT>(tensor.layout());
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Tensor<Engine,Layout> const& tensor)
{
  return make_fragment_like<typename Engine::value_type>(tensor.layout());
}

//
// make_counting_tensor
//   Make a tensor from a layout by binding it to a counting iter with 0-offset of the same profile as the codomain.
//

template <class Layout, __CUTE_REQUIRES(is_layout<Layout>::value)>
CUTE_HOST_DEVICE constexpr
auto
make_counting_tensor(Layout const& layout)
{
  return make_tensor(make_inttuple_iter(repeat_like(coshape(layout), Int<0>{})), layout);
}

//
// make_identity_tensor
//   Make a tensor that maps coordinates within a shape to themselves.
//

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_identity_tensor(Shape const& shape)
{
  return make_counting_tensor(make_identity_layout(shape));
}

//
// Utilities
//

// Return the subtensor of a mode
template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
decltype(auto)
tensor(Tensor&& tensor)
{
  return std::forward<Tensor>(tensor);
}

template <int I, int... Is, class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
decltype(auto)
tensor(Tensor&& tensor)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), get<I,Is...>(tensor.layout()));
}

// Return the subtensor of a range of modes
template <int B, int E, class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
decltype(auto)
take(Tensor&& tensor)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), take<B,E>(tensor.layout()));
}

// Return the layout of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
decltype(auto)
layout(Tensor<Engine,Layout> const& tensor)
{
  return layout<Is...>(tensor.layout());
}

// Return the shape of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
decltype(auto)
shape(Tensor<Engine,Layout> const& tensor)
{
  return shape<Is...>(tensor.layout());
}

// Return the stride of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
decltype(auto)
stride(Tensor<Engine,Layout> const& tensor)
{
  return stride<Is...>(tensor.layout());
}

// Return the number of elements in a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
decltype(auto)
size(Tensor<Engine,Layout> const& tensor)
{
  return size<Is...>(tensor.layout());
}

// Return the rank of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
rank(Tensor<Engine,Layout> const& tensor)
{
  return rank<Is...>(tensor.layout());
}

// Return the depth of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
depth(Tensor<Engine, Layout> const& tensor)
{
  return depth<Is...>(tensor.layout());
}

//
// Operations to manipulate Tensors like a Layout
//

template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
flatten(Tensor&& tensor)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), flatten(tensor.layout()));
}

template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Tensor&& tensor)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), coalesce(tensor.layout()));
}

template <class Tensor, class Profile,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Tensor&& tensor, Profile const& profile)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), coalesce(tensor.layout(), profile));
}

template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor&& tensor)
{
  return make_tensor(cute::forward<Tensor>(tensor).data(), filter_zeros(tensor.layout()));
}

template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
filter(Tensor&& tensor)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), filter(tensor.layout()));
}

template <class Tensor, class Profile,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
filter(Tensor&& tensor, Profile const& profile)
{
  return make_tensor(std::forward<Tensor>(tensor).data(), filter(tensor.layout(), profile));
}

// Return a tensor with the same shape as input but offset by a given coordinate
template <class Coord, class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
domain_offset(Coord const& coord, Tensor&& tensor)
{
  auto [layout, ptr_offset] = domain_offset(coord, tensor.layout());
  return make_tensor(std::forward<Tensor>(tensor).data() + ptr_offset, layout);
}

// Group the modes [B,E) into a single mode
// e.g. group<2,4>(make_tensor<int>(Layout<Shape<_1,_2,_3,_4,_5,_6>>{}))
//      => make_tensor<int>(Layout<Shape<_1,_2,Shape<_3,_4>,_5,_6>>{})
template <int B, int E, class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
group_modes(Tensor&& tensor)
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     group<B,E>(tensor.layout()));
}

//
// Recast
//

// NOTE: This is very dangerous to do
//   -- doesn't check dynamic integer divisibility
//   -- doesn't check alignment

template <class NewType, class Tensor>
CUTE_HOST_DEVICE constexpr
auto
recast(Tensor&& tensor)
{
  using OldType = typename remove_cvref_t<Tensor>::value_type;
  auto old_layout = tensor.layout();
  auto new_layout = recast_layout<OldType,NewType>(old_layout);

  // If this is an upcast of a normal Layout with static negative strides, then offset as well
  if constexpr (sizeof(OldType) < sizeof(NewType) && not is_composed_layout<decltype(old_layout)>::value) {
    auto shape_diff = transform(flatten(old_layout.shape()), flatten(new_layout.shape()), minus{});
    auto extent_diff = transform(shape_diff, flatten(old_layout.stride()), multiplies{});
    auto offset = fold(extent_diff, Int<0>{}, [](auto const& i, auto const& a) { return i + cute::min(a,Int<0>{}); });

    return make_tensor(recast_ptr<NewType>(std::forward<Tensor>(tensor).data() + offset), new_layout);
  } else {
    return make_tensor(recast_ptr<NewType>(std::forward<Tensor>(tensor).data()         ), new_layout);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// max_common_vector
//

/* Return Int<N> such that N is the maximum number of contiguous elements
 * that logically correspond in the tensors of @a a and @a b. This is,
 * the number of elements that could reasonably be vectorized into a single load/store.
 *
 * @returns Int<N> with N >= 0
 *
 * A return value of Int<0> indicates that no such conclusion can be made and no
 * vectorization should be attempted.
 *
 * Note that the return value does NOT include alignment concerns such as the pointer value and
 * the divisbility of dynamic strides.
 */
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr
auto
max_common_vector(Tensor<SrcEngine,SrcLayout> const& a,
                  Tensor<DstEngine,DstLayout> const& b)
{
  using SrcType = typename Tensor<SrcEngine,SrcLayout>::value_type;
  using DstType = typename Tensor<DstEngine,DstLayout>::value_type;
  using SrcRef  = typename Tensor<SrcEngine,SrcLayout>::reference;
  using DstRef  = typename Tensor<SrcEngine,SrcLayout>::reference;

  // Determine if vectorization candidates at all
  if constexpr (// Should be the same value_types, else the copy is also performing a cast
                sizeof_bits_v<SrcType> == sizeof_bits_v<DstType> &&
                // The types should be trivially copyable so that vectorization is valid
                is_trivially_copyable<SrcType>::value &&
                is_trivially_copyable<DstType>::value &&
                // Should be load/storing real data, rather than implicit iterators or such
                is_reference<SrcRef>::value &&
                is_reference<DstRef>::value)
  {
    return max_common_vector(a.layout(), b.layout());
  } else {
    return Int<0>{};
  }

  CUTE_GCC_UNREACHABLE;
}

/* Return a layout that points to the maximum number of contiguous elements
 * that logically correspond in the tensors of @a a and @a b. This is,
 * the elements that could reasonably be "vectorized" into a single load/store.
 *
 * @returns Layout R such that composition(a.layout(), R) and composition(b.layout(), R)
 *          are both identity Layouts.
 *
 * Note that the returned layout does NOT include alignment concerns such as the pointer value and
 * the divisbility of dynamic strides.
 */
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr
auto
max_common_layout(Tensor<SrcEngine,SrcLayout> const& a,
                  Tensor<DstEngine,DstLayout> const& b)
{
  using SrcType = typename Tensor<SrcEngine,SrcLayout>::value_type;
  using DstType = typename Tensor<DstEngine,DstLayout>::value_type;
  using SrcRef  = typename Tensor<SrcEngine,SrcLayout>::reference;
  using DstRef  = typename Tensor<SrcEngine,SrcLayout>::reference;

  // Determine if vectorization candidates at all
  if constexpr (// Should be the same value_types, else the copy is also performing a cast
                sizeof_bits_v<SrcType> == sizeof_bits_v<DstType> &&
                // The types should be trivially copyable so that vectorization is valid
                is_trivially_copyable<SrcType>::value &&
                is_trivially_copyable<DstType>::value &&
                // Should be load/storing real data, rather than implicit iterators or such
                is_reference<SrcRef>::value &&
                is_reference<DstRef>::value)
  {
    return max_common_layout(a.layout(), b.layout());
  } else {
    return Layout<_1,_0>{};
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Key algebraic operations -- Divide and Product
//

// Apply a Tiler to the Tensor.
//
// Consider a Tensor with shape (A,B,x,y)
// And a Tiler that is:
//
// * A Layout with shape (BLK_A,BLK_B)
// ** Result Tensor shape ((BLK_A,BLK_B),Rest).
// ** That is, the Tensor and Tile are treated as 1D for the tiling.
// ** See logical_divide(Layout,Layout)
//
// * A Tile<Layout...> with shape <BLK_A,BLK_B>
// ** Result Tensor shape ((BLK_A,a),(BLK_B,b),x,y).
// ** Each mode of the Tile<Layout...> is applied to the corresponding mode of the Tensor.
// ** See logical_divide(Layout,Tuple)
//
// * A Shape (BLK_A,BLK_B)
// ** Result Tensor shape ((BLK_A,a),(BLK_B,b),x,y).
// ** Equivalent to applying Tile<BLK_A:_1,BLK_B:_1>.
// ** See logical_divide(Layout,Tuple) and logical_divide(Layout,Int)
//
// Note that the Tile<Layout...>/Shape Tilers must be weakly_congruent to the Tensor
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(Tensor    && tensor,
               Tiler const& tiler)   // Layout or Tile<Layout...> or Shape
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     logical_divide(tensor.layout(), tiler));
}

// zipped_divide is logical_divide with Tiler modes and Rest modes gathered together: (Tiler,Rest)
// When Tiler is Layout, this has no effect as logical_divide results in the same.
// When Tiler is Tile<Layout...> or Shape, this zips modes into standard form ((BLK_A,BLK_B),(a,b,x,y))
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
zipped_divide(Tensor    && tensor,
              Tiler const& tiler)    // Layout or Tile<Layout...> or Shape
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     zipped_divide(tensor.layout(), tiler));
}

// tiled_divide is zipped_divide with the second output mode flattened ((BLK_A,BLK_B),a,b,x,y)
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
tiled_divide(Tensor   && tensor,
             Tiler const& tiler)     // Layout or Tile<Layout...> or Shape
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     tiled_divide(tensor.layout(), tiler));
}

// flat_divide is zipped_divide with the both modes flattened (BLK_A,BLK_B,a,b,x,y)
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
flat_divide(Tensor    && tensor,
            Tiler const& tiler)      // Layout or Tile<Layout...> or Shape
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     flat_divide(tensor.layout(), tiler));
}

// logical_product on a Tensor doesn't make sense since it often increases cosize
//   though this might make sense for creating Tensors with broadcasted (stride-0) modes

//
// Tensor partitioning utilities
//

// Apply a Tiler to the Tensor, then slice out one of those tiles by slicing into the "Rest" modes.
// With an inner_partition, you get everything that's inside the Tiler. Everything that the Tiler is pointing to.
// Split the modes of tensor according to the Tiler
//   zipped_divide returns something like ((BLK_A,BLK_B,...),(a,b,...,x,y))
// Then slice into the second mode (the "Rest" mode) with Coord
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
inner_partition(Tensor    && tensor,
                Tiler const& tiler,
                Coord const& coord)
{
  auto tensor_tiled = zipped_divide(std::forward<Tensor>(tensor), tiler);
  constexpr int R0 = decltype(rank<0>(tensor_tiled))::value;

  // The coord slices into the second mode (the "rest" mode), flatten the first
  if constexpr (is_tuple<Coord>::value) {
    // Append trailing modes if coord is tuple
    constexpr int R1 = decltype(rank<1>(tensor_tiled))::value;;
    return tensor_tiled(repeat<R0>(_), append<R1>(coord,_));
  } else {
    // Flat indexing if coord is not tuple
    return tensor_tiled(repeat<R0>(_), coord);
  }
}

// Apply a Tiler to the Tensor, then slice out the remainder by slicing into the "Tile" modes.
// With an outer_partition, you get everything that's outside the Tiler. The layout of the Tile in the Tensor.
// Split the modes of tensor according to the Tiler
//   zipped_divide returns something like ((BLK_A,BLK_B,...),(a,b,...,x,y))
// Then slice into the first mode (the "Tile" mode) with Coord
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
outer_partition(Tensor    && tensor,
                Tiler const& tiler,
                Coord const& coord)
{
  auto tensor_tiled = zipped_divide(std::forward<Tensor>(tensor), tiler);
  constexpr int R1 = decltype(rank<1>(tensor_tiled))::value;

  // The coord slices into the first mode (the "tile" mode), flatten the second
  if constexpr (is_tuple<Coord>::value) {
    // Append trailing modes if coord is tuple
    constexpr int R0 = decltype(rank<0>(tensor_tiled))::value;
    return tensor_tiled(append<R0>(coord,_), repeat<R1>(_));
  } else {
    // Flat indexing if coord is not tuple
    return tensor_tiled(coord, repeat<R1>(_));
  }
}

// Tile a tensor according to @a tiler and use @a coord to index into the remainder, keeping the tile.
// This is typical at the CTA level where tiles of data are extracted:
//   Tensor data = ...                                                                         // (  M,  N)
//   Tensor cta_data = local_tile(data, Shape<_32,_64>{}, make_coord(blockIdx.x,blockIdx.y));  // (_32,_64)
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord)   // coord to slice into "remainder"
{
  return inner_partition(std::forward<Tensor>(tensor),
                         tiler,
                         coord);
}

// Same as above, but with a projection parameter to strip out unwanted tiling modes for convenience
//   when using projections of the same tiler.
// This is typical at the CTA level where tiles of data are extracted as projections:
//   Tensor dataA = ...                                                        // (M,K)
//   Tensor dataB = ...                                                        // (N,K)
//   Tensor dataC = ...                                                        // (M,N)
//   auto cta_tiler = Shape<_32, _64, _4>{};
//   auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
//   Tensor ctaA = local_tile(dataA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (_32,_4,k)
//   Tensor ctaB = local_tile(dataA, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (_64,_4,k)
//   Tensor ctaC = local_tile(dataA, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (_32,_64)
template <class Tensor, class Tiler, class Coord, class Proj,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord,   // coord to slice into "remainder"
           Proj  const& proj)    // projection to apply to tiler and coord
{
  return local_tile(std::forward<Tensor>(tensor),
                    dice(proj, tiler),
                    dice(proj, coord));
}

// Tile a tensor according to the flat shape of a layout that provides the coordinate of the target index.
// This is typical at the Thread level where data is partitioned across repeated patterns of threads:
//   Tensor data = ...                                                            // (_16,_64)
//   Tensor thr_data = local_partition(data, Layout<Shape<_2,_16>>{}, thr_idx);   // ( _8, _4)
template <class Tensor, class LShape, class LStride, class Index,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,    // coord -> index
                Index                  const& index)   // index to slice for
{
  static_assert(is_integral<Index>::value);
  return outer_partition(std::forward<Tensor>(tensor),
                         product_each(shape(tile)),
                         tile.get_flat_coord(index));
}

// Same as above, but with a projection parameter to strip out unwanted tiling modes for convenience
//   when using projections of the same tiler.
// This is typical at the Thread level where data is partitioned across projected layouts of threads:
//   Tensor dataA = ...                                                            // (M,K)
//   Tensor dataB = ...                                                            // (N,K)
//   Tensor dataC = ...                                                            // (M,N)
//   auto thr_layout = Layout<Shape<_2,_16,_1>, Stride<_16,_1,_0>>{};
//   Tensor thrA = local_partition(dataA, thr_layout, thr_idx, Step<_1, X,_1>{});  // (M/2,K/1)
//   Tensor thrB = local_partition(dataB, thr_layout, thr_idx, Step< X,_1,_1>{});  // (N/16,K/1)
//   Tensor thrC = local_partition(dataC, thr_layout, thr_idx, Step<_1,_1, X>{});  // (M/2,N/16)
template <class Tensor, class LShape, class LStride, class Index, class Projection,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,   // coord -> index
                Index                  const& index,  // index to slice for
                Projection             const& proj)
{
  return local_partition(std::forward<Tensor>(tensor),
                         dice(proj, tile),
                         index);
}

//
// Display utilities
//

template <class Engine, class Layout>
CUTE_HOST_DEVICE void print(Tensor<Engine,Layout> const& tensor)
{
  print(tensor.data()); print(" o "); print(tensor.layout());
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE void print_tensor(Tensor<Engine,Layout> const& tensor)
{
  print(tensor); print(":\n");

  if constexpr (Layout::rank == 1)
  {
    for (int m = 0; m < size(tensor); ++m) {
      pretty_print(tensor(m));
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        pretty_print(tensor(m,n));
      }
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    print_tensor(tensor(_,_,0));
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("-"); } print("\n");
      print_tensor(tensor(_,_,k));
    }
  } else
  if constexpr (Layout::rank == 4)
  {
    print_tensor(tensor(_,_,_,0));
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("="); } print("\n");
      print_tensor(tensor(_,_,_,p));
    }
  }
}

#if !defined(__CUDACC_RTC__)
template <class Engine, class Layout>
CUTE_HOST std::ostream& print_tensor_os(std::ostream& os, Tensor<Engine,Layout> const& tensor)
{
  int digits = 9;

  if constexpr (Layout::rank == 1)
  {
    for (int m = 0; m < size(tensor); ++m) {
      os << std::setw(digits) << tensor(m) << std::endl;
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        os << std::setw(digits) << tensor(m,n);
      }
      os << std::endl;
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    print_tensor_os(os, tensor(_,_,0));
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < digits*size<1>(tensor); ++i) { os << "-"; } os << std::endl;
      print_tensor_os(os, tensor(_,_,k));
    }
  } else
  if constexpr (Layout::rank == 4)
  {
    print_tensor_os(os, tensor(_,_,_,0));
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < digits*size<1>(tensor); ++i) { os << "="; } os << std::endl;
      print_tensor_os(os, tensor(_,_,_,p));
    }
  }

  return os;
}

template <class Engine, class Layout>
CUTE_HOST std::ostream& operator<<(std::ostream& os, Tensor<Engine,Layout> const& tensor)
{
  os << tensor.layout() << std::endl;
  return print_tensor_os(os, tensor);
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace cute

//
// Extended Engines
//

#include <cute/pointer_swizzle.hpp>
#include <cute/pointer_flagged.hpp>

//
// Tensor Algorithms
//

#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/algorithm/fill.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/axpby.hpp>
#include <cute/algorithm/gemm.hpp>
