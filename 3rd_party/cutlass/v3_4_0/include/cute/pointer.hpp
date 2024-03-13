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
#include <cute/numeric/int.hpp>        // sizeof_bits
#include <cute/numeric/math.hpp>
#include <cute/numeric/integral_constant.hpp>

#include <cute/container/array_subbyte.hpp>

#include <cute/pointer_base.hpp>
#include <cute/pointer_swizzle.hpp>
#include <cute/layout.hpp>
namespace cute
{

//
// recast_ptr<T> -- Create an iterator over values of type T.
// For most types this will simply be T*, but certain types require more care.
// Subbyte Types: uint2_t, uint4_t, etc
//   Requires construction of a subbyte_iterator<T> in order to properly
//   resolve each element in byte-addressed memory.
//

template <class NewT>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(void* ptr)
{
  if constexpr (is_subbyte<NewT>::value) {
    return subbyte_iterator<NewT>(ptr);
  } else {
    return reinterpret_cast<NewT*>(ptr);
  }
  CUTE_GCC_UNREACHABLE;
}

template <class NewT>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(void const* ptr)
{
  if constexpr (is_subbyte<NewT>::value) {
    return subbyte_iterator<NewT const>(ptr);
  } else {
    return reinterpret_cast<NewT const*>(ptr);
  }
  CUTE_GCC_UNREACHABLE;
}

// Disambiguate nullptr
template <class NewT>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(decltype(nullptr)) {   // nullptr_t
  return recast_ptr<NewT>(static_cast<NewT*>(nullptr));
}

//
// gmem_ptr
//

template <class P>
struct gmem_ptr : iter_adaptor<P, gmem_ptr<P>> {
  using iter_adaptor<P, gmem_ptr<P>>::iter_adaptor;
};

template <class T, class = void>
struct is_gmem : false_type {};
template <class P>                     // Found the gmem
struct is_gmem<gmem_ptr<P>> : true_type {};
template <class P>                     // Recurse on ::iterator, if possible
struct is_gmem<P, void_t<typename P::iterator>> : is_gmem<typename P::iterator> {};

// Idempotent gmem tag on an iterator
template <class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(Iterator iter) {
  if constexpr (is_gmem<Iterator>::value) {
    return iter;
  } else {
    return gmem_ptr<Iterator>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(void* ptr) {
  return make_gmem_ptr(recast_ptr<T>(ptr));
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(void const* ptr) {
  return make_gmem_ptr(recast_ptr<T const>(ptr));
}

// nullptr_t overload for make_gmem_ptr<float>(nullptr) disambiguation
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(decltype(nullptr)) { // nullptr_t
  return make_gmem_ptr(recast_ptr<T>(nullptr));
}

// The gmem tag is invariant over type-recast
template <class NewT, class P>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(gmem_ptr<P> const& ptr) {
  return make_gmem_ptr(recast_ptr<NewT>(ptr.get()));
}

//
// smem_ptr
//

template <class P>
struct smem_ptr : iter_adaptor<P, smem_ptr<P>> {
  using iter_adaptor<P, smem_ptr<P>>::iter_adaptor;
};

template <class T, class = void>
struct is_smem : false_type {};
template <class P>                     // Found the smem
struct is_smem<smem_ptr<P>> : true_type {};
template <class P>                     // Recurse on ::iterator, if possible
struct is_smem<P, void_t<typename P::iterator>> : is_smem<typename P::iterator> {};

// Idempotent smem tag on an iterator
template <class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(Iterator iter) {
  if constexpr (is_smem<Iterator>::value) {
    return iter;
  } else {
    return smem_ptr<Iterator>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

// Make a smem swizzle pointer, common operation
template <class Iterator, class Swizzle>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(Iterator ptr, Swizzle sw)
{
  return make_swizzle_ptr(make_smem_ptr(ptr), sw);
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(void* ptr) {
  return make_smem_ptr(recast_ptr<T>(ptr));
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(void const* ptr) {
  return make_smem_ptr(recast_ptr<T const>(ptr));
}

// The smem tag is invariant over type-recast
template <class NewT, class P>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(smem_ptr<P> const& ptr) {
  return make_smem_ptr(recast_ptr<NewT>(ptr.get()));
}

//
// rmem_ptr
//

template <class P>
struct rmem_ptr : iter_adaptor<P, rmem_ptr<P>> {
  using iter_adaptor<P, rmem_ptr<P>>::iter_adaptor;
};

// Anything that is not gmem or smem is rmem
template <class T, class = void>
struct is_rmem : bool_constant<not (is_gmem<T>::value || is_smem<T>::value)> {};
template <class P>
struct is_rmem<rmem_ptr<P>> : true_type {};

// Idempotent rmem tag on an iterator
template <class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_rmem_ptr(Iterator iter) {
  if constexpr (is_rmem<Iterator>::value) {
    return iter;
  } else {
    return rmem_ptr<Iterator>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_rmem_ptr(void* ptr) {
  return make_rmem_ptr(recast_ptr<T>(ptr));
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_rmem_ptr(void const* ptr) {
  return make_rmem_ptr(recast_ptr<T const>(ptr));
}

// The rmem tag is invariant over type-recast
template <class NewT, class P>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(rmem_ptr<P> const& ptr) {
  return make_rmem_ptr(recast_ptr<NewT>(ptr.get()));
}

//
// Display utilities
//

template <class T>
CUTE_HOST_DEVICE void print(gmem_ptr<T> ptr)
{
  printf("gmem_"); print(ptr.get());
}

template <class T>
CUTE_HOST_DEVICE void print(smem_ptr<T> ptr)
{
  printf("smem_"); print(ptr.get());
}

template <class T>
CUTE_HOST_DEVICE void print(rmem_ptr<T> ptr)
{
  printf("rmem_"); print(ptr.get());
}

#if !defined(__CUDACC_RTC__)
template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, gmem_ptr<T> ptr)
{
  return os << "gmem_[" << int(sizeof_bits<iter_value_t<T>>::value) << "b]";
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, smem_ptr<T> ptr)
{
  return os << "smem_[" << int(sizeof_bits<iter_value_t<T>>::value) << "b]";
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, rmem_ptr<T> ptr)
{
  return os << "rmem_[" << int(sizeof_bits<iter_value_t<T>>::value) << "b]";
}

#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
