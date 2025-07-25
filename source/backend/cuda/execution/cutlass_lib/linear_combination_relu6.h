/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
  \brief Functor performing linear combination with RELU6 operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

namespace cutlass {
namespace epilogue {
namespace thread {

#if defined(MNN_SUPPORT_TRANSFORMER_FUSE)
/// For Cutlass v4.0.0
/// ReLu6 operator - propagates NaNs
/// Always put threshold in the right hand side of max to propagate NaN.
template <typename T>
struct ReLu6 {
  static const bool kIsHeavy=false;
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold, T value0, T value6) const {
    constexpr bool PropagateNaN = true;
    maximum<T, PropagateNaN> mx;
    minimum<T, PropagateNaN> mn;

    return mn(mx(value0, threshold), value6);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    constexpr bool PropagateNaN = true;
    maximum<T, PropagateNaN> mx;
    minimum<T, PropagateNaN> mn;

    return mn(mx(value, T(0)), T(6));
  }
};

template <typename T, int N>
struct ReLu6<Array<T, N>> {
  static const bool kIsHeavy=false;
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold, Array<T, N> const &frag0, Array<T, N> const &frag6) const {
    constexpr bool PropagateNaN = true;
    maximum<Array<T, N>, PropagateNaN> mx;
    minimum<Array<T, N>, PropagateNaN> mn;

    return mn(mx(frag0, threshold), frag6);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag) const {
    constexpr bool PropagateNaN = true;
    maximum<Array<T, N>, PropagateNaN> mx;
    minimum<Array<T, N>, PropagateNaN> mn;

    return mn(mx(frag, T(0)), T(6));
  }
};

#else

/// For Cutlass v2.9.0
template <typename T>
struct ReLu6 {
  static const bool kIsHeavy=false;
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold, T value0, T value6) const {
    maximum<T> mx;
    minimum<T> mn;

    return mn(mx(value0, threshold), value6);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    maximum<T> mx;
    minimum<T> mn;

    return mn(mx(value, T(0)), T(6));
  }
};

template <typename T, int N>
struct ReLu6<Array<T, N>> {
  static const bool kIsHeavy=false;
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold, Array<T, N> const &frag0, Array<T, N> const &frag6) const {
    maximum<Array<T, N> > mx;
    minimum<Array<T, N> > mn;

    return mn(mx(frag0, threshold), frag6);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag) const {
    maximum<Array<T, N> > mx;
    minimum<Array<T, N> > mn;

    return mn(mx(frag, T(0)), T(6));
  }
};

#endif

/// Applies a linear combination operator followed by the RELU6 activation to an array of elements.
///
/// D = relu6(alpha * accumulator + beta * source + uniform)
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
using LinearCombinationRelu6 = LinearCombinationGeneric<ReLu6, ElementOutput_, Count, ElementAccumulator_,
                                                       ElementCompute_, Scale, Round>;

} // namespace thread
} // namespace epilogue
} // namespace cutlass
