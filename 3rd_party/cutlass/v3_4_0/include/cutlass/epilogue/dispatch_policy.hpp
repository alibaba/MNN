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

#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue {

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//
// Builder Epilogue Schedules
//
//////////////////////////////////////////////////////////////////////////////

struct NoSmemWarpSpecialized {};
struct NoSmemWarpSpecializedArray {};
struct NoSmemWarpSpecializedGroup {};
struct TmaWarpSpecialized {};
struct TmaWarpSpecializedCooperative {};
// DEPRECATED schedules, will be removed in next release
struct TmaWarpSpecializedElementwiseBase : public TmaWarpSpecialized {};
struct TmaWarpSpecializedCooperativeElementwiseBase : public TmaWarpSpecializedCooperative {};
template <
  template <class T> class ActivationFunctor_,
  thread::ScaleType::Kind Scale_ = thread::ScaleType::Default,
  FloatRoundStyle Round_ = FloatRoundStyle::round_to_nearest
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombEltAct instead")]]
TmaWarpSpecializedElementwise : public TmaWarpSpecializedElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  static constexpr thread::ScaleType::Kind Scale = Scale_;
  static constexpr FloatRoundStyle Round = Round_;
};

template <
  template <class T> class ActivationFunctor_,
  thread::ScaleType::Kind Scale_ = thread::ScaleType::Default,
  FloatRoundStyle Round_ = FloatRoundStyle::round_to_nearest
>
struct [[deprecated("Use TmaWarpSpecializedCooperative with fusion::LinCombEltAct instead")]]
TmaWarpSpecializedCooperativeElementwise : public TmaWarpSpecializedCooperativeElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  static constexpr thread::ScaleType::Kind Scale = Scale_;
  static constexpr FloatRoundStyle Round = Round_;
};

struct TmaWarpSpecializedBiasElementwiseBase : public TmaWarpSpecialized{};
struct TmaWarpSpecializedCooperativeBiasElementwiseBase : public TmaWarpSpecializedCooperative {};

template <
  template <class T> class ActivationFunctor_,
  class ElementT_,
  template <class T> class BiasOp_,
  bool StoreT_,
  class ElementBias_
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombPerRowBiasEltActAux instead")]]
TmaWarpSpecializedBiasElementwise : public TmaWarpSpecializedBiasElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  using ElementT = ElementT_;

  template <class T>
  using BiasOp = BiasOp_<T>;

  static constexpr bool StoreT = StoreT_;
  using ElementBias = ElementBias_;
};

template <
  template <class T> class ActivationFunctor_,
  class ElementT_,
  template <class T> class BiasOp_,
  bool StoreT_,
  class ElementBias_
>
struct [[deprecated("Use TmaWarpSpecializedCooperative with fusion::LinCombPerRowBiasEltActAux instead")]]
TmaWarpSpecializedCooperativeBiasElementwise : public TmaWarpSpecializedCooperativeBiasElementwiseBase {
  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;

  using ElementT = ElementT_;

  template <class T>
  using BiasOp = BiasOp_<T>;

  static constexpr bool StoreT = StoreT_;
  using ElementBias = ElementBias_;
};

//////////////////////////////////////////////////////////////////////////////
//
// Collective Dispatch Policies
//
//////////////////////////////////////////////////////////////////////////////

template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_
>
struct Sm90TmaWarpSpecialized {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
  constexpr static bool ReuseSmemC = ReuseSmemC_;
};


// DEPRECATED policies, will be removed in next release
template<
  int StagesC_,
  int StagesD_,
  int FragmentSize_ = 2
>
struct Sm90TmaWarpSpecializedBiasElementwise {
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static int FragmentSize = FragmentSize_;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue
