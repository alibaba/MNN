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

#include <cutlass/numeric_conversion.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fusion Operations
// Template args must not be implementation dependent
//
/////////////////////////////////////////////////////////////////////////////////////////////////

struct FusionOperation {
  // metadata types/queries that can be overrided
  using ElementOutput = void;
  using ElementCompute = void;

  using ElementSource = void;
  static constexpr bool IsSourceSupported = false;

  using ElementScalar = void;
  static constexpr int AlignmentScalar = 0;
  static constexpr bool IsScaleFactorSupported = false;
  static constexpr bool IsPerRowScaleSupported = false;
  using ElementBias = void;
  static constexpr int AlignmentBias = 0;
  static constexpr bool IsPerRowBiasSupported = false;
  static constexpr bool IsDePerRowBiasSupported = false;

  using ActivationFn = void;
  static constexpr bool IsEltActSupported = false;
  static constexpr bool IsDeEltActSupported = false;

  using ElementAux = void;
  using GmemLayoutTagAux = void;
  static constexpr int AlignmentAux = 0;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAuxInSupported = false;

  using ElementAmax = void;
  static constexpr bool IsAbsMaxSupported = false;

};

// D = alpha * acc
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledAcc : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentScalar = 1;
  static constexpr auto RoundStyle = RoundStyle_;
};

// D = alpha * acc + beta * C
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinearCombination
    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_> {
  using ElementSource = ElementSource_;
  static constexpr bool IsSourceSupported = true;
};

// D = activation(alpha * acc + beta * C)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombEltAct
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
};

// D = alpha * acc + beta * C + per-row bias
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBias
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerRowBiasSupported = true;
};

// D = activation(alpha * acc + beta * C + per-row bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltAct
    : LinCombPerRowBias<ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
};

// D = activation(alpha * acc + beta * C + per-row bias)
// aux = alpha * acc + beta * C + per-row bias
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltActAux
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
};

// D = activation(per-row alpha * acc + per-row beta * C + per-row bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_, // per-row alpha/beta
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  int AlignmentScalar_ = 128 / sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct PerRowLinCombPerRowBiasEltAct
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerRowScaleSupported = true;
};

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
// if D is fp8 
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerRowBiasEltAct
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr bool IsScaleFactorSupported = true;
};

// Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
// if D is fp8 
//   amax_d = max(abs(elements in activation(Z)))
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
// if Aux is fp8 
//   amax_aux = max(abs(elements in Z))
//   Aux = scale_aux * Z
// else
//   Aux = Z
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementAmax_ = ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerRowBiasEltActAmaxAux
    : ScaledLinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAmax = ElementAmax_;
  static constexpr bool IsAbsMaxSupported = true;

  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
};

// Z = Aux
// dY = alpha * acc + beta * C
// D = d_activation(dY, Z)
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / sizeof_bits_v<ElementAux_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombDeEltAct
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsDeEltActSupported = true;

  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxInSupported = true;
};

// Z = Aux
// dY = alpha * acc + beta * C
// D = d_activation(dY, Z)
// dBias = sum of columns of D
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementBias_ = ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombDeEltActDePerRowBias
    : LinCombDeEltAct<GmemLayoutTagAux_, ActivationFn_, ElementOutput_, ElementCompute_,
        ElementAux_, ElementSource_, ElementScalar_, AlignmentAux_, RoundStyle_> {
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsDePerRowBiasSupported = true;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
