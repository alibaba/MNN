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
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementBias_,            // bias data type   --> int32_t
  typename ElementScale_,        // Scale data type  --> float
  typename ElementAccumulator_,  // gemm output & bias Accumulator  --> int32_t
  typename ElementCompute_,      // compute data type  --> float
  typename ElementOutput_,      // output data type  --> int8_t
  int ElementsPerAccess,
  typename BinaryOp_ = multiplies<ElementCompute_>
>
class LinearCombinationBiasScaleClamp {
public:

  using ElementBias = ElementBias_;
  using ElementScale = ElementScale_;
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  using BinaryOp = BinaryOp_;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementBias, kElementsPerAccess>;
  using FragmentOutput = Array<ElementOutput, kElementsPerAccess>;

  static bool const kIsHeavy = false;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = true;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = true;

  /// Host-constructable parameters structure
  struct Params {

    ElementOutput clamp_max;
    ElementOutput clamp_min;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      clamp_max(ElementOutput(platform::numeric_limits<ElementOutput>::max())),
      clamp_min(ElementOutput(platform::numeric_limits<ElementOutput>::lowest())) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementOutput clamp_max,
      ElementOutput clamp_min
    ): clamp_max(clamp_max), clamp_min(clamp_min) {

    }
  };

private:

  //
  // Data members
  //

  ElementOutput clamp_max_;
  ElementOutput clamp_min_;
  bool skip_elementwise_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationBiasScaleClamp(Params const &params) {
    clamp_max_ = params.clamp_max;
    clamp_min_ = params.clamp_min;
  }

  // /// Functionally required for serial reduction in the epilogue
  // CUTLASS_HOST_DEVICE
  // void set_k_partition(int k_partition, int k_partition_count) {
  //   if (k_partition) {
  //     beta_ = ElementCompute(1);
  //   }

  //   if (k_partition != k_partition_count - 1) {
  //     skip_elementwise_ = true;
  //   }
  // }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const &V) const {

    BinaryOp binary_op;
    FragmentCompute intermediate;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_Bias = NumericArrayConverter<ElementCompute, ElementBias, kElementsPerAccess>()(frag_C);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute temp = tmp_Accum[i] + tmp_Bias[i];
      intermediate[i] = binary_op(temp, V[i]);
    }

    minimum<FragmentCompute> min_accumulator;
    maximum<FragmentCompute> max_accumulator;

    /// Clamping constant value
    ElementCompute const kClampMax = NumericConverter<ElementCompute, ElementOutput>()(clamp_max_);
    ElementCompute const kClampMin = NumericConverter<ElementCompute, ElementOutput>()(clamp_min_);

    intermediate = max_accumulator(intermediate, kClampMin);
    intermediate = min_accumulator(intermediate, kClampMax);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess> destination_converter;

    return destination_converter(intermediate);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
