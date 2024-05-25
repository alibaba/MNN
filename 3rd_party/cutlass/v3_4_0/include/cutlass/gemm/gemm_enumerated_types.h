/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Defines common types used for all GEMM-like operators.
*/

/*
  Note:  CUTLASS 3x increases the host compiler requirements to C++17. However, certain
         existing integrations of CUTLASS require C++11 host compilers.

         Until this requirement can be lifted, certain headers with this annotation are required
         to be remain consistent with C++11 syntax.

         C++11 compatibility is enforced by `cutlass_test_unit_core_cpp11`.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

namespace cutlass {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM operand enumeration: D = A * B + C
enum class Operand {
  kA, /// A multiplicand
  kB, /// B multiplicand
  kC, /// Source accumulator
  kD  /// Destination accumulator
};

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class GemmUniversalMode {
  kGemm,
  kGemmSplitKParallel,
  kBatched,
  kArray,
  kGrouped,
  kInvalid
};

////////////////////////////////////////////////////////////////////////////////

/// Some options for clearing shared memory
enum class SharedMemoryClearOption {
  kNone,            ///< SMEM is in don't-care state
  kZfill,           ///< Kernels fill out of bounds accesses with zeros
  kClearLastStage   ///< Last SMEM stage is explicitly cleared. Mainloop uses 'kNone'
};

/////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
