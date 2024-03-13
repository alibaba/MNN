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
#include <cutlass/complex.h>

namespace cute
{

using cutlass::complex;
using cutlass::is_complex;
using cutlass::RealType;
using cutlass::real;
using cutlass::imag;
using cutlass::conj;

/// Fused multiply-add for complex numbers
template <class T>
CUTE_HOST_DEVICE constexpr
void
fma(complex<T>      & d,
    complex<T> const& a,
    complex<T> const& b,
    complex<T> const& c)
{
  d.real(fma( a.real(), b.real(), c.real()));
  d.imag(fma( a.real(), b.imag(), c.imag()));
  d.real(fma(-a.imag(), b.imag(), d.real()));
  d.imag(fma( a.imag(), b.real(), d.imag()));
}

/// Fused multiply-add for triplets
template <class T>
CUTE_HOST_DEVICE constexpr
void
fma(complex<T> const& a,
    complex<T> const& b,
    complex<T>      & c)
{
  return fma(c, a, b, c);
}

} // end namespace cute
