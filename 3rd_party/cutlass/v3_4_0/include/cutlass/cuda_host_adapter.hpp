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
    \brief Interface betweeen a CUTLASS device-wide operator and CUDA.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "cutlass/cutlass.h"
#include "cutlass/trace.h"

#include "cutlass/platform/platform.h"
#if ! defined(__CUDACC_RTC__)
#include <cstdio>
#endif

#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
#  define CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Macro-level guard for CUDA Host Adapter
//
#if !defined(CUTLASS_ENABLE_CUDA_HOST_ADAPTER)
#define CUTLASS_ENABLE_CUDA_HOST_ADAPTER false
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class defines an object which abstracts interactions between the CUTLASS device-wide GEMM and
/// CUDA. The intention is to enable CUTLASS to be used with both the CUDA Runtime API and CUDA Driver API.
struct CudaHostAdapter {

  /// Limit the number of kernels
  static constexpr int32_t kMaximumKernelCount = 4;

  /// Maximum cluster size
  static constexpr int MaxClusterSize = 32;

  //
  // Data members
  //

  /// Handles
  void        *kernel_handles[kMaximumKernelCount];
  int32_t      kernel_count = 0;

  //
  // Methods
  //

  /// Ctor
  CudaHostAdapter() = default;

  /// Dtor
  virtual ~CudaHostAdapter() {}

  /// Copy Ctor
  inline CudaHostAdapter(const CudaHostAdapter & rhs):
    kernel_count(rhs.kernel_count)
  {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Copy Assignment
  inline CudaHostAdapter& operator=(const CudaHostAdapter & rhs) {

    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;
    return *this;
  }

  /// Move ctor
  inline CudaHostAdapter(CudaHostAdapter && rhs):
    kernel_count(rhs.kernel_count)
  {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Move assignment
  inline CudaHostAdapter& operator=(CudaHostAdapter && rhs) {

    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }

    kernel_count = rhs.kernel_count;

    return *this;
  }

  /// Ctor
  inline CudaHostAdapter(
    void **kernel_handles_, 
    int32_t kernel_count_
  ): 
    kernel_count(kernel_count_)
  {
    CUTLASS_ASSERT(kernel_count >= 0);
    for (int32_t i = 0; i < kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = kernel_handles_[i];
    }
  }

  /// Returns true if the CudaHostAdapter is empty (kernel_count == 0)
  inline bool empty() const { return !kernel_count; }

  /// Returns kernel_count
  inline size_t size() const { return static_cast<size_t>(kernel_count); }

  /// Queries the occupancy of a kernel
  virtual Status query_occupancy(
    int32_t *device_sms, 
    int32_t *sm_occupancy,
    int32_t kernel_index,
    int32_t thread_count,
    int32_t smem_size) const = 0;
 
  /// Launches a kernel without using Threadblock Clusters. 
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  /// Launches a kernel using the CUDA Extensible Launch API and Threadblock Clusters.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
