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
    \brief PTX for TMA Tensor Memory Access operators on memory added for SM90
*/

#pragma once

#include <cuda_runtime_api.h>
#include "cutlass/cutlass.h"
#include "cutlass/trace.h"

#if defined(__CUDACC_RTC__)
#include <cuda/std/type_traits>
#else
#include <type_traits>
#include <cstdio>
#endif

#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
#  define CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED
#endif

namespace cutlass {

#ifndef NDEBUG
#define Return_Status(cudaError_t_status)            \
  if (cudaError_t_status != cudaSuccess) {           \
    fprintf(stderr,                                  \
            "[ ERROR: CUDA Runtime ] %s:%d: %s\n",   \
            __FILE__,                                \
            __LINE__,                                \
            cudaGetErrorString(cudaError_t_status)); \
    return Status::kInvalid;                         \
  } else {                                           \
    return Status::kSuccess;                         \
  }
#else
#define Return_Status(cudaError_t_status)          \
  if (cudaError_t_status != cudaSuccess) {         \
    return Status::kInvalid;                       \
  } else {                                         \
    return Status::kSuccess;                       \
  }
#endif

struct ClusterLauncher {
  constexpr static int MaxClusterSize = 32;

  // Check for hardware compatibility
  static inline CUTLASS_HOST
  Status check_cluster_dims(dim3 grid, dim3 cluster) {
    if (((cluster.x * cluster.y * cluster.z) <= MaxClusterSize) &&
        (grid.x % cluster.x == 0) && (grid.y % cluster.y == 0) && (grid.z % cluster.z == 0)) {
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("ClusterLauncher: Invalid cluster configuration -- aborting launch.");
      return Status::kInvalid;
    }
  }

  static inline CUTLASS_HOST
  Status
#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)
  init(void const* kernel_function)
#else
  init(void const* /* kernel_function */)
#endif
  {
#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)
    // This attribute was added in CUDA 11.8.
    cudaError_t status =
        cudaFuncSetAttribute(
          kernel_function, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    Return_Status(status);
#else
    return Status::kInvalid;
#endif
  }

  // This is the method we expect to use going forward
  static inline CUTLASS_HOST
  Status launch(
      dim3 const grid_dims,
      dim3 const cluster_dims,
      dim3 const block_dims,
      size_t const smem_size,
      cudaStream_t cuda_stream,
      void const* kernel,
      void** kernel_params) {
#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)
    if (check_cluster_dims(grid_dims, cluster_dims) != Status::kSuccess) {
      CUTLASS_TRACE_HOST("ClusterLauncher: check_cluster_dims() failed. Aborting.");
      return Status::kInvalid;
    }

    auto init_status = init(kernel);
    if (init_status != Status::kSuccess) {
      CUTLASS_TRACE_HOST("ClusterLauncher: init(kernel) failed with status " << int(init_status) << ". Aborting.");
      return Status::kInvalid;
    }

    cudaLaunchConfig_t launch_config;
    launch_config.gridDim = {grid_dims.x, grid_dims.y, grid_dims.z};
    launch_config.blockDim = {block_dims.x, block_dims.y, block_dims.z};
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = cuda_stream;

    cudaLaunchAttribute launch_attribute[1];
    launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
    launch_attribute[0].val.clusterDim.x = cluster_dims.x;
    launch_attribute[0].val.clusterDim.y = cluster_dims.y;
    launch_attribute[0].val.clusterDim.z = cluster_dims.z;

    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = 1;

    CUTLASS_TRACE_HOST("ClusterLauncher: Launching GPC_CLUSTER_GRID GridDims = "
        "(" << grid_dims.x << ", " << grid_dims.y << ", " << grid_dims.z << "), "
        "And ClusterDims = "
        "(" << cluster_dims.x << ", " << cluster_dims.y << ", " << cluster_dims.z << ")\n");

    cudaError_t status = cudaLaunchKernelExC(&launch_config, kernel, kernel_params);
    Return_Status(status);
#else
    CUTLASS_TRACE_HOST("ClusterLauncher: CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED not defined! Aborting cluster launch.");
    return Status::kInvalid;
#endif
  }
};

namespace detail {

template<class Arg>
void* checked_addressof(Arg&& arg) {
  static_assert(! std::is_rvalue_reference_v<Arg> || ! std::is_const_v<Arg>, "You cannot take the address of a const rvalue reference (const T&&).");
  // We use std::addressof to ensure we get the address,
  // in case the type has an overloaded operator&.
  // Note that this precludes `const T&&` references.
  return const_cast<void*>(reinterpret_cast<void const*>(std::addressof(arg)));
}

} // namespace detail

//! Parameters for launch_on_cluster (see below).
struct ClusterLaunchParams {
  //! Grid dimensions
  dim3 grid_dims{1, 1, 1};

  //! Block dimensions
  dim3 block_dims{1, 1, 1};

  //! Cluster dimensions
  dim3 cluster_dims{1, 1, 1};

  //! Number of bytes required for the kernel's shared memory.
  int smem_size_in_bytes = 0;

  //! CUDA stream on which to launch the kernel.
  cudaStream_t cuda_stream = nullptr;
};

/// @brief Launch the kernel on the stream using cluster launch.
///
/// @param params Cluster launch parameters (see above).
/// @param kernel_ptr Pointer to the kernel function (see example).
/// @param args Zero or more arguments to pass to the kernel.
///
/// @tparam Args Types of the arguments passed to the kernel.
///   Don't specify this/these template argument(s) explicitly.
///
/// @return Status::Success on success, else an error code.
///
/// @code
/// template<class SharedMemoryType, class A, class B, class C>
/// __global__ void kernel(A a, B b, C c);
///
/// X x = get_x();
/// Y y = get_y();
/// Z z = get_z();
///
/// void const* kernel_ptr =
///   const_cast<void const*>(reinterpret_cast<void*>(
///     &kernel<SharedMemory, X, Y, Z>));
/// auto status = launch_on_cluster(
///   {grid_dims, block_dims, cluster_dims, sizeof(SharedMemory)},
///   kernel_ptr, x, y, z);
/// @endcode
template<class ... Args>
CUTLASS_HOST cutlass::Status
launch_kernel_on_cluster(const ClusterLaunchParams& params,
  void const* kernel_ptr,
  Args&& ... args)
{
  // Unfortunately, we find ourselves needing to pass in
  // the parameters as an array of raw pointers.
  if constexpr (sizeof...(Args) == 0) {
    return cutlass::ClusterLauncher::launch(
      params.grid_dims, params.cluster_dims, params.block_dims,
      params.smem_size_in_bytes, params.cuda_stream,
      kernel_ptr, nullptr);
  }
  else {
    void* kernel_params[sizeof...(Args)] = {
      detail::checked_addressof(std::forward<Args>(args))...
    };
    return cutlass::ClusterLauncher::launch(
      params.grid_dims, params.cluster_dims, params.block_dims,
      params.smem_size_in_bytes, params.cuda_stream,
      kernel_ptr, kernel_params);
  }
}

}  // namespace cutlass
