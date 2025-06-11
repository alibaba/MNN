// sherpa-mnn/csrc/slice.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SLICE_H_
#define SHERPA_ONNX_CSRC_SLICE_H_

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

/** Get a deep copy by slicing a 3-D tensor v.
 *
 * It returns v[dim0_start:dim0_end, dim1_start:dim1_end, :]
 *
 * @param allocator
 * @param v A 3-D tensor. Its data type is T.
 * @param dim0_start  Start index of the first dimension..
 * @param dim0_end    End index of the first dimension..
 * @param dim1_start Start index of the second dimension.
 * @param dim1_end  End index of the second dimension.
 *
 * @return Return a 3-D tensor of shape
 *         (dim0_end-dim0_start, dim1_end-dim1_start, v.shape[2])
 */
template <typename T = float>
MNN::Express::VARP Slice(MNNAllocator *allocator, MNN::Express::VARP v,
                 int32_t dim0_start, int32_t dim0_end, int32_t dim1_start,
                 int32_t dim1_end);

/** Get a deep copy by slicing a 2-D tensor v.
 *
 * It returns v[dim0_start:dim0_end, :]
 *
 * @param allocator
 * @param v A 2-D tensor. Its data type is T.
 * @param dim0_start  Start index of the first dimension..
 * @param dim0_end    End index of the first dimension..
 *
 * @return Return a 2-D tensor of shape
 *         (dim0_end-dim0_start, v.shape[1])
 */
template <typename T = float>
MNN::Express::VARP Slice(MNNAllocator *allocator, MNN::Express::VARP v,
                 int32_t dim0_start, int32_t dim0_end);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SLICE_H_
