// sherpa-mnn/csrc/transpose.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TRANSPOSE_H_
#define SHERPA_ONNX_CSRC_TRANSPOSE_H_

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {
/** Transpose a 3-D tensor from shape (B, T, C) to (T, B, C).
 *
 * @param allocator
 * @param v A 3-D tensor of shape (B, T, C). Its dataype is type.
 *
 * @return Return a 3-D tensor of shape (T, B, C). Its datatype is type.
 */
template <typename type = float>
MNN::Express::VARP Transpose01(MNNAllocator *allocator, MNN::Express::VARP v);

/** Transpose a 3-D tensor from shape (B, T, C) to (B, C, T).
 *
 * @param allocator
 * @param v A 3-D tensor of shape (B, T, C). Its dataype is type.
 *
 * @return Return a 3-D tensor of shape (B, C, T). Its datatype is type.
 */
template <typename type = float>
MNN::Express::VARP Transpose12(MNNAllocator *allocator, MNN::Express::VARP v);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_TRANSPOSE_H_
