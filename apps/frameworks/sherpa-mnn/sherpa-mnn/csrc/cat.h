// sherpa-mnn/csrc/cat.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_CAT_H_
#define SHERPA_ONNX_CSRC_CAT_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

/** Cat a list of tensors along the given dim.
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param values  Pointer to a list of tensors. The shape of the tensor must
 *                be the same except on the dim to be concatenated.
 * @param dim  The dim along which to concatenate the input tensors
 *
 * @return Return the concatenated tensor
 */
template <typename T = float>
MNN::Express::VARP Cat(MNNAllocator *allocator,
               const std::vector<MNN::Express::VARP> &values, int32_t dim);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_CAT_H_
