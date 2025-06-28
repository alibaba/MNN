// sherpa-mnn/csrc/stack.h
//
// Copyright (c) 2023 Jingzhao Ou (jingzhao.ou@gmail.com)

#ifndef SHERPA_ONNX_CSRC_STACK_H_
#define SHERPA_ONNX_CSRC_STACK_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

/** Stack a list of tensors along the given dim.
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param values  Pointer to a list of tensors. The shape of the tensor must
 *                be the same except on the dim to be stacked.
 * @param dim  The dim along which to concatenate the input tensors
 *
 * @return Return the stacked tensor
 */
template <typename T = float>
MNN::Express::VARP Stack(MNNAllocator *allocator,
                 const std::vector<MNN::Express::VARP> &values, int32_t dim);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_STACK_H_
