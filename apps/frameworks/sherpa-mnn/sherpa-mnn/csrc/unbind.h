// sherpa-mnn/csrc/unbind.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UNBIND_H_
#define SHERPA_ONNX_CSRC_UNBIND_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

/** It is similar to torch.unbind() but we keep the unbind dim to 1 in
 * the output
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param value  The tensor to unbind
 * @param dim  The dim along which to unbind the tensor
 *
 * @return Return a list of tensors
 */
template <typename T = float>
std::vector<MNN::Express::VARP> Unbind(MNNAllocator *allocator, MNN::Express::VARP value,
                               int32_t dim);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_UNBIND_H_
