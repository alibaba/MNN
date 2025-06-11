// sherpa-mnn/csrc/pad-sequence.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_PAD_SEQUENCE_H_
#define SHERPA_ONNX_CSRC_PAD_SEQUENCE_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

/** Similar to torch.nn.utils.rnn.pad_sequence but it supports only
 * batch_first=true.
 *
 * @param allocator
 * @param values A list of 2-D tensors. Each tensor's second dimension
 *               must be the same and the data type of each tensor should
 *               be float.
 * @param padding_value Value used for padding. For log-fbank, you usually use
 *                      -23.025850929940457f as the padding value.
 *
 * @return Return a 3-D tensor of shape (B, max_T, C).
 */
MNN::Express::VARP PadSequence(MNNAllocator *allocator,
                       const std::vector<MNN::Express::VARP> &values,
                       float padding_value);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_PAD_SEQUENCE_H_
