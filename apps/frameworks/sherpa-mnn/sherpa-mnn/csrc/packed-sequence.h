// sherpa-mnn/csrc/packed-sequence.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_PACKED_SEQUENCE_H_
#define SHERPA_ONNX_CSRC_PACKED_SEQUENCE_H_

#include <vector>

#include "MNNUtils.hpp"  // NOLINT

namespace sherpa_mnn {

struct PackedSequence {
  std::vector<int32_t> sorted_indexes;
  std::vector<int32_t> batch_sizes;

  // data is a 2-D tensor of shape (sum(batch_sizes), channels)
  MNN::Express::VARP data{nullptr};

  // Return a shallow copy of data[start:start+size, :]
  MNN::Express::VARP Get(int32_t start, int32_t size) {
    auto shape = data->getInfo()->dim;

    std::array<int, 2> ans_shape{size, shape[1]};

    float *p = data->writeMap<float>();

    auto memory_info =
        (MNNAllocator*)(nullptr);

    // a shallow copy
    return MNNUtilsCreateTensor(memory_info, p + start * shape[1],
                                    size * shape[1], ans_shape.data(),
                                    ans_shape.size());
  }
};

/** Similar to torch.nn.utils.rnn.pad_sequence but it supports only
 * batch_first=true.
 *
 * @param allocator
 * @param value  A 3-D tensor of shape (B, T, C). Its dtype is float.
 * @param length A 1-D tensor of shape (B,). Its dtype is int. Each
 *               element in it specifies the valid length of the corresponding
 *               entry in value before padding.
 */
PackedSequence PackPaddedSequence(MNNAllocator *allocator,
                                  MNN::Express::VARP value, MNN::Express::VARP length);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_PACKED_SEQUENCE_H_
