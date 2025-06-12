// sherpa-mnn/csrc/offline-transducer-greedy-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-transducer-greedy-search-nemo-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

static std::pair<MNN::Express::VARP, MNN::Express::VARP> BuildDecoderInput(
    int32_t token, MNNAllocator *allocator) {
  std::array<int, 2> shape{1, 1};

  MNN::Express::VARP decoder_input =
      MNNUtilsCreateTensor<int32_t>(allocator, shape.data(), shape.size());

  std::array<int, 1> length_shape{1};
  MNN::Express::VARP decoder_input_length = MNNUtilsCreateTensor<int32_t>(
      allocator, length_shape.data(), length_shape.size());

  int32_t *p = decoder_input->writeMap<int32_t>();

  int32_t *p_length = decoder_input_length->writeMap<int32_t>();

  p[0] = token;

  p_length[0] = 1;

  return {std::move(decoder_input), std::move(decoder_input_length)};
}

static OfflineTransducerDecoderResult DecodeOne(
    const float *p, int32_t num_rows, int32_t num_cols,
    OfflineTransducerNeMoModel *model, float blank_penalty) {
  auto memory_info =
      (MNNAllocator*)(nullptr);

  OfflineTransducerDecoderResult ans;

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;

  auto decoder_input_pair = BuildDecoderInput(blank_id, model->Allocator());

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> decoder_output_pair =
      model->RunDecoder(std::move(decoder_input_pair.first),
                        std::move(decoder_input_pair.second),
                        model->GetDecoderInitStates(1));

  std::array<int, 3> encoder_shape{1, num_cols, 1};

  for (int32_t t = 0; t != num_rows; ++t) {
    MNN::Express::VARP cur_encoder_out = MNNUtilsCreateTensor(
        memory_info, const_cast<float *>(p) + t * num_cols, num_cols,
        encoder_shape.data(), encoder_shape.size());

    MNN::Express::VARP logit = model->RunJoiner(std::move(cur_encoder_out),
                                        View(decoder_output_pair.first));

    float *p_logit = logit->writeMap<float>();
    if (blank_penalty > 0) {
      p_logit[blank_id] -= blank_penalty;
    }

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(p_logit),
        std::max_element(static_cast<const float *>(p_logit),
                         static_cast<const float *>(p_logit) + vocab_size)));

    if (y != blank_id) {
      ans.tokens.push_back(y);
      ans.timestamps.push_back(t);

      decoder_input_pair = BuildDecoderInput(y, model->Allocator());

      decoder_output_pair =
          model->RunDecoder(std::move(decoder_input_pair.first),
                            std::move(decoder_input_pair.second),
                            std::move(decoder_output_pair.second));
    }  // if (y != blank_id)
  }    // for (int32_t i = 0; i != num_rows; ++i)

  return ans;
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerGreedySearchNeMoDecoder::Decode(
    MNN::Express::VARP encoder_out, MNN::Express::VARP encoder_out_length,
    OfflineStream ** /*ss = nullptr*/, int32_t /*n= 0*/) {
  auto shape = encoder_out->getInfo()->dim;

  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t dim1 = static_cast<int32_t>(shape[1]);
  int32_t dim2 = static_cast<int32_t>(shape[2]);

  const int *p_length = encoder_out_length->readMap<int>();
  const float *p = encoder_out->readMap<float>();

  std::vector<OfflineTransducerDecoderResult> ans(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *this_p = p + dim1 * dim2 * i;
    int32_t this_len = p_length[i];

    ans[i] = DecodeOne(this_p, this_len, dim2, model_, blank_penalty_);
  }

  return ans;
}

}  // namespace sherpa_mnn
