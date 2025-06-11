// sherpa-mnn/csrc/online-transducer-greedy-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#include "sherpa-mnn/csrc/online-transducer-greedy-search-nemo-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-stream.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

static MNN::Express::VARP BuildDecoderInput(int32_t token, MNNAllocator *allocator) {
  std::array<int, 2> shape{1, 1};

  MNN::Express::VARP decoder_input =
      MNNUtilsCreateTensor<int32_t>(allocator, shape.data(), shape.size());

  int32_t *p = decoder_input->writeMap<int32_t>();

  p[0] = token;

  return decoder_input;
}

static void DecodeOne(const float *encoder_out, int32_t num_rows,
                      int32_t num_cols, OnlineTransducerNeMoModel *model,
                      float blank_penalty, OnlineStream *s) {
  auto memory_info =
      (MNNAllocator*)(nullptr);

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;

  auto &r = s->GetResult();

  MNN::Express::VARP decoder_out{nullptr};

  auto decoder_input = BuildDecoderInput(
      r.tokens.empty() ? blank_id : r.tokens.back(), model->Allocator());

  std::vector<MNN::Express::VARP> &last_decoder_states = s->GetNeMoDecoderStates();

  std::vector<MNN::Express::VARP> tmp_decoder_states;
  tmp_decoder_states.reserve(last_decoder_states.size());
  for (auto &v : last_decoder_states) {
    tmp_decoder_states.push_back(View(v));
  }

  // decoder_output_pair.second returns the next decoder state
  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> decoder_output_pair =
      model->RunDecoder(std::move(decoder_input),
                        std::move(tmp_decoder_states));

  std::array<int, 3> encoder_shape{1, num_cols, 1};

  bool emitted = false;

  for (int32_t t = 0; t != num_rows; ++t) {
    MNN::Express::VARP cur_encoder_out = MNNUtilsCreateTensor(
        memory_info, const_cast<float *>(encoder_out) + t * num_cols, num_cols,
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
      emitted = true;
      r.tokens.push_back(y);
      r.timestamps.push_back(t + r.frame_offset);
      r.num_trailing_blanks = 0;

      decoder_input = BuildDecoderInput(y, model->Allocator());

      // last decoder state becomes the current state for the first chunk
      decoder_output_pair = model->RunDecoder(
          std::move(decoder_input), std::move(decoder_output_pair.second));
    } else {
      ++r.num_trailing_blanks;
    }
  }

  if (emitted) {
    s->SetNeMoDecoderStates(std::move(decoder_output_pair.second));
  }

  r.frame_offset += num_rows;
}

void OnlineTransducerGreedySearchNeMoDecoder::Decode(MNN::Express::VARP encoder_out,
                                                     OnlineStream **ss,
                                                     int32_t n) const {
  auto shape = encoder_out->getInfo()->dim;
  int32_t batch_size = static_cast<int32_t>(shape[0]);  // bs = 1

  if (batch_size != n) {
    SHERPA_ONNX_LOGE("Size mismatch! encoder_out.size(0) %d, n: %d",
                     static_cast<int32_t>(shape[0]), n);
    exit(-1);
  }

  int32_t dim1 = static_cast<int32_t>(shape[1]);  // T
  int32_t dim2 = static_cast<int32_t>(shape[2]);  // encoder_out_dim

  const float *p = encoder_out->readMap<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *this_p = p + dim1 * dim2 * i;

    DecodeOne(this_p, dim1, dim2, model_, blank_penalty_, ss[i]);
  }
}

}  // namespace sherpa_mnn
