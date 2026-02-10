// sherpa-mnn/csrc/online-recognizer-paraformer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-lm.h"
#include "sherpa-mnn/csrc/online-paraformer-decoder.h"
#include "sherpa-mnn/csrc/online-paraformer-model.h"
#include "sherpa-mnn/csrc/online-recognizer-impl.h"
#include "sherpa-mnn/csrc/online-recognizer.h"
#include "sherpa-mnn/csrc/symbol-table.h"

namespace sherpa_mnn {

static OnlineRecognizerResult Convert(const OnlineParaformerDecoderResult &src,
                                      const SymbolTable &sym_table) {
  OnlineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());

  std::string text;

  // When the current token ends with "@@" we set mergeable to true
  bool mergeable = false;

  for (int32_t i = 0; i != src.tokens.size(); ++i) {
    auto sym = sym_table[src.tokens[i]];
    r.tokens.push_back(sym);

    if ((sym.back() != '@') || (sym.size() > 2 && sym[sym.size() - 2] != '@')) {
      // sym does not end with "@@"
      const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
      if (p[0] < 0x80) {
        // an ascii
        if (mergeable) {
          mergeable = false;
          text.append(sym);
        } else {
          text.append(" ");
          text.append(sym);
        }
      } else {
        // not an ascii
        mergeable = false;

        if (i > 0) {
          const uint8_t p = reinterpret_cast<const uint8_t *>(
              sym_table[src.tokens[i - 1]].c_str())[0];
          if (p < 0x80) {
            // put a space between ascii and non-ascii
            text.append(" ");
          }
        }
        text.append(sym);
      }
    } else {
      // this sym ends with @@
      sym = std::string(sym.data(), sym.size() - 2);
      if (mergeable) {
        text.append(sym);
      } else {
        text.append(" ");
        text.append(sym);
        mergeable = true;
      }
    }
  }
  r.text = std::move(text);

  return r;
}

// y[i] += x[i] * scale
static void ScaleAddInPlace(const float *x, int32_t n, float scale, float *y) {
  for (int32_t i = 0; i != n; ++i) {
    y[i] += x[i] * scale;
  }
}

// y[i] = x[i] * scale
static void Scale(const float *x, int32_t n, float scale, float *y) {
  for (int32_t i = 0; i != n; ++i) {
    y[i] = x[i] * scale;
  }
}

class OnlineRecognizerParaformerImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerParaformerImpl(const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        model_(config.model_config),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    if (config.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE(
          "Unsupported decoding method: %s. Support only greedy_search at "
          "present",
          config.decoding_method.c_str());
      exit(-1);
    }

    // Paraformer models assume input samples are in the range
    // [-32768, 32767], so we set normalize_samples to false
    config_.feat_config.normalize_samples = false;
  }

  template <typename Manager>
  explicit OnlineRecognizerParaformerImpl(Manager *mgr,
                                          const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        model_(mgr, config.model_config),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (config.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    // Paraformer models assume input samples are in the range
    // [-32768, 32767], so we set normalize_samples to false
    config_.feat_config.normalize_samples = false;
  }

  OnlineRecognizerParaformerImpl(const OnlineRecognizerParaformerImpl &) =
      delete;

  OnlineRecognizerParaformerImpl operator=(
      const OnlineRecognizerParaformerImpl &) = delete;

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);

    OnlineParaformerDecoderResult r;
    stream->SetParaformerResult(r);

    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + chunk_size_ < s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    // TODO(fangjun): Support batch size > 1
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    auto decoder_result = s->GetParaformerResult();

    auto r = Convert(decoder_result, sym_);
    r.text = ApplyInverseTextNormalization(r.text);
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    const auto &result = s->GetParaformerResult();

    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    int32_t trailing_silence_frames =
        num_processed_frames - result.last_non_blank_frame_index;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    OnlineParaformerDecoderResult r;
    s->SetParaformerResult(r);

    s->GetStates().clear();
    s->GetParaformerEncoderOutCache().clear();
    s->GetParaformerAlphaCache().clear();

    // s->GetParaformerFeatCache().clear();

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

 private:
  void DecodeStream(OnlineStream *s) const {
    const auto num_processed_frames = s->GetNumProcessedFrames();
    std::vector<float> frames = s->GetFrames(num_processed_frames, chunk_size_);
    s->GetNumProcessedFrames() += chunk_size_ - 1;

    frames = ApplyLFR(frames);
    ApplyCMVN(&frames);
    PositionalEncoding(&frames, num_processed_frames / model_.LfrWindowShift());

    int32_t feat_dim = model_.NegativeMean().size();

    // We have scaled inv_stddev by sqrt(encoder_output_size)
    // so the following line can be commented out
    // frames *= encoder_output_size ** 0.5

    // add overlap chunk
    std::vector<float> &feat_cache = s->GetParaformerFeatCache();
    if (feat_cache.empty()) {
      int32_t n = (left_chunk_size_ + right_chunk_size_) * feat_dim;
      feat_cache.resize(n, 0);
    }

    frames.insert(frames.begin(), feat_cache.begin(), feat_cache.end());
    std::copy(frames.end() - feat_cache.size(), frames.end(),
              feat_cache.begin());

    int32_t num_frames = frames.size() / feat_dim;

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 3> x_shape{1, num_frames, feat_dim};
    MNN::Express::VARP x =
        MNNUtilsCreateTensor(memory_info, frames.data(), frames.size(),
                                 x_shape.data(), x_shape.size());

    int x_len_shape = 1;
    int32_t x_len_val = num_frames;

    MNN::Express::VARP x_length =
        MNNUtilsCreateTensor(memory_info, &x_len_val, 1, &x_len_shape, 1);

    auto encoder_out_vec =
        model_.ForwardEncoder(std::move(x), std::move(x_length));

    // CIF search
    auto &encoder_out = encoder_out_vec[0];
    auto &encoder_out_len = encoder_out_vec[1];
    auto &alpha = encoder_out_vec[2];

    float *p_alpha = alpha->writeMap<float>();

    std::vector<int> alpha_shape =
        alpha->getInfo()->dim;

    std::fill(p_alpha, p_alpha + left_chunk_size_, 0);
    std::fill(p_alpha + alpha_shape[1] - right_chunk_size_,
              p_alpha + alpha_shape[1], 0);

    const float *p_encoder_out = encoder_out->readMap<float>();

    std::vector<int> encoder_out_shape =
        encoder_out->getInfo()->dim;

    std::vector<float> &initial_hidden = s->GetParaformerEncoderOutCache();
    if (initial_hidden.empty()) {
      initial_hidden.resize(encoder_out_shape[2]);
    }

    std::vector<float> &alpha_cache = s->GetParaformerAlphaCache();
    if (alpha_cache.empty()) {
      alpha_cache.resize(1);
    }

    std::vector<float> acoustic_embedding;
    acoustic_embedding.reserve(encoder_out_shape[1] * encoder_out_shape[2]);

    float threshold = 1.0;

    float integrate = alpha_cache[0];

    for (int32_t i = 0; i != encoder_out_shape[1]; ++i) {
      float this_alpha = p_alpha[i];
      if (integrate + this_alpha < threshold) {
        integrate += this_alpha;
        ScaleAddInPlace(p_encoder_out + i * encoder_out_shape[2],
                        encoder_out_shape[2], this_alpha,
                        initial_hidden.data());
        continue;
      }

      // fire
      ScaleAddInPlace(p_encoder_out + i * encoder_out_shape[2],
                      encoder_out_shape[2], threshold - integrate,
                      initial_hidden.data());
      acoustic_embedding.insert(acoustic_embedding.end(),
                                initial_hidden.begin(), initial_hidden.end());
      integrate += this_alpha - threshold;

      Scale(p_encoder_out + i * encoder_out_shape[2], encoder_out_shape[2],
            integrate, initial_hidden.data());
    }

    alpha_cache[0] = integrate;

    if (acoustic_embedding.empty()) {
      return;
    }

    auto &states = s->GetStates();
    if (states.empty()) {
      states.reserve(model_.DecoderNumBlocks());

      std::array<int, 3> shape{1, model_.EncoderOutputSize(),
                                   model_.DecoderKernelSize() - 1};

      int32_t num_bytes = sizeof(float) * shape[0] * shape[1] * shape[2];

      for (int32_t i = 0; i != model_.DecoderNumBlocks(); ++i) {
        MNN::Express::VARP this_state = MNNUtilsCreateTensor<float>(
            model_.Allocator(), shape.data(), shape.size());

        memset(this_state->writeMap<float>(), 0, num_bytes);

        states.push_back(std::move(this_state));
      }
    }

    int32_t num_tokens = acoustic_embedding.size() / initial_hidden.size();
    std::array<int, 3> acoustic_embedding_shape{
        1, num_tokens, static_cast<int32_t>(initial_hidden.size())};

    MNN::Express::VARP acoustic_embedding_tensor = MNNUtilsCreateTensor(
        memory_info, acoustic_embedding.data(), acoustic_embedding.size(),
        acoustic_embedding_shape.data(), acoustic_embedding_shape.size());

    std::array<int, 1> acoustic_embedding_length_shape{1};
    MNN::Express::VARP acoustic_embedding_length_tensor = MNNUtilsCreateTensor(
        memory_info, &num_tokens, 1, acoustic_embedding_length_shape.data(),
        acoustic_embedding_length_shape.size());

    auto decoder_out_vec = model_.ForwardDecoder(
        std::move(encoder_out), std::move(encoder_out_len),
        std::move(acoustic_embedding_tensor),
        std::move(acoustic_embedding_length_tensor), std::move(states));

    states.reserve(model_.DecoderNumBlocks());
    for (int32_t i = 2; i != decoder_out_vec.size(); ++i) {
      // TODO(fangjun): When we change chunk_size_, we need to
      // slice decoder_out_vec[i] accordingly.
      states.push_back(std::move(decoder_out_vec[i]));
    }

    const auto &sample_ids = decoder_out_vec[1];
    const int *p_sample_ids = sample_ids->readMap<int>();

    bool non_blank_detected = false;

    auto &result = s->GetParaformerResult();

    for (int32_t i = 0; i != num_tokens; ++i) {
      int32_t t = p_sample_ids[i];
      if (t == 0) {
        continue;
      }

      non_blank_detected = true;
      result.tokens.push_back(t);
    }

    if (non_blank_detected) {
      result.last_non_blank_frame_index = num_processed_frames;
    }
  }

  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    int32_t lfr_window_size = model_.LfrWindowSize();
    int32_t lfr_window_shift = model_.LfrWindowShift();
    int32_t in_feat_dim = config_.feat_config.feature_dim;

    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void ApplyCMVN(std::vector<float> *v) const {
    const std::vector<float> &neg_mean = model_.NegativeMean();
    const std::vector<float> &inv_stddev = model_.InverseStdDev();

    int32_t dim = neg_mean.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + neg_mean[k]) * inv_stddev[k];
      }

      p += dim;
    }
  }

  void PositionalEncoding(std::vector<float> *v, int32_t t_offset) const {
    int32_t lfr_window_size = model_.LfrWindowSize();
    int32_t in_feat_dim = config_.feat_config.feature_dim;

    int32_t feat_dim = in_feat_dim * lfr_window_size;
    int32_t T = v->size() / feat_dim;

    // log(10000)/(7*80/2-1) == 0.03301197265941284
    // 7 is lfr_window_size
    // 80 is in_feat_dim
    // 7*80 is feat_dim
    constexpr float kScale = -0.03301197265941284;

    for (int32_t t = 0; t != T; ++t) {
      float *p = v->data() + t * feat_dim;

      int32_t offset = t + 1 + t_offset;

      for (int32_t d = 0; d < feat_dim / 2; ++d) {
        float inv_timescale = offset * std::exp(d * kScale);

        float sin_d = std::sin(inv_timescale);
        float cos_d = std::cos(inv_timescale);

        p[d] += sin_d;
        p[d + feat_dim / 2] += cos_d;
      }
    }
  }

 private:
  OnlineRecognizerConfig config_;
  OnlineParaformerModel model_;
  SymbolTable sym_;
  Endpoint endpoint_;

  // 0.61 seconds
  int32_t chunk_size_ = 61;
  // (61 - 7) / 6 + 1 = 10

  int32_t left_chunk_size_ = 5;
  int32_t right_chunk_size_ = 3;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_
