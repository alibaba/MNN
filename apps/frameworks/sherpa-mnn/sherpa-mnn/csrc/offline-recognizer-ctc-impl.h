// sherpa-mnn/csrc/offline-recognizer-ctc-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_

#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/offline-ctc-decoder.h"
#include "sherpa-mnn/csrc/offline-ctc-fst-decoder.h"
#include "sherpa-mnn/csrc/offline-ctc-greedy-search-decoder.h"
#include "sherpa-mnn/csrc/offline-ctc-model.h"
#include "sherpa-mnn/csrc/offline-recognizer-impl.h"
#include "sherpa-mnn/csrc/pad-sequence.h"
#include "sherpa-mnn/csrc/symbol-table.h"

namespace sherpa_mnn {

static OfflineRecognitionResult Convert(const OfflineCtcDecoderResult &src,
                                        const SymbolTable &sym_table,
                                        int32_t frame_shift_ms,
                                        int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;

  for (int32_t i = 0; i != src.tokens.size(); ++i) {
    if (sym_table.Contains("SIL") && src.tokens[i] == sym_table["SIL"]) {
      // tdnn models from yesno have a SIL token, we should remove it.
      continue;
    }
    auto sym = sym_table[src.tokens[i]];
    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback
      // (but don't rewrite printable characters 0x20..0x7e,
      //  which collide with standard BPE units)
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
  }

  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  r.words = std::move(src.words);

  return r;
}

class OfflineRecognizerCtcImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCtcImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(OfflineCtcModel::Create(config_.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerCtcImpl(Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(OfflineCtcModel::Create(mgr, config_.model_config)) {
    Init();
  }

  void Init() {
    if (!config_.model_config.telespeech_ctc.empty()) {
      config_.feat_config.snip_edges = true;
      config_.feat_config.num_ceps = 40;
      config_.feat_config.feature_dim = 40;
      config_.feat_config.low_freq = 40;
      config_.feat_config.high_freq = -200;
      config_.feat_config.use_energy = false;
      config_.feat_config.normalize_samples = false;
      config_.feat_config.is_mfcc = true;
    }

    if (!config_.model_config.nemo_ctc.model.empty()) {
      if (model_->IsGigaAM()) {
        config_.feat_config.low_freq = 0;
        config_.feat_config.high_freq = 8000;
        config_.feat_config.remove_dc_offset = false;
        config_.feat_config.preemph_coeff = 0;
        config_.feat_config.window_type = "hann";
        config_.feat_config.feature_dim = 64;
      } else {
        config_.feat_config.low_freq = 0;
        config_.feat_config.high_freq = 0;
        config_.feat_config.is_librosa = true;
        config_.feat_config.remove_dc_offset = false;
        config_.feat_config.window_type = "hann";
      }
    }

    if (!config_.model_config.wenet_ctc.model.empty()) {
      // WeNet CTC models assume input samples are in the range
      // [-32768, 32767], so we set normalize_samples to false
      config_.feat_config.normalize_samples = false;
    }

    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    if (!config_.ctc_fst_decoder_config.graph.empty()) {
      // TODO(fangjun): Support android to read the graph from
      // asset_manager
      decoder_ = std::make_unique<OfflineCtcFstDecoder>(
          config_.ctc_fst_decoder_config);
    } else if (config_.decoding_method == "greedy_search") {
      if (!symbol_table_.Contains("<blk>") &&
          !symbol_table_.Contains("<eps>") &&
          !symbol_table_.Contains("<blank>")) {
        SHERPA_ONNX_LOGE(
            "We expect that tokens.txt contains "
            "the symbol <blk> or <eps> or <blank> and its ID.");
        exit(-1);
      }

      int32_t blank_id = 0;
      if (symbol_table_.Contains("<blk>")) {
        blank_id = symbol_table_["<blk>"];
      } else if (symbol_table_.Contains("<eps>")) {
        // for tdnn models of the yesno recipe from icefall
        blank_id = symbol_table_["<eps>"];
      } else if (symbol_table_.Contains("<blank>")) {
        // for Wenet CTC models
        blank_id = symbol_table_["<blank>"];
      }

      decoder_ = std::make_unique<OfflineCtcGreedySearchDecoder>(blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    if (!model_->SupportBatchProcessing()) {
      // If the model does not support batch process,
      // we process each stream independently.
      for (int32_t i = 0; i != n; ++i) {
        DecodeStream(ss[i]);
      }
      return;
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t feat_dim = config_.feat_config.feature_dim;

    std::vector<MNN::Express::VARP> features;
    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int> features_length_vec(n);

    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> f = ss[i]->GetFrames();

      int32_t num_frames = f.size() / feat_dim;
      features_vec[i] = std::move(f);

      features_length_vec[i] = num_frames;

      std::array<int, 2> shape = {num_frames, feat_dim};

      MNN::Express::VARP x = MNNUtilsCreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }  // for (int32_t i = 0; i != n; ++i)

    std::vector<MNN::Express::VARP> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = features[i];
    }

    std::array<int, 1> features_length_shape = {n};
    MNN::Express::VARP x_length = MNNUtilsCreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    MNN::Express::VARP x = PadSequence(model_->Allocator(), features_pointer,
                               -23.025850929940457f);
    auto t = model_->Forward(std::move(x), std::move(x_length));

    auto results = decoder_->Decode(std::move(t[0]), std::move(t[1]));

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  // Decode a single stream.
  // Some models do not support batch size > 1, e.g., WeNet CTC models.
  void DecodeStream(OfflineStream *s) const {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t feat_dim = config_.feat_config.feature_dim;
    std::vector<float> f = s->GetFrames();

    int32_t num_frames = f.size() / feat_dim;

    std::array<int, 3> shape = {1, num_frames, feat_dim};

    MNN::Express::VARP x = MNNUtilsCreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int x_length_scalar = num_frames;
    std::array<int, 1> x_length_shape = {1};
    MNN::Express::VARP x_length =
        MNNUtilsCreateTensor(memory_info, &x_length_scalar, 1,
                                 x_length_shape.data(), x_length_shape.size());

    auto t = model_->Forward(std::move(x), std::move(x_length));
    auto results = decoder_->Decode(std::move(t[0]), std::move(t[1]));
    int32_t frame_shift_ms = 10;

    auto r = Convert(results[0], symbol_table_, frame_shift_ms,
                     model_->SubsamplingFactor());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    s->SetResult(r);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCtcModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_
