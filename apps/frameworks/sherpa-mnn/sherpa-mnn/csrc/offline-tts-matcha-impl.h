// sherpa-mnn/csrc/offline-tts-matcha-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_IMPL_H_

#include <memory>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-mnn/csrc/hifigan-vocoder.h"
#include "sherpa-mnn/csrc/jieba-lexicon.h"
#include "sherpa-mnn/csrc/lexicon.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/melo-tts-lexicon.h"
#include "sherpa-mnn/csrc/offline-tts-character-frontend.h"
#include "sherpa-mnn/csrc/offline-tts-frontend.h"
#include "sherpa-mnn/csrc/offline-tts-impl.h"
#include "sherpa-mnn/csrc/offline-tts-matcha-model.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/piper-phonemize-lexicon.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

class OfflineTtsMatchaImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsMatchaImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMatchaModel>(config.model)),
        vocoder_(std::make_unique<HifiganVocoder>(
            config.model.num_threads, config.model.provider,
            config.model.matcha.vocoder)) {
    InitFrontend();

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }

    if (!config.rule_fars.empty()) {
      if (config.model.debug) {
        SHERPA_ONNX_LOGE("Loading FST archives");
      }
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);

      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
        }
        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(f));
        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }

      if (config.model.debug) {
        SHERPA_ONNX_LOGE("FST archives loaded!");
      }
    }
  }

  template <typename Manager>
  OfflineTtsMatchaImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMatchaModel>(mgr, config.model)),
        vocoder_(std::make_unique<HifiganVocoder>(
            mgr, config.model.num_threads, config.model.provider,
            config.model.matcha.vocoder)) {
    InitFrontend(mgr);

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
        }
        auto buf = ReadFile(mgr, f);
        std::istrstream is(buf.data(), buf.size());
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
      }
    }

    if (!config.rule_fars.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);
      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
        }

        auto buf = ReadFile(mgr, f);

        std::unique_ptr<std::istream> s(
            new std::istrstream(buf.data(), buf.size()));

        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(std::move(s)));

        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }  // for (; !reader->Done(); reader->Next())
      }    // for (const auto &f : files)
    }      // if (!config.rule_fars.empty())
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  GeneratedAudio Generate(
      const std::string &_text, int sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;

    if (num_speakers == 0 && sid != 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%{public}d. sid is ignored",
          static_cast<int32_t>(sid));
#else
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%d. sid is ignored",
          static_cast<int32_t>(sid));
#endif
    }

    if (num_speakers != 0 && (sid >= num_speakers || sid < 0)) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "This model contains only %{public}d speakers. sid should be in the "
          "range [%{public}d, %{public}d]. Given: %{public}d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
#else
      SHERPA_ONNX_LOGE(
          "This model contains only %d speakers. sid should be in the range "
          "[%d, %d]. Given: %d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
#endif
      sid = 0;
    }

    std::string text = _text;
    if (config_.model.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Raw text: %{public}s", text.c_str());
#else
      SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
#endif
    }

    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        text = tn->Normalize(text);
        if (config_.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("After normalizing: %{public}s", text.c_str());
#else
          SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
#endif
        }
      }
    }

    std::vector<TokenIDs> token_ids =
        frontend_->ConvertTextToTokenIds(text, meta_data.voice);

    if (token_ids.empty() ||
        (token_ids.size() == 1 && token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Failed to convert '%{public}s' to token IDs",
                       text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert '%s' to token IDs", text.c_str());
#endif
      return {};
    }

    std::vector<std::vector<int>> x;

    x.reserve(token_ids.size());

    for (auto &i : token_ids) {
      x.push_back(std::move(i.tokens));
    }

    for (auto &k : x) {
      k = AddBlank(k, meta_data.pad_id);
    }

    int32_t x_size = static_cast<int32_t>(x.size());

    if (config_.max_num_sentences <= 0 || x_size <= config_.max_num_sentences) {
      auto ans = Process(x, sid, speed);
      if (callback) {
        callback(ans.samples.data(), ans.samples.size(), 1.0);
      }
      return ans;
    }

    // the input text is too long, we process sentences within it in batches
    // to avoid OOM. Batch size is config_.max_num_sentences
    std::vector<std::vector<int>> batch_x;

    int32_t batch_size = config_.max_num_sentences;
    batch_x.reserve(config_.max_num_sentences);
    int32_t num_batches = x_size / batch_size;

    if (config_.model.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Text is too long. Split it into %{public}d batches. batch size: "
          "%{public}d. Number of sentences: %{public}d",
          num_batches, batch_size, x_size);
#else
      SHERPA_ONNX_LOGE(
          "Text is too long. Split it into %d batches. batch size: %d. Number "
          "of sentences: %d",
          num_batches, batch_size, x_size);
#endif
    }

    GeneratedAudio ans;

    int32_t should_continue = 1;

    int32_t k = 0;

    for (int32_t b = 0; b != num_batches && should_continue; ++b) {
      batch_x.clear();
      for (int32_t i = 0; i != batch_size; ++i, ++k) {
        batch_x.push_back(std::move(x[k]));
      }

      auto audio = Process(batch_x, sid, speed);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        should_continue = callback(audio.samples.data(), audio.samples.size(),
                                   (b + 1) * 1.0 / num_batches);
        // Caution(fangjun): audio is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }
    }

    batch_x.clear();
    while (k < static_cast<int32_t>(x.size()) && should_continue) {
      batch_x.push_back(std::move(x[k]));

      ++k;
    }

    if (!batch_x.empty()) {
      auto audio = Process(batch_x, sid, speed);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        callback(audio.samples.data(), audio.samples.size(), 1.0);
        // Caution(fangjun): audio is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }
    }

    return ans;
  }

 private:
  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    // for piper phonemizer
    // we require that you copy espeak_ng_data
    // from assets to disk
    //
    // for jieba
    // we require that you copy dict from assets to disk
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.jieba && !meta_data.has_espeak) {
      frontend_ = std::make_unique<JiebaLexicon>(
          mgr, config_.model.matcha.lexicon, config_.model.matcha.tokens,
          config_.model.matcha.dict_dir, config_.model.debug);
    } else if (meta_data.has_espeak && !meta_data.jieba) {
      frontend_ = std::make_unique<PiperPhonemizeLexicon>(
          mgr, config_.model.matcha.tokens, config_.model.matcha.data_dir,
          meta_data);
    } else {
      SHERPA_ONNX_LOGE("jieba + espeaker-ng is not supported yet");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitFrontend() {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.jieba && !meta_data.has_espeak) {
      frontend_ = std::make_unique<JiebaLexicon>(
          config_.model.matcha.lexicon, config_.model.matcha.tokens,
          config_.model.matcha.dict_dir, config_.model.debug);
    } else if (meta_data.has_espeak && !meta_data.jieba) {
      frontend_ = std::make_unique<PiperPhonemizeLexicon>(
          config_.model.matcha.tokens, config_.model.matcha.data_dir,
          meta_data);
    } else {
      SHERPA_ONNX_LOGE("jieba + espeaker-ng is not supported yet");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  GeneratedAudio Process(const std::vector<std::vector<int>> &tokens,
                         int32_t sid, float speed) const {
    int32_t num_tokens = 0;
    for (const auto &k : tokens) {
      num_tokens += k.size();
    }

    std::vector<int> x;
    x.reserve(num_tokens);
    for (const auto &k : tokens) {
      x.insert(x.end(), k.begin(), k.end());
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    MNN::Express::VARP x_tensor = MNNUtilsCreateTensor<int>(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    MNN::Express::VARP mel = model_->Run(std::move(x_tensor), sid, speed);
    MNN::Express::VARP audio = vocoder_->Run(std::move(mel));

    std::vector<int> audio_shape =
        audio->getInfo()->dim;

    int total = 1;
    // The output shape may be (1, 1, total) or (1, total) or (total,)
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = audio->readMap<float>();

    GeneratedAudio ans;
    ans.sample_rate = model_->GetMetaData().sample_rate;
    ans.samples = std::vector<float>(p, p + total);

    float silence_scale = config_.silence_scale;
    if (silence_scale != 1) {
      ans = ans.ScaleSilence(silence_scale);
    }

    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsMatchaModel> model_;
  std::unique_ptr<HifiganVocoder> vocoder_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
};

}  // namespace sherpa_mnn
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_IMPL_H_
