// sherpa-mnn/csrc/offline-tts.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-tts-model-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineTtsConfig {
  OfflineTtsModelConfig model;
  // If not empty, it contains a list of rule FST filenames.
  // Filenames are separated by a comma.
  // Example value: rule1.fst,rule2,fst,rule3.fst
  //
  // If there are multiple rules, they are applied from left to right.
  std::string rule_fsts;

  // If there are multiple FST archives, they are applied from left to right.
  std::string rule_fars;

  // Maximum number of sentences that we process at a time.
  // This is to avoid OOM for very long input text.
  // If you set it to -1, then we process all sentences in a single batch.
  int32_t max_num_sentences = 1;

  // A silence interval contains audio samples with value close to 0.
  //
  // the duration of the new interval is old_duration * silence_scale.
  float silence_scale = 0.2;

  OfflineTtsConfig() = default;
  OfflineTtsConfig(const OfflineTtsModelConfig &model,
                   const std::string &rule_fsts, const std::string &rule_fars,
                   int32_t max_num_sentences, float silence_scale)
      : model(model),
        rule_fsts(rule_fsts),
        rule_fars(rule_fars),
        max_num_sentences(max_num_sentences),
        silence_scale(silence_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct GeneratedAudio {
  std::vector<float> samples;
  int32_t sample_rate;

  // Silence means pause here.
  // If scale > 1, then it increases the duration of a pause
  // If scale < 1, then it reduces the duration of a pause
  GeneratedAudio ScaleSilence(float scale) const;
};

class OfflineTtsImpl;

// If the callback returns 0, then it stop generating
// if the callback returns 1, then it keeps generating
using GeneratedAudioCallback = std::function<int32_t(
    const float * /*samples*/, int32_t /*n*/, float /*progress*/)>;

class OfflineTts {
 public:
  ~OfflineTts();
  explicit OfflineTts(const OfflineTtsConfig &config);

  template <typename Manager>
  OfflineTts(Manager *mgr, const OfflineTtsConfig &config);

  // @param text A string containing words separated by spaces
  // @param sid Speaker ID. Used only for multi-speaker models, e.g., models
  //            trained using the VCTK dataset. It is not used for
  //            single-speaker models, e.g., models trained using the ljspeech
  //            dataset.
  // @param speed The speed for the generated speech. E.g., 2 means 2x faster.
  // @param callback If not NULL, it is called whenever config.max_num_sentences
  //                 sentences have been processed. Note that the passed
  //                 pointer `samples` for the callback might be invalidated
  //                 after the callback is returned, so the caller should not
  //                 keep a reference to it. The caller can copy the data if
  //                 he/she wants to access the samples after the callback
  //                 returns. The callback is called in the current thread.
  GeneratedAudio Generate(const std::string &text, int sid = 0,
                          float speed = 1.0,
                          GeneratedAudioCallback callback = nullptr) const;

  // Return the sample rate of the generated audio
  int32_t SampleRate() const;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  int32_t NumSpeakers() const;

 private:
  std::unique_ptr<OfflineTtsImpl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_H_
