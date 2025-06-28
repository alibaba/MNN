// sherpa-mnn/c-api/cxx-api.h
//
// Copyright (c)  2024  Xiaomi Corporation

// C++ Wrapper of the C API for sherpa-mnn
#ifndef SHERPA_ONNX_C_API_CXX_API_H_
#define SHERPA_ONNX_C_API_CXX_API_H_

#include <string>
#include <vector>

#include "sherpa-mnn/c-api/c-api.h"

namespace sherpa_mnn::cxx {

// ============================================================================
// Streaming ASR
// ============================================================================
struct OnlineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;
};

struct OnlineParaformerModelConfig {
  std::string encoder;
  std::string decoder;
};

struct OnlineZipformer2CtcModelConfig {
  std::string model;
};

struct OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  OnlineParaformerModelConfig paraformer;
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  std::string tokens;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  bool debug = false;
  std::string model_type;
  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;
  std::string tokens_buf;
};

struct FeatureConfig {
  int32_t sample_rate = 16000;
  int32_t feature_dim = 80;
};

struct OnlineCtcFstDecoderConfig {
  std::string graph;
  int32_t max_active = 3000;
};

struct OnlineRecognizerConfig {
  FeatureConfig feat_config;
  OnlineModelConfig model_config;

  std::string decoding_method = "greedy_search";

  int32_t max_active_paths = 4;

  bool enable_endpoint = false;

  float rule1_min_trailing_silence = 2.4;

  float rule2_min_trailing_silence = 1.2;

  float rule3_min_utterance_length = 20;

  std::string hotwords_file;

  float hotwords_score = 1.5;

  OnlineCtcFstDecoderConfig ctc_fst_decoder_config;
  std::string rule_fsts;
  std::string rule_fars;
  float blank_penalty = 0;

  std::string hotwords_buf;
};

struct OnlineRecognizerResult {
  std::string text;
  std::vector<std::string> tokens;
  std::vector<float> timestamps;
  std::string json;
};

struct Wave {
  std::vector<float> samples;
  int32_t sample_rate;
};

SHERPA_ONNX_API Wave ReadWave(const std::string &filename);

// Return true on success;
// Return false on failure
SHERPA_ONNX_API bool WriteWave(const std::string &filename, const Wave &wave);

template <typename Derived, typename T>
class SHERPA_ONNX_API MoveOnly {
 public:
  explicit MoveOnly(const T *p) : p_(p) {}

  ~MoveOnly() { Destroy(); }

  MoveOnly(const MoveOnly &) = delete;

  MoveOnly &operator=(const MoveOnly &) = delete;

  MoveOnly(MoveOnly &&other) : p_(other.Release()) {}

  MoveOnly &operator=(MoveOnly &&other) {
    if (&other == this) {
      return *this;
    }

    Destroy();

    p_ = other.Release();

    return *this;
  }

  const T *Get() const { return p_; }

  const T *Release() {
    const T *p = p_;
    p_ = nullptr;
    return p;
  }

 private:
  void Destroy() {
    if (p_ == nullptr) {
      return;
    }

    static_cast<Derived *>(this)->Destroy(p_);

    p_ = nullptr;
  }

 protected:
  const T *p_ = nullptr;
};

class SHERPA_ONNX_API OnlineStream
    : public MoveOnly<OnlineStream, SherpaMnnOnlineStream> {
 public:
  explicit OnlineStream(const SherpaMnnOnlineStream *p);

  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  void InputFinished() const;

  void Destroy(const SherpaMnnOnlineStream *p) const;
};

class SHERPA_ONNX_API OnlineRecognizer
    : public MoveOnly<OnlineRecognizer, SherpaMnnOnlineRecognizer> {
 public:
  static OnlineRecognizer Create(const OnlineRecognizerConfig &config);

  void Destroy(const SherpaMnnOnlineRecognizer *p) const;

  OnlineStream CreateStream() const;

  OnlineStream CreateStream(const std::string &hotwords) const;

  bool IsReady(const OnlineStream *s) const;

  void Decode(const OnlineStream *s) const;

  void Decode(const OnlineStream *ss, int32_t n) const;

  OnlineRecognizerResult GetResult(const OnlineStream *s) const;

  void Reset(const OnlineStream *s) const;

  bool IsEndpoint(const OnlineStream *s) const;

 private:
  explicit OnlineRecognizer(const SherpaMnnOnlineRecognizer *p);
};

// ============================================================================
// Non-streaming ASR
// ============================================================================
struct SHERPA_ONNX_API OfflineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;
};

struct SHERPA_ONNX_API OfflineParaformerModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineNemoEncDecCtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineWhisperModelConfig {
  std::string encoder;
  std::string decoder;
  std::string language;
  std::string task = "transcribe";
  int32_t tail_paddings = -1;
};

struct SHERPA_ONNX_API OfflineFireRedAsrModelConfig {
  std::string encoder;
  std::string decoder;
};

struct SHERPA_ONNX_API OfflineTdnnModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineSenseVoiceModelConfig {
  std::string model;
  std::string language;
  bool use_itn = false;
};

struct SHERPA_ONNX_API OfflineMoonshineModelConfig {
  std::string preprocessor;
  std::string encoder;
  std::string uncached_decoder;
  std::string cached_decoder;
};

struct SHERPA_ONNX_API OfflineModelConfig {
  OfflineTransducerModelConfig transducer;
  OfflineParaformerModelConfig paraformer;
  OfflineNemoEncDecCtcModelConfig nemo_ctc;
  OfflineWhisperModelConfig whisper;
  OfflineTdnnModelConfig tdnn;

  std::string tokens;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
  std::string model_type;
  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;
  std::string telespeech_ctc;
  OfflineSenseVoiceModelConfig sense_voice;
  OfflineMoonshineModelConfig moonshine;
  OfflineFireRedAsrModelConfig fire_red_asr;
};

struct SHERPA_ONNX_API OfflineLMConfig {
  std::string model;
  float scale = 1.0;
};

struct SHERPA_ONNX_API OfflineRecognizerConfig {
  FeatureConfig feat_config;
  OfflineModelConfig model_config;
  OfflineLMConfig lm_config;

  std::string decoding_method = "greedy_search";
  int32_t max_active_paths = 4;

  std::string hotwords_file;

  float hotwords_score = 1.5;
  std::string rule_fsts;
  std::string rule_fars;
  float blank_penalty = 0;
};

struct SHERPA_ONNX_API OfflineRecognizerResult {
  std::string text;
  std::vector<float> timestamps;
  std::vector<std::string> tokens;
  std::string json;
  std::string lang;
  std::string emotion;
  std::string event;
};

class SHERPA_ONNX_API OfflineStream
    : public MoveOnly<OfflineStream, SherpaMnnOfflineStream> {
 public:
  explicit OfflineStream(const SherpaMnnOfflineStream *p);

  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  void Destroy(const SherpaMnnOfflineStream *p) const;
};

class SHERPA_ONNX_API OfflineRecognizer
    : public MoveOnly<OfflineRecognizer, SherpaMnnOfflineRecognizer> {
 public:
  static OfflineRecognizer Create(const OfflineRecognizerConfig &config);

  void Destroy(const SherpaMnnOfflineRecognizer *p) const;

  OfflineStream CreateStream() const;

  OfflineStream CreateStream(const std::string &hotwords) const;

  void Decode(const OfflineStream *s) const;

  void Decode(const OfflineStream *ss, int32_t n) const;

  OfflineRecognizerResult GetResult(const OfflineStream *s) const;

 private:
  explicit OfflineRecognizer(const SherpaMnnOfflineRecognizer *p);
};

// ============================================================================
// Non-streaming TTS
// ============================================================================
struct OfflineTtsVitsModelConfig {
  std::string model;
  std::string lexicon;
  std::string tokens;
  std::string data_dir;
  std::string dict_dir;

  float noise_scale = 0.667;
  float noise_scale_w = 0.8;
  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsMatchaModelConfig {
  std::string acoustic_model;
  std::string vocoder;
  std::string lexicon;
  std::string tokens;
  std::string data_dir;
  std::string dict_dir;

  float noise_scale = 0.667;
  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsKokoroModelConfig {
  std::string model;
  std::string voices;
  std::string tokens;
  std::string data_dir;
  std::string dict_dir;
  std::string lexicon;

  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsModelConfig {
  OfflineTtsVitsModelConfig vits;
  OfflineTtsMatchaModelConfig matcha;
  OfflineTtsKokoroModelConfig kokoro;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
};

struct OfflineTtsConfig {
  OfflineTtsModelConfig model;
  std::string rule_fsts;
  std::string rule_fars;
  int32_t max_num_sentences = 1;
  float silence_scale = 0.2;
};

struct GeneratedAudio {
  std::vector<float> samples;  // in the range [-1, 1]
  int32_t sample_rate;
};

// Return 1 to continue generating
// Return 0 to stop generating
using OfflineTtsCallback = int32_t (*)(const float *samples,
                                       int32_t num_samples, float progress,
                                       void *arg);

class SHERPA_ONNX_API OfflineTts
    : public MoveOnly<OfflineTts, SherpaMnnOfflineTts> {
 public:
  static OfflineTts Create(const OfflineTtsConfig &config);

  void Destroy(const SherpaMnnOfflineTts *p) const;

  // Return the sample rate of the generated audio
  int32_t SampleRate() const;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  int32_t NumSpeakers() const;

  // @param text A string containing words separated by spaces
  // @param sid Speaker ID. Used only for multi-speaker models, e.g., models
  //            trained using the VCTK dataset. It is not used for
  //            single-speaker models, e.g., models trained using the ljspeech
  //            dataset.
  // @param speed The speed for the generated speech. E.g., 2 means 2x faster.
  // @param callback If not NULL, it is called whenever config.max_num_sentences
  //                 sentences have been processed. The callback is called in
  //                 the current thread.
  GeneratedAudio Generate(const std::string &text, int32_t sid = 0,
                          float speed = 1.0,
                          OfflineTtsCallback callback = nullptr,
                          void *arg = nullptr) const;

 private:
  explicit OfflineTts(const SherpaMnnOfflineTts *p);
};

// ============================================================
// For Keyword Spotter
// ============================================================

struct KeywordResult {
  std::string keyword;
  std::vector<std::string> tokens;
  std::vector<float> timestamps;
  float start_time;
  std::string json;
};

struct KeywordSpotterConfig {
  FeatureConfig feat_config;
  OnlineModelConfig model_config;
  int32_t max_active_paths = 4;
  int32_t num_trailing_blanks = 1;
  float keywords_score = 1.0f;
  float keywords_threshold = 0.25f;
  std::string keywords_file;
};

class SHERPA_ONNX_API KeywordSpotter
    : public MoveOnly<KeywordSpotter, SherpaMnnKeywordSpotter> {
 public:
  static KeywordSpotter Create(const KeywordSpotterConfig &config);

  void Destroy(const SherpaMnnKeywordSpotter *p) const;

  OnlineStream CreateStream() const;

  OnlineStream CreateStream(const std::string &keywords) const;

  bool IsReady(const OnlineStream *s) const;

  void Decode(const OnlineStream *s) const;

  void Decode(const OnlineStream *ss, int32_t n) const;

  void Reset(const OnlineStream *s) const;

  KeywordResult GetResult(const OnlineStream *s) const;

 private:
  explicit KeywordSpotter(const SherpaMnnKeywordSpotter *p);
};

struct OfflineSpeechDenoiserGtcrnModelConfig {
  std::string model;
};

struct OfflineSpeechDenoiserModelConfig {
  OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  int32_t num_threads = 1;
  int32_t debug = false;
  std::string provider = "cpu";
};

struct OfflineSpeechDenoiserConfig {
  OfflineSpeechDenoiserModelConfig model;
};

struct DenoisedAudio {
  std::vector<float> samples;  // in the range [-1, 1]
  int32_t sample_rate;
};

class SHERPA_ONNX_API OfflineSpeechDenoiser
    : public MoveOnly<OfflineSpeechDenoiser, SherpaMnnOfflineSpeechDenoiser> {
 public:
  static OfflineSpeechDenoiser Create(
      const OfflineSpeechDenoiserConfig &config);

  void Destroy(const SherpaMnnOfflineSpeechDenoiser *p) const;

  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate) const;

  int32_t GetSampleRate() const;

 private:
  explicit OfflineSpeechDenoiser(const SherpaMnnOfflineSpeechDenoiser *p);
};

}  // namespace sherpa_mnn::cxx

#endif  // SHERPA_ONNX_C_API_CXX_API_H_
