// sherpa-mnn/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/c-api/c-api.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/audio-tagging.h"
#include "sherpa-mnn/csrc/circular-buffer.h"
#include "sherpa-mnn/csrc/display.h"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/keyword-spotter.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-punctuation.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/offline-speech-denoiser.h"
#include "sherpa-mnn/csrc/online-punctuation.h"
#include "sherpa-mnn/csrc/online-recognizer.h"
#include "sherpa-mnn/csrc/resample.h"
#include "sherpa-mnn/csrc/speaker-embedding-extractor.h"
#include "sherpa-mnn/csrc/speaker-embedding-manager.h"
#include "sherpa-mnn/csrc/spoken-language-identification.h"
#include "sherpa-mnn/csrc/voice-activity-detector.h"
#include "sherpa-mnn/csrc/wave-reader.h"
#include "sherpa-mnn/csrc/wave-writer.h"

#if SHERPA_MNN_ENABLE_TTS == 1
#include "sherpa-mnn/csrc/offline-tts.h"
#endif

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
#include "sherpa-mnn/csrc/offline-speaker-diarization.h"
#endif

struct SherpaMnnOnlineRecognizer {
  std::unique_ptr<sherpa_mnn::OnlineRecognizer> impl;
};

struct SherpaMnnOnlineStream {
  std::unique_ptr<sherpa_mnn::OnlineStream> impl;
  explicit SherpaMnnOnlineStream(std::unique_ptr<sherpa_mnn::OnlineStream> p)
      : impl(std::move(p)) {}
};

struct SherpaMnnDisplay {
  std::unique_ptr<sherpa_mnn::Display> impl;
};

#define SHERPA_ONNX_OR(x, y) (x ? x : y)

static sherpa_mnn::OnlineRecognizerConfig GetOnlineRecognizerConfig(
    const SherpaMnnOnlineRecognizerConfig *config) {
  sherpa_mnn::OnlineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);
  recognizer_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  recognizer_config.model_config.transducer.encoder =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");
  recognizer_config.model_config.transducer.decoder =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");
  recognizer_config.model_config.transducer.joiner =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  recognizer_config.model_config.paraformer.encoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.encoder, "");
  recognizer_config.model_config.paraformer.decoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.decoder, "");

  recognizer_config.model_config.zipformer2_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer2_ctc.model, "");

  recognizer_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  if (config->model_config.tokens_buf &&
      config->model_config.tokens_buf_size > 0) {
    recognizer_config.model_config.tokens_buf = std::string(
        config->model_config.tokens_buf, config->model_config.tokens_buf_size);
  }

  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.provider_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");

  if (recognizer_config.model_config.provider_config.provider.empty()) {
    recognizer_config.model_config.provider_config.provider = "cpu";
  }

  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);
  recognizer_config.model_config.modeling_unit =
      SHERPA_ONNX_OR(config->model_config.modeling_unit, "cjkchar");

  if (recognizer_config.model_config.modeling_unit.empty()) {
    recognizer_config.model_config.modeling_unit = "cjkchar";
  }

  recognizer_config.model_config.bpe_vocab =
      SHERPA_ONNX_OR(config->model_config.bpe_vocab, "");

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");
  if (recognizer_config.decoding_method.empty()) {
    recognizer_config.decoding_method = "greedy_search";
  }

  recognizer_config.max_active_paths =
      SHERPA_ONNX_OR(config->max_active_paths, 4);

  recognizer_config.enable_endpoint =
      SHERPA_ONNX_OR(config->enable_endpoint, 0);

  recognizer_config.endpoint_config.rule1.min_trailing_silence =
      SHERPA_ONNX_OR(config->rule1_min_trailing_silence, 2.4);

  recognizer_config.endpoint_config.rule2.min_trailing_silence =
      SHERPA_ONNX_OR(config->rule2_min_trailing_silence, 1.2);

  recognizer_config.endpoint_config.rule3.min_utterance_length =
      SHERPA_ONNX_OR(config->rule3_min_utterance_length, 20);

  recognizer_config.hotwords_file = SHERPA_ONNX_OR(config->hotwords_file, "");
  recognizer_config.hotwords_score =
      SHERPA_ONNX_OR(config->hotwords_score, 1.5);
  if (config->hotwords_buf && config->hotwords_buf_size > 0) {
    recognizer_config.hotwords_buf =
        std::string(config->hotwords_buf, config->hotwords_buf_size);
  }

  recognizer_config.blank_penalty = config->blank_penalty;

  recognizer_config.ctc_fst_decoder_config.graph =
      SHERPA_ONNX_OR(config->ctc_fst_decoder_config.graph, "");
  recognizer_config.ctc_fst_decoder_config.max_active =
      SHERPA_ONNX_OR(config->ctc_fst_decoder_config.max_active, 3000);

  recognizer_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  recognizer_config.rule_fars = SHERPA_ONNX_OR(config->rule_fars, "");

  if (config->model_config.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", recognizer_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", recognizer_config.ToString().c_str());
#endif
  }

  return recognizer_config;
}

const SherpaMnnOnlineRecognizer *SherpaMnnCreateOnlineRecognizer(
    const SherpaMnnOnlineRecognizerConfig *config) {
  sherpa_mnn::OnlineRecognizerConfig recognizer_config =
      GetOnlineRecognizerConfig(config);

  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaMnnOnlineRecognizer *recognizer = new SherpaMnnOnlineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_mnn::OnlineRecognizer>(recognizer_config);

  return recognizer;
}

void SherpaMnnDestroyOnlineRecognizer(
    const SherpaMnnOnlineRecognizer *recognizer) {
  delete recognizer;
}

const SherpaMnnOnlineStream *SherpaMnnCreateOnlineStream(
    const SherpaMnnOnlineRecognizer *recognizer) {
  SherpaMnnOnlineStream *stream =
      new SherpaMnnOnlineStream(recognizer->impl->CreateStream());
  return stream;
}

const SherpaMnnOnlineStream *SherpaMnnCreateOnlineStreamWithHotwords(
    const SherpaMnnOnlineRecognizer *recognizer, const char *hotwords) {
  SherpaMnnOnlineStream *stream =
      new SherpaMnnOnlineStream(recognizer->impl->CreateStream(hotwords));
  return stream;
}

void SherpaMnnDestroyOnlineStream(const SherpaMnnOnlineStream *stream) {
  delete stream;
}

void SherpaMnnOnlineStreamAcceptWaveform(const SherpaMnnOnlineStream *stream,
                                          int32_t sample_rate,
                                          const float *samples, int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

int32_t SherpaMnnIsOnlineStreamReady(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream) {
  return recognizer->impl->IsReady(stream->impl.get());
}

void SherpaMnnDecodeOnlineStream(const SherpaMnnOnlineRecognizer *recognizer,
                                  const SherpaMnnOnlineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void SherpaMnnDecodeMultipleOnlineStreams(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream **streams, int32_t n) {
  std::vector<sherpa_mnn::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaMnnOnlineRecognizerResult *SherpaMnnGetOnlineStreamResult(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream) {
  sherpa_mnn::OnlineRecognizerResult result =
      recognizer->impl->GetResult(stream->impl.get());
  const auto &text = result.text;

  auto r = new SherpaMnnOnlineRecognizerResult;
  memset(r, 0, sizeof(SherpaMnnOnlineRecognizerResult));

  // copy text
  char *pText = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), pText);
  pText[text.size()] = 0;
  r->text = pText;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *tokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = tokens + pos;
      memcpy(tokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty() && result.timestamps.size() == r->count) {
      r->timestamps = new float[r->count];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

    r->tokens = tokens;
  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaMnnDestroyOnlineRecognizerResult(
    const SherpaMnnOnlineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

const char *SherpaMnnGetOnlineStreamResultAsJson(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream) {
  sherpa_mnn::OnlineRecognizerResult result =
      recognizer->impl->GetResult(stream->impl.get());
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  return pJson;
}

void SherpaMnnDestroyOnlineStreamResultJson(const char *s) { delete[] s; }

void SherpaMnnOnlineStreamReset(const SherpaMnnOnlineRecognizer *recognizer,
                                 const SherpaMnnOnlineStream *stream) {
  recognizer->impl->Reset(stream->impl.get());
}

void SherpaMnnOnlineStreamInputFinished(const SherpaMnnOnlineStream *stream) {
  stream->impl->InputFinished();
}

int32_t SherpaMnnOnlineStreamIsEndpoint(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream) {
  return recognizer->impl->IsEndpoint(stream->impl.get());
}

const SherpaMnnDisplay *SherpaMnnCreateDisplay(int32_t max_word_per_line) {
  SherpaMnnDisplay *ans = new SherpaMnnDisplay;
  ans->impl = std::make_unique<sherpa_mnn::Display>(max_word_per_line);
  return ans;
}

void SherpaMnnDestroyDisplay(const SherpaMnnDisplay *display) {
  delete display;
}

void SherpaMnnPrint(const SherpaMnnDisplay *display, int32_t idx,
                     const char *s) {
  display->impl->Print(idx, s);
}

// ============================================================
// For offline ASR (i.e., non-streaming ASR)
// ============================================================
//
struct SherpaMnnOfflineRecognizer {
  std::unique_ptr<sherpa_mnn::OfflineRecognizer> impl;
};

struct SherpaMnnOfflineStream {
  std::unique_ptr<sherpa_mnn::OfflineStream> impl;
  explicit SherpaMnnOfflineStream(
      std::unique_ptr<sherpa_mnn::OfflineStream> p)
      : impl(std::move(p)) {}
};

static sherpa_mnn::OfflineRecognizerConfig GetOfflineRecognizerConfig(
    const SherpaMnnOfflineRecognizerConfig *config) {
  sherpa_mnn::OfflineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);

  recognizer_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  recognizer_config.model_config.transducer.encoder_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");

  recognizer_config.model_config.transducer.decoder_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");

  recognizer_config.model_config.transducer.joiner_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  recognizer_config.model_config.paraformer.model =
      SHERPA_ONNX_OR(config->model_config.paraformer.model, "");

  recognizer_config.model_config.nemo_ctc.model =
      SHERPA_ONNX_OR(config->model_config.nemo_ctc.model, "");

  recognizer_config.model_config.whisper.encoder =
      SHERPA_ONNX_OR(config->model_config.whisper.encoder, "");

  recognizer_config.model_config.whisper.decoder =
      SHERPA_ONNX_OR(config->model_config.whisper.decoder, "");

  recognizer_config.model_config.whisper.language =
      SHERPA_ONNX_OR(config->model_config.whisper.language, "");

  recognizer_config.model_config.whisper.task =
      SHERPA_ONNX_OR(config->model_config.whisper.task, "transcribe");
  if (recognizer_config.model_config.whisper.task.empty()) {
    recognizer_config.model_config.whisper.task = "transcribe";
  }

  recognizer_config.model_config.whisper.tail_paddings =
      SHERPA_ONNX_OR(config->model_config.whisper.tail_paddings, -1);

  recognizer_config.model_config.tdnn.model =
      SHERPA_ONNX_OR(config->model_config.tdnn.model, "");

  recognizer_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);
  recognizer_config.model_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  if (recognizer_config.model_config.provider.empty()) {
    recognizer_config.model_config.provider = "cpu";
  }

  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  recognizer_config.model_config.modeling_unit =
      SHERPA_ONNX_OR(config->model_config.modeling_unit, "cjkchar");

  if (recognizer_config.model_config.modeling_unit.empty()) {
    recognizer_config.model_config.modeling_unit = "cjkchar";
  }

  recognizer_config.model_config.bpe_vocab =
      SHERPA_ONNX_OR(config->model_config.bpe_vocab, "");

  recognizer_config.model_config.telespeech_ctc =
      SHERPA_ONNX_OR(config->model_config.telespeech_ctc, "");

  recognizer_config.model_config.sense_voice.model =
      SHERPA_ONNX_OR(config->model_config.sense_voice.model, "");

  recognizer_config.model_config.sense_voice.language =
      SHERPA_ONNX_OR(config->model_config.sense_voice.language, "");

  recognizer_config.model_config.sense_voice.use_itn =
      config->model_config.sense_voice.use_itn;

  recognizer_config.model_config.moonshine.preprocessor =
      SHERPA_ONNX_OR(config->model_config.moonshine.preprocessor, "");

  recognizer_config.model_config.moonshine.encoder =
      SHERPA_ONNX_OR(config->model_config.moonshine.encoder, "");

  recognizer_config.model_config.moonshine.uncached_decoder =
      SHERPA_ONNX_OR(config->model_config.moonshine.uncached_decoder, "");

  recognizer_config.model_config.moonshine.cached_decoder =
      SHERPA_ONNX_OR(config->model_config.moonshine.cached_decoder, "");

  recognizer_config.model_config.fire_red_asr.encoder =
      SHERPA_ONNX_OR(config->model_config.fire_red_asr.encoder, "");

  recognizer_config.model_config.fire_red_asr.decoder =
      SHERPA_ONNX_OR(config->model_config.fire_red_asr.decoder, "");

  recognizer_config.lm_config.model =
      SHERPA_ONNX_OR(config->lm_config.model, "");
  recognizer_config.lm_config.scale =
      SHERPA_ONNX_OR(config->lm_config.scale, 1.0);

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");

  if (recognizer_config.decoding_method.empty()) {
    recognizer_config.decoding_method = "greedy_search";
  }

  recognizer_config.max_active_paths =
      SHERPA_ONNX_OR(config->max_active_paths, 4);

  recognizer_config.hotwords_file = SHERPA_ONNX_OR(config->hotwords_file, "");
  recognizer_config.hotwords_score =
      SHERPA_ONNX_OR(config->hotwords_score, 1.5);

  recognizer_config.blank_penalty = config->blank_penalty;

  recognizer_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  recognizer_config.rule_fars = SHERPA_ONNX_OR(config->rule_fars, "");

  if (config->model_config.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", recognizer_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", recognizer_config.ToString().c_str());
#endif
  }

  return recognizer_config;
}

const SherpaMnnOfflineRecognizer *SherpaMnnCreateOfflineRecognizer(
    const SherpaMnnOfflineRecognizerConfig *config) {
  sherpa_mnn::OfflineRecognizerConfig recognizer_config =
      GetOfflineRecognizerConfig(config);

  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnOfflineRecognizer *recognizer = new SherpaMnnOfflineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_mnn::OfflineRecognizer>(recognizer_config);

  return recognizer;
}

void SherpaMnnOfflineRecognizerSetConfig(
    const SherpaMnnOfflineRecognizer *recognizer,
    const SherpaMnnOfflineRecognizerConfig *config) {
  sherpa_mnn::OfflineRecognizerConfig recognizer_config =
      GetOfflineRecognizerConfig(config);
  recognizer->impl->SetConfig(recognizer_config);
}

void SherpaMnnDestroyOfflineRecognizer(
    const SherpaMnnOfflineRecognizer *recognizer) {
  delete recognizer;
}

const SherpaMnnOfflineStream *SherpaMnnCreateOfflineStream(
    const SherpaMnnOfflineRecognizer *recognizer) {
  SherpaMnnOfflineStream *stream =
      new SherpaMnnOfflineStream(recognizer->impl->CreateStream());
  return stream;
}

const SherpaMnnOfflineStream *SherpaMnnCreateOfflineStreamWithHotwords(
    const SherpaMnnOfflineRecognizer *recognizer, const char *hotwords) {
  SherpaMnnOfflineStream *stream =
      new SherpaMnnOfflineStream(recognizer->impl->CreateStream(hotwords));
  return stream;
}

void SherpaMnnDestroyOfflineStream(const SherpaMnnOfflineStream *stream) {
  delete stream;
}

void SherpaMnnAcceptWaveformOffline(const SherpaMnnOfflineStream *stream,
                                     int32_t sample_rate, const float *samples,
                                     int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

void SherpaMnnDecodeOfflineStream(
    const SherpaMnnOfflineRecognizer *recognizer,
    const SherpaMnnOfflineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void SherpaMnnDecodeMultipleOfflineStreams(
    const SherpaMnnOfflineRecognizer *recognizer,
    const SherpaMnnOfflineStream **streams, int32_t n) {
  std::vector<sherpa_mnn::OfflineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaMnnOfflineRecognizerResult *SherpaMnnGetOfflineStreamResult(
    const SherpaMnnOfflineStream *stream) {
  const sherpa_mnn::OfflineRecognitionResult &result =
      stream->impl->GetResult();
  const auto &text = result.text;

  auto r = new SherpaMnnOfflineRecognizerResult;
  memset(r, 0, sizeof(SherpaMnnOfflineRecognizerResult));

  char *pText = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), pText);
  pText[text.size()] = 0;
  r->text = pText;

  // lang
  const auto &lang = result.lang;
  char *c_lang = new char[lang.size() + 1];
  std::copy(lang.begin(), lang.end(), c_lang);
  c_lang[lang.size()] = '\0';
  r->lang = c_lang;

  // emotion
  const auto &emotion = result.emotion;
  char *c_emotion = new char[emotion.size() + 1];
  std::copy(emotion.begin(), emotion.end(), c_emotion);
  c_emotion[emotion.size()] = '\0';
  r->emotion = c_emotion;

  // event
  const auto &event = result.event;
  char *c_event = new char[event.size() + 1];
  std::copy(event.begin(), event.end(), c_event);
  c_event[event.size()] = '\0';
  r->event = c_event;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *tokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = tokens + pos;
      memcpy(tokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty() && result.timestamps.size() == r->count) {
      r->timestamps = new float[r->count];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

    r->tokens = tokens;
  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaMnnDestroyOfflineRecognizerResult(
    const SherpaMnnOfflineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->timestamps;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->json;
    delete[] r->lang;
    delete[] r->emotion;
    delete[] r->event;
    delete r;
  }
}

const char *SherpaMnnGetOfflineStreamResultAsJson(
    const SherpaMnnOfflineStream *stream) {
  const sherpa_mnn::OfflineRecognitionResult &result =
      stream->impl->GetResult();
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  return pJson;
}

void SherpaMnnDestroyOfflineStreamResultJson(const char *s) { delete[] s; }

// ============================================================
// For Keyword Spot
// ============================================================

struct SherpaMnnKeywordSpotter {
  std::unique_ptr<sherpa_mnn::KeywordSpotter> impl;
};

static sherpa_mnn::KeywordSpotterConfig GetKeywordSpotterConfig(
    const SherpaMnnKeywordSpotterConfig *config) {
  sherpa_mnn::KeywordSpotterConfig spotter_config;

  spotter_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);
  spotter_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  spotter_config.model_config.transducer.encoder =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");
  spotter_config.model_config.transducer.decoder =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");
  spotter_config.model_config.transducer.joiner =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  spotter_config.model_config.paraformer.encoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.encoder, "");
  spotter_config.model_config.paraformer.decoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.decoder, "");

  spotter_config.model_config.zipformer2_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer2_ctc.model, "");

  spotter_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  if (config->model_config.tokens_buf &&
      config->model_config.tokens_buf_size > 0) {
    spotter_config.model_config.tokens_buf = std::string(
        config->model_config.tokens_buf, config->model_config.tokens_buf_size);
  }

  spotter_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  spotter_config.model_config.provider_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  if (spotter_config.model_config.provider_config.provider.empty()) {
    spotter_config.model_config.provider_config.provider = "cpu";
  }

  spotter_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  spotter_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);

  spotter_config.max_active_paths = SHERPA_ONNX_OR(config->max_active_paths, 4);

  spotter_config.num_trailing_blanks =
      SHERPA_ONNX_OR(config->num_trailing_blanks, 1);

  spotter_config.keywords_score = SHERPA_ONNX_OR(config->keywords_score, 1.0);

  spotter_config.keywords_threshold =
      SHERPA_ONNX_OR(config->keywords_threshold, 0.25);

  spotter_config.keywords_file = SHERPA_ONNX_OR(config->keywords_file, "");
  if (config->keywords_buf && config->keywords_buf_size > 0) {
    spotter_config.keywords_buf =
        std::string(config->keywords_buf, config->keywords_buf_size);
  }

  if (spotter_config.model_config.debug) {
#if OHOS
    SHERPA_ONNX_LOGE("%{public}s\n", spotter_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", spotter_config.ToString().c_str());
#endif
  }

  return spotter_config;
}

const SherpaMnnKeywordSpotter *SherpaMnnCreateKeywordSpotter(
    const SherpaMnnKeywordSpotterConfig *config) {
  auto spotter_config = GetKeywordSpotterConfig(config);
  if (!spotter_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaMnnKeywordSpotter *spotter = new SherpaMnnKeywordSpotter;

  spotter->impl = std::make_unique<sherpa_mnn::KeywordSpotter>(spotter_config);

  return spotter;
}

void SherpaMnnDestroyKeywordSpotter(const SherpaMnnKeywordSpotter *spotter) {
  delete spotter;
}

const SherpaMnnOnlineStream *SherpaMnnCreateKeywordStream(
    const SherpaMnnKeywordSpotter *spotter) {
  SherpaMnnOnlineStream *stream =
      new SherpaMnnOnlineStream(spotter->impl->CreateStream());
  return stream;
}

const SherpaMnnOnlineStream *SherpaMnnCreateKeywordStreamWithKeywords(
    const SherpaMnnKeywordSpotter *spotter, const char *keywords) {
  SherpaMnnOnlineStream *stream =
      new SherpaMnnOnlineStream(spotter->impl->CreateStream(keywords));
  return stream;
}

int32_t SherpaMnnIsKeywordStreamReady(const SherpaMnnKeywordSpotter *spotter,
                                       const SherpaMnnOnlineStream *stream) {
  return spotter->impl->IsReady(stream->impl.get());
}

void SherpaMnnDecodeKeywordStream(const SherpaMnnKeywordSpotter *spotter,
                                   const SherpaMnnOnlineStream *stream) {
  spotter->impl->DecodeStream(stream->impl.get());
}

void SherpaMnnResetKeywordStream(const SherpaMnnKeywordSpotter *spotter,
                                  const SherpaMnnOnlineStream *stream) {
  spotter->impl->Reset(stream->impl.get());
}

void SherpaMnnDecodeMultipleKeywordStreams(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream **streams, int32_t n) {
  std::vector<sherpa_mnn::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  spotter->impl->DecodeStreams(ss.data(), n);
}

const SherpaMnnKeywordResult *SherpaMnnGetKeywordResult(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream *stream) {
  const sherpa_mnn::KeywordResult &result =
      spotter->impl->GetResult(stream->impl.get());
  const auto &keyword = result.keyword;

  auto r = new SherpaMnnKeywordResult;
  memset(r, 0, sizeof(SherpaMnnKeywordResult));

  r->start_time = result.start_time;

  // copy keyword
  char *pKeyword = new char[keyword.size() + 1];
  std::copy(keyword.begin(), keyword.end(), pKeyword);
  pKeyword[keyword.size()] = 0;
  r->keyword = pKeyword;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *pTokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = pTokens + pos;
      memcpy(pTokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens = pTokens;
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty()) {
      r->timestamps = new float[result.timestamps.size()];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaMnnDestroyKeywordResult(const SherpaMnnKeywordResult *r) {
  if (r) {
    delete[] r->keyword;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

const char *SherpaMnnGetKeywordResultAsJson(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream *stream) {
  const sherpa_mnn::KeywordResult &result =
      spotter->impl->GetResult(stream->impl.get());

  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  return pJson;
}

void SherpaMnnFreeKeywordResultJson(const char *s) { delete[] s; }

// ============================================================
// For VAD
// ============================================================
//
struct SherpaMnnCircularBuffer {
  std::unique_ptr<sherpa_mnn::CircularBuffer> impl;
};

const SherpaMnnCircularBuffer *SherpaMnnCreateCircularBuffer(
    int32_t capacity) {
  SherpaMnnCircularBuffer *buffer = new SherpaMnnCircularBuffer;
  buffer->impl = std::make_unique<sherpa_mnn::CircularBuffer>(capacity);
  return buffer;
}

void SherpaMnnDestroyCircularBuffer(const SherpaMnnCircularBuffer *buffer) {
  delete buffer;
}

void SherpaMnnCircularBufferPush(const SherpaMnnCircularBuffer *buffer,
                                  const float *p, int32_t n) {
  buffer->impl->Push(p, n);
}

const float *SherpaMnnCircularBufferGet(const SherpaMnnCircularBuffer *buffer,
                                         int32_t start_index, int32_t n) {
  std::vector<float> v = buffer->impl->Get(start_index, n);

  float *p = new float[n];
  std::copy(v.begin(), v.end(), p);
  return p;
}

void SherpaMnnCircularBufferFree(const float *p) { delete[] p; }

void SherpaMnnCircularBufferPop(const SherpaMnnCircularBuffer *buffer,
                                 int32_t n) {
  buffer->impl->Pop(n);
}

int32_t SherpaMnnCircularBufferSize(const SherpaMnnCircularBuffer *buffer) {
  return buffer->impl->Size();
}

int32_t SherpaMnnCircularBufferHead(const SherpaMnnCircularBuffer *buffer) {
  return buffer->impl->Head();
}

void SherpaMnnCircularBufferReset(const SherpaMnnCircularBuffer *buffer) {
  buffer->impl->Reset();
}

struct SherpaMnnVoiceActivityDetector {
  std::unique_ptr<sherpa_mnn::VoiceActivityDetector> impl;
};

sherpa_mnn::VadModelConfig GetVadModelConfig(
    const SherpaMnnVadModelConfig *config) {
  sherpa_mnn::VadModelConfig vad_config;

  vad_config.silero_vad.model = SHERPA_ONNX_OR(config->silero_vad.model, "");
  vad_config.silero_vad.threshold =
      SHERPA_ONNX_OR(config->silero_vad.threshold, 0.5);

  vad_config.silero_vad.min_silence_duration =
      SHERPA_ONNX_OR(config->silero_vad.min_silence_duration, 0.5);

  vad_config.silero_vad.min_speech_duration =
      SHERPA_ONNX_OR(config->silero_vad.min_speech_duration, 0.25);

  vad_config.silero_vad.window_size =
      SHERPA_ONNX_OR(config->silero_vad.window_size, 512);

  vad_config.silero_vad.max_speech_duration =
      SHERPA_ONNX_OR(config->silero_vad.max_speech_duration, 20);

  vad_config.sample_rate = SHERPA_ONNX_OR(config->sample_rate, 16000);
  vad_config.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  vad_config.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  if (vad_config.provider.empty()) {
    vad_config.provider = "cpu";
  }

  vad_config.debug = SHERPA_ONNX_OR(config->debug, false);

  if (vad_config.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", vad_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", vad_config.ToString().c_str());
#endif
  }

  return vad_config;
}

const SherpaMnnVoiceActivityDetector *SherpaMnnCreateVoiceActivityDetector(
    const SherpaMnnVadModelConfig *config, float buffer_size_in_seconds) {
  auto vad_config = GetVadModelConfig(config);

  if (!vad_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnVoiceActivityDetector *p = new SherpaMnnVoiceActivityDetector;
  p->impl = std::make_unique<sherpa_mnn::VoiceActivityDetector>(
      vad_config, buffer_size_in_seconds);

  return p;
}

void SherpaMnnDestroyVoiceActivityDetector(
    const SherpaMnnVoiceActivityDetector *p) {
  delete p;
}

void SherpaMnnVoiceActivityDetectorAcceptWaveform(
    const SherpaMnnVoiceActivityDetector *p, const float *samples, int32_t n) {
  p->impl->AcceptWaveform(samples, n);
}

int32_t SherpaMnnVoiceActivityDetectorEmpty(
    const SherpaMnnVoiceActivityDetector *p) {
  return p->impl->Empty();
}

int32_t SherpaMnnVoiceActivityDetectorDetected(
    const SherpaMnnVoiceActivityDetector *p) {
  return p->impl->IsSpeechDetected();
}

void SherpaMnnVoiceActivityDetectorPop(
    const SherpaMnnVoiceActivityDetector *p) {
  p->impl->Pop();
}

void SherpaMnnVoiceActivityDetectorClear(
    const SherpaMnnVoiceActivityDetector *p) {
  p->impl->Clear();
}

const SherpaMnnSpeechSegment *SherpaMnnVoiceActivityDetectorFront(
    const SherpaMnnVoiceActivityDetector *p) {
  const sherpa_mnn::SpeechSegment &segment = p->impl->Front();

  SherpaMnnSpeechSegment *ans = new SherpaMnnSpeechSegment;
  ans->start = segment.start;
  ans->samples = new float[segment.samples.size()];
  std::copy(segment.samples.begin(), segment.samples.end(), ans->samples);
  ans->n = segment.samples.size();

  return ans;
}

void SherpaMnnDestroySpeechSegment(const SherpaMnnSpeechSegment *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}

void SherpaMnnVoiceActivityDetectorReset(
    const SherpaMnnVoiceActivityDetector *p) {
  p->impl->Reset();
}

void SherpaMnnVoiceActivityDetectorFlush(
    const SherpaMnnVoiceActivityDetector *p) {
  p->impl->Flush();
}

#if SHERPA_MNN_ENABLE_TTS == 1
struct SherpaMnnOfflineTts {
  std::unique_ptr<sherpa_mnn::OfflineTts> impl;
};

static sherpa_mnn::OfflineTtsConfig GetOfflineTtsConfig(
    const SherpaMnnOfflineTtsConfig *config) {
  sherpa_mnn::OfflineTtsConfig tts_config;

  // vits
  tts_config.model.vits.model = SHERPA_ONNX_OR(config->model.vits.model, "");
  tts_config.model.vits.lexicon =
      SHERPA_ONNX_OR(config->model.vits.lexicon, "");
  tts_config.model.vits.tokens = SHERPA_ONNX_OR(config->model.vits.tokens, "");
  tts_config.model.vits.data_dir =
      SHERPA_ONNX_OR(config->model.vits.data_dir, "");
  tts_config.model.vits.noise_scale =
      SHERPA_ONNX_OR(config->model.vits.noise_scale, 0.667);
  tts_config.model.vits.noise_scale_w =
      SHERPA_ONNX_OR(config->model.vits.noise_scale_w, 0.8);
  tts_config.model.vits.length_scale =
      SHERPA_ONNX_OR(config->model.vits.length_scale, 1.0);
  tts_config.model.vits.dict_dir =
      SHERPA_ONNX_OR(config->model.vits.dict_dir, "");

  // matcha
  tts_config.model.matcha.acoustic_model =
      SHERPA_ONNX_OR(config->model.matcha.acoustic_model, "");
  tts_config.model.matcha.vocoder =
      SHERPA_ONNX_OR(config->model.matcha.vocoder, "");
  tts_config.model.matcha.lexicon =
      SHERPA_ONNX_OR(config->model.matcha.lexicon, "");
  tts_config.model.matcha.tokens =
      SHERPA_ONNX_OR(config->model.matcha.tokens, "");
  tts_config.model.matcha.data_dir =
      SHERPA_ONNX_OR(config->model.matcha.data_dir, "");
  tts_config.model.matcha.noise_scale =
      SHERPA_ONNX_OR(config->model.matcha.noise_scale, 0.667);
  tts_config.model.matcha.length_scale =
      SHERPA_ONNX_OR(config->model.matcha.length_scale, 1.0);
  tts_config.model.matcha.dict_dir =
      SHERPA_ONNX_OR(config->model.matcha.dict_dir, "");

  // kokoro
  tts_config.model.kokoro.model =
      SHERPA_ONNX_OR(config->model.kokoro.model, "");
  tts_config.model.kokoro.voices =
      SHERPA_ONNX_OR(config->model.kokoro.voices, "");
  tts_config.model.kokoro.tokens =
      SHERPA_ONNX_OR(config->model.kokoro.tokens, "");
  tts_config.model.kokoro.data_dir =
      SHERPA_ONNX_OR(config->model.kokoro.data_dir, "");
  tts_config.model.kokoro.length_scale =
      SHERPA_ONNX_OR(config->model.kokoro.length_scale, 1.0);
  tts_config.model.kokoro.dict_dir =
      SHERPA_ONNX_OR(config->model.kokoro.dict_dir, "");
  tts_config.model.kokoro.lexicon =
      SHERPA_ONNX_OR(config->model.kokoro.lexicon, "");

  tts_config.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  tts_config.model.debug = config->model.debug;
  tts_config.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  if (tts_config.model.provider.empty()) {
    tts_config.model.provider = "cpu";
  }

  tts_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  tts_config.rule_fars = SHERPA_ONNX_OR(config->rule_fars, "");
  tts_config.max_num_sentences = SHERPA_ONNX_OR(config->max_num_sentences, 1);
  tts_config.silence_scale = SHERPA_ONNX_OR(config->silence_scale, 0.2);

  if (tts_config.model.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", tts_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", tts_config.ToString().c_str());
#endif
  }

  return tts_config;
}

const SherpaMnnOfflineTts *SherpaMnnCreateOfflineTts(
    const SherpaMnnOfflineTtsConfig *config) {
  auto tts_config = GetOfflineTtsConfig(config);

  if (!tts_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnOfflineTts *tts = new SherpaMnnOfflineTts;

  tts->impl = std::make_unique<sherpa_mnn::OfflineTts>(tts_config);

  return tts;
}

void SherpaMnnDestroyOfflineTts(const SherpaMnnOfflineTts *tts) {
  delete tts;
}

int32_t SherpaMnnOfflineTtsSampleRate(const SherpaMnnOfflineTts *tts) {
  return tts->impl->SampleRate();
}

int32_t SherpaMnnOfflineTtsNumSpeakers(const SherpaMnnOfflineTts *tts) {
  return tts->impl->NumSpeakers();
}

static const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerateInternal(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    std::function<int32_t(const float *, int32_t, float)> callback) {
  sherpa_mnn::GeneratedAudio audio =
      tts->impl->Generate(text, sid, speed, callback);

  if (audio.samples.empty()) {
    return nullptr;
  }

  SherpaMnnGeneratedAudio *ans = new SherpaMnnGeneratedAudio;

  float *samples = new float[audio.samples.size()];
  std::copy(audio.samples.begin(), audio.samples.end(), samples);

  ans->samples = samples;
  ans->n = audio.samples.size();
  ans->sample_rate = audio.sample_rate;

  return ans;
}

const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerate(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid,
    float speed) {
  return SherpaMnnOfflineTtsGenerateInternal(tts, text, sid, speed, nullptr);
}

const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerateWithCallback(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioCallback callback) {
  auto wrapper = [callback](const float *samples, int32_t n,
                            float /*progress*/) {
    return callback(samples, n);
  };

  return SherpaMnnOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithProgressCallback(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioProgressCallback callback) {
  auto wrapper = [callback](const float *samples, int32_t n, float progress) {
    return callback(samples, n, progress);
  };
  return SherpaMnnOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithProgressCallbackWithArg(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioProgressCallbackWithArg callback, void *arg) {
  auto wrapper = [callback, arg](const float *samples, int32_t n,
                                 float progress) {
    return callback(samples, n, progress, arg);
  };
  return SherpaMnnOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerateWithCallbackWithArg(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioCallbackWithArg callback, void *arg) {
  auto wrapper = [callback, arg](const float *samples, int32_t n,
                                 float /*progress*/) {
    return callback(samples, n, arg);
  };

  return SherpaMnnOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

void SherpaMnnDestroyOfflineTtsGeneratedAudio(
    const SherpaMnnGeneratedAudio *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}
#else
const SherpaMnnOfflineTts *SherpaMnnCreateOfflineTts(
    const SherpaMnnOfflineTtsConfig *config) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

void SherpaMnnDestroyOfflineTts(const SherpaMnnOfflineTts *tts) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
}

int32_t SherpaMnnOfflineTtsSampleRate(const SherpaMnnOfflineTts *tts) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return 0;
}

int32_t SherpaMnnOfflineTtsNumSpeakers(const SherpaMnnOfflineTts *tts) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return 0;
}

const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerate(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid,
    float speed) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerateWithCallback(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioCallback callback) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithProgressCallback(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioProgressCallback callback) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithProgressCallbackWithArg(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioProgressCallbackWithArg callback, void *arg) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerateWithCallbackWithArg(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioCallbackWithArg callback, void *arg) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

void SherpaMnnDestroyOfflineTtsGeneratedAudio(
    const SherpaMnnGeneratedAudio *p) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
}

#endif  // SHERPA_MNN_ENABLE_TTS == 1

int32_t SherpaMnnWriteWave(const float *samples, int32_t n,
                            int32_t sample_rate, const char *filename) {
  return sherpa_mnn::WriteWave(filename, sample_rate, samples, n);
}

int64_t SherpaMnnWaveFileSize(int32_t n_samples) {
  return sherpa_mnn::WaveFileSize(n_samples);
}

SHERPA_ONNX_API void SherpaMnnWriteWaveToBuffer(const float *samples,
                                                 int32_t n, int32_t sample_rate,
                                                 char *buffer) {
  sherpa_mnn::WriteWave(buffer, sample_rate, samples, n);
}

const SherpaMnnWave *SherpaMnnReadWave(const char *filename) {
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_mnn::ReadWave(filename, &sample_rate, &is_ok);
  if (!is_ok) {
    return nullptr;
  }

  float *c_samples = new float[samples.size()];
  std::copy(samples.begin(), samples.end(), c_samples);

  SherpaMnnWave *wave = new SherpaMnnWave;
  wave->samples = c_samples;
  wave->sample_rate = sample_rate;
  wave->num_samples = samples.size();
  return wave;
}

const SherpaMnnWave *SherpaMnnReadWaveFromBinaryData(const char *data,
                                                       int32_t n) {
  int32_t sample_rate = -1;
  bool is_ok = false;

  std::istrstream is(data, n);

  std::vector<float> samples = sherpa_mnn::ReadWave(is, &sample_rate, &is_ok);
  if (!is_ok) {
    return nullptr;
  }

  float *c_samples = new float[samples.size()];
  std::copy(samples.begin(), samples.end(), c_samples);

  SherpaMnnWave *wave = new SherpaMnnWave;
  wave->samples = c_samples;
  wave->sample_rate = sample_rate;
  wave->num_samples = samples.size();
  return wave;
}

void SherpaMnnFreeWave(const SherpaMnnWave *wave) {
  if (wave) {
    delete[] wave->samples;
    delete wave;
  }
}

struct SherpaMnnSpokenLanguageIdentification {
  std::unique_ptr<sherpa_mnn::SpokenLanguageIdentification> impl;
};

const SherpaMnnSpokenLanguageIdentification *
SherpaMnnCreateSpokenLanguageIdentification(
    const SherpaMnnSpokenLanguageIdentificationConfig *config) {
  sherpa_mnn::SpokenLanguageIdentificationConfig slid_config;
  slid_config.whisper.encoder = SHERPA_ONNX_OR(config->whisper.encoder, "");
  slid_config.whisper.decoder = SHERPA_ONNX_OR(config->whisper.decoder, "");
  slid_config.whisper.tail_paddings =
      SHERPA_ONNX_OR(config->whisper.tail_paddings, -1);
  slid_config.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  slid_config.debug = config->debug;
  slid_config.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  if (slid_config.provider.empty()) {
    slid_config.provider = "cpu";
  }

  if (slid_config.debug) {
    SHERPA_ONNX_LOGE("%s\n", slid_config.ToString().c_str());
  }

  if (!slid_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnSpokenLanguageIdentification *slid =
      new SherpaMnnSpokenLanguageIdentification;
  slid->impl =
      std::make_unique<sherpa_mnn::SpokenLanguageIdentification>(slid_config);

  return slid;
}

void SherpaMnnDestroySpokenLanguageIdentification(
    const SherpaMnnSpokenLanguageIdentification *slid) {
  delete slid;
}

SherpaMnnOfflineStream *
SherpaMnnSpokenLanguageIdentificationCreateOfflineStream(
    const SherpaMnnSpokenLanguageIdentification *slid) {
  SherpaMnnOfflineStream *stream =
      new SherpaMnnOfflineStream(slid->impl->CreateStream());
  return stream;
}

const SherpaMnnSpokenLanguageIdentificationResult *
SherpaMnnSpokenLanguageIdentificationCompute(
    const SherpaMnnSpokenLanguageIdentification *slid,
    const SherpaMnnOfflineStream *s) {
  std::string lang = slid->impl->Compute(s->impl.get());
  char *c_lang = new char[lang.size() + 1];
  std::copy(lang.begin(), lang.end(), c_lang);
  c_lang[lang.size()] = '\0';
  SherpaMnnSpokenLanguageIdentificationResult *r =
      new SherpaMnnSpokenLanguageIdentificationResult;
  r->lang = c_lang;
  return r;
}

void SherpaMnnDestroySpokenLanguageIdentificationResult(
    const SherpaMnnSpokenLanguageIdentificationResult *r) {
  if (r) {
    delete[] r->lang;
    delete r;
  }
}

struct SherpaMnnSpeakerEmbeddingExtractor {
  std::unique_ptr<sherpa_mnn::SpeakerEmbeddingExtractor> impl;
};

static sherpa_mnn::SpeakerEmbeddingExtractorConfig
GetSpeakerEmbeddingExtractorConfig(
    const SherpaMnnSpeakerEmbeddingExtractorConfig *config) {
  sherpa_mnn::SpeakerEmbeddingExtractorConfig c;
  c.model = SHERPA_ONNX_OR(config->model, "");

  c.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  c.debug = SHERPA_ONNX_OR(config->debug, 0);
  c.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  if (c.provider.empty()) {
    c.provider = "cpu";
  }

  if (config->debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", c.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", c.ToString().c_str());
#endif
  }

  return c;
}

const SherpaMnnSpeakerEmbeddingExtractor *
SherpaMnnCreateSpeakerEmbeddingExtractor(
    const SherpaMnnSpeakerEmbeddingExtractorConfig *config) {
  auto c = GetSpeakerEmbeddingExtractorConfig(config);

  if (!c.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  auto p = new SherpaMnnSpeakerEmbeddingExtractor;

  p->impl = std::make_unique<sherpa_mnn::SpeakerEmbeddingExtractor>(c);

  return p;
}

void SherpaMnnDestroySpeakerEmbeddingExtractor(
    const SherpaMnnSpeakerEmbeddingExtractor *p) {
  delete p;
}

int32_t SherpaMnnSpeakerEmbeddingExtractorDim(
    const SherpaMnnSpeakerEmbeddingExtractor *p) {
  return p->impl->Dim();
}

const SherpaMnnOnlineStream *SherpaMnnSpeakerEmbeddingExtractorCreateStream(
    const SherpaMnnSpeakerEmbeddingExtractor *p) {
  SherpaMnnOnlineStream *stream =
      new SherpaMnnOnlineStream(p->impl->CreateStream());
  return stream;
}

int32_t SherpaMnnSpeakerEmbeddingExtractorIsReady(
    const SherpaMnnSpeakerEmbeddingExtractor *p,
    const SherpaMnnOnlineStream *s) {
  return p->impl->IsReady(s->impl.get());
}

const float *SherpaMnnSpeakerEmbeddingExtractorComputeEmbedding(
    const SherpaMnnSpeakerEmbeddingExtractor *p,
    const SherpaMnnOnlineStream *s) {
  std::vector<float> v = p->impl->Compute(s->impl.get());
  float *ans = new float[v.size()];
  std::copy(v.begin(), v.end(), ans);
  return ans;
}

void SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(const float *v) {
  delete[] v;
}

struct SherpaMnnSpeakerEmbeddingManager {
  std::unique_ptr<sherpa_mnn::SpeakerEmbeddingManager> impl;
};

const SherpaMnnSpeakerEmbeddingManager *
SherpaMnnCreateSpeakerEmbeddingManager(int32_t dim) {
  auto p = new SherpaMnnSpeakerEmbeddingManager;
  p->impl = std::make_unique<sherpa_mnn::SpeakerEmbeddingManager>(dim);
  return p;
}

void SherpaMnnDestroySpeakerEmbeddingManager(
    const SherpaMnnSpeakerEmbeddingManager *p) {
  delete p;
}

int32_t SherpaMnnSpeakerEmbeddingManagerAdd(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float *v) {
  return p->impl->Add(name, v);
}

int32_t SherpaMnnSpeakerEmbeddingManagerAddList(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float **v) {
  int32_t n = 0;
  auto q = v;
  while (q && q[0]) {
    ++n;
    ++q;
  }

  if (n == 0) {
    SHERPA_ONNX_LOGE("Empty embedding!");
    return 0;
  }

  std::vector<std::vector<float>> vec(n);
  int32_t dim = p->impl->Dim();

  for (int32_t i = 0; i != n; ++i) {
    vec[i] = std::vector<float>(v[i], v[i] + dim);
  }

  return p->impl->Add(name, vec);
}

int32_t SherpaMnnSpeakerEmbeddingManagerAddListFlattened(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float *v, int32_t n) {
  std::vector<std::vector<float>> vec(n);

  int32_t dim = p->impl->Dim();

  for (int32_t i = 0; i != n; ++i, v += dim) {
    vec[i] = std::vector<float>(v, v + dim);
  }

  return p->impl->Add(name, vec);
}

int32_t SherpaMnnSpeakerEmbeddingManagerRemove(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name) {
  return p->impl->Remove(name);
}

const char *SherpaMnnSpeakerEmbeddingManagerSearch(
    const SherpaMnnSpeakerEmbeddingManager *p, const float *v,
    float threshold) {
  auto r = p->impl->Search(v, threshold);
  if (r.empty()) {
    return nullptr;
  }

  char *name = new char[r.size() + 1];
  std::copy(r.begin(), r.end(), name);
  name[r.size()] = '\0';

  return name;
}

void SherpaMnnSpeakerEmbeddingManagerFreeSearch(const char *name) {
  delete[] name;
}

const SherpaMnnSpeakerEmbeddingManagerBestMatchesResult *
SherpaMnnSpeakerEmbeddingManagerGetBestMatches(
    const SherpaMnnSpeakerEmbeddingManager *p, const float *v, float threshold,
    int32_t n) {
  auto matches = p->impl->GetBestMatches(v, threshold, n);

  if (matches.empty()) {
    return nullptr;
  }

  auto resultMatches =
      new SherpaMnnSpeakerEmbeddingManagerSpeakerMatch[matches.size()];
  for (int i = 0; i < matches.size(); ++i) {
    resultMatches[i].score = matches[i].score;

    char *name = new char[matches[i].name.size() + 1];
    std::copy(matches[i].name.begin(), matches[i].name.end(), name);
    name[matches[i].name.size()] = '\0';

    resultMatches[i].name = name;
  }

  auto *result = new SherpaMnnSpeakerEmbeddingManagerBestMatchesResult();
  result->count = matches.size();
  result->matches = resultMatches;

  return result;
}

void SherpaMnnSpeakerEmbeddingManagerFreeBestMatches(
    const SherpaMnnSpeakerEmbeddingManagerBestMatchesResult *r) {
  if (r == nullptr) {
    return;
  }

  for (int32_t i = 0; i < r->count; ++i) {
    delete[] r->matches[i].name;
  }
  delete[] r->matches;
  delete r;
}

int32_t SherpaMnnSpeakerEmbeddingManagerVerify(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float *v, float threshold) {
  return p->impl->Verify(name, v, threshold);
}

int32_t SherpaMnnSpeakerEmbeddingManagerContains(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name) {
  return p->impl->Contains(name);
}

int32_t SherpaMnnSpeakerEmbeddingManagerNumSpeakers(
    const SherpaMnnSpeakerEmbeddingManager *p) {
  return p->impl->NumSpeakers();
}

const char *const *SherpaMnnSpeakerEmbeddingManagerGetAllSpeakers(
    const SherpaMnnSpeakerEmbeddingManager *manager) {
  std::vector<std::string> all_speakers = manager->impl->GetAllSpeakers();
  int32_t num_speakers = all_speakers.size();
  char **p = new char *[num_speakers + 1];
  p[num_speakers] = nullptr;

  int32_t i = 0;
  for (const auto &name : all_speakers) {
    p[i] = new char[name.size() + 1];
    std::copy(name.begin(), name.end(), p[i]);
    p[i][name.size()] = '\0';

    i += 1;
  }
  return p;
}

void SherpaMnnSpeakerEmbeddingManagerFreeAllSpeakers(
    const char *const *names) {
  auto p = names;

  while (p && p[0]) {
    delete[] p[0];
    ++p;
  }

  delete[] names;
}

struct SherpaMnnAudioTagging {
  std::unique_ptr<sherpa_mnn::AudioTagging> impl;
};

const SherpaMnnAudioTagging *SherpaMnnCreateAudioTagging(
    const SherpaMnnAudioTaggingConfig *config) {
  sherpa_mnn::AudioTaggingConfig ac;
  ac.model.zipformer.model = SHERPA_ONNX_OR(config->model.zipformer.model, "");
  ac.model.ced = SHERPA_ONNX_OR(config->model.ced, "");
  ac.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  ac.model.debug = config->model.debug;
  ac.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  if (ac.model.provider.empty()) {
    ac.model.provider = "cpu";
  }

  ac.labels = SHERPA_ONNX_OR(config->labels, "");
  ac.top_k = SHERPA_ONNX_OR(config->top_k, 5);

  if (ac.model.debug) {
    SHERPA_ONNX_LOGE("%s\n", ac.ToString().c_str());
  }

  if (!ac.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnAudioTagging *tagger = new SherpaMnnAudioTagging;
  tagger->impl = std::make_unique<sherpa_mnn::AudioTagging>(ac);

  return tagger;
}

void SherpaMnnDestroyAudioTagging(const SherpaMnnAudioTagging *tagger) {
  delete tagger;
}

const SherpaMnnOfflineStream *SherpaMnnAudioTaggingCreateOfflineStream(
    const SherpaMnnAudioTagging *tagger) {
  const SherpaMnnOfflineStream *stream =
      new SherpaMnnOfflineStream(tagger->impl->CreateStream());
  return stream;
}

const SherpaMnnAudioEvent *const *SherpaMnnAudioTaggingCompute(
    const SherpaMnnAudioTagging *tagger, const SherpaMnnOfflineStream *s,
    int32_t top_k) {
  std::vector<sherpa_mnn::AudioEvent> events =
      tagger->impl->Compute(s->impl.get(), top_k);

  int32_t n = static_cast<int32_t>(events.size());
  SherpaMnnAudioEvent **ans = new SherpaMnnAudioEvent *[n + 1];
  ans[n] = nullptr;

  int32_t i = 0;
  for (const auto &e : events) {
    SherpaMnnAudioEvent *p = new SherpaMnnAudioEvent;

    char *name = new char[e.name.size() + 1];
    std::copy(e.name.begin(), e.name.end(), name);
    name[e.name.size()] = 0;

    p->name = name;

    p->index = e.index;
    p->prob = e.prob;

    ans[i] = p;
    i += 1;
  }

  return ans;
}

void SherpaMnnAudioTaggingFreeResults(
    const SherpaMnnAudioEvent *const *events) {
  auto p = events;

  while (p && *p) {
    auto e = *p;

    delete[] e->name;
    delete e;

    ++p;
  }

  delete[] events;
}

struct SherpaMnnOfflinePunctuation {
  std::unique_ptr<sherpa_mnn::OfflinePunctuation> impl;
};

const SherpaMnnOfflinePunctuation *SherpaMnnCreateOfflinePunctuation(
    const SherpaMnnOfflinePunctuationConfig *config) {
  sherpa_mnn::OfflinePunctuationConfig c;
  c.model.ct_transformer = SHERPA_ONNX_OR(config->model.ct_transformer, "");
  c.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  c.model.debug = config->model.debug;
  c.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  if (c.model.provider.empty()) {
    c.model.provider = "cpu";
  }

  if (c.model.debug) {
    SHERPA_ONNX_LOGE("%s\n", c.ToString().c_str());
  }

  if (!c.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnOfflinePunctuation *punct = new SherpaMnnOfflinePunctuation;
  punct->impl = std::make_unique<sherpa_mnn::OfflinePunctuation>(c);

  return punct;
}

void SherpaMnnDestroyOfflinePunctuation(
    const SherpaMnnOfflinePunctuation *punct) {
  delete punct;
}

const char *SherpaOfflinePunctuationAddPunct(
    const SherpaMnnOfflinePunctuation *punct, const char *text) {
  std::string text_with_punct = punct->impl->AddPunctuation(text);

  char *ans = new char[text_with_punct.size() + 1];
  std::copy(text_with_punct.begin(), text_with_punct.end(), ans);
  ans[text_with_punct.size()] = 0;

  return ans;
}

void SherpaOfflinePunctuationFreeText(const char *text) { delete[] text; }

struct SherpaMnnOnlinePunctuation {
  std::unique_ptr<sherpa_mnn::OnlinePunctuation> impl;
};

const SherpaMnnOnlinePunctuation *SherpaMnnCreateOnlinePunctuation(
    const SherpaMnnOnlinePunctuationConfig *config) {
  auto p = new SherpaMnnOnlinePunctuation;
  try {
    sherpa_mnn::OnlinePunctuationConfig punctuation_config;
    punctuation_config.model.cnn_bilstm =
        SHERPA_ONNX_OR(config->model.cnn_bilstm, "");
    punctuation_config.model.bpe_vocab =
        SHERPA_ONNX_OR(config->model.bpe_vocab, "");
    punctuation_config.model.num_threads =
        SHERPA_ONNX_OR(config->model.num_threads, 1);
    punctuation_config.model.debug = config->model.debug;
    punctuation_config.model.provider =
        SHERPA_ONNX_OR(config->model.provider, "cpu");

    p->impl =
        std::make_unique<sherpa_mnn::OnlinePunctuation>(punctuation_config);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to create online punctuation: %s", e.what());
    delete p;
    return nullptr;
  }
  return p;
}

void SherpaMnnDestroyOnlinePunctuation(const SherpaMnnOnlinePunctuation *p) {
  delete p;
}

const char *SherpaMnnOnlinePunctuationAddPunct(
    const SherpaMnnOnlinePunctuation *punctuation, const char *text) {
  if (!punctuation || !text) return nullptr;

  try {
    std::string s = punctuation->impl->AddPunctuationWithCase(text);
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = '\0';
    return p;
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to add punctuation: %s", e.what());
    return nullptr;
  }
}

void SherpaMnnOnlinePunctuationFreeText(const char *text) { delete[] text; }

struct SherpaMnnLinearResampler {
  std::unique_ptr<sherpa_mnn::LinearResample> impl;
};

const SherpaMnnLinearResampler *SherpaMnnCreateLinearResampler(
    int32_t samp_rate_in_hz, int32_t samp_rate_out_hz, float filter_cutoff_hz,
    int32_t num_zeros) {
  SherpaMnnLinearResampler *p = new SherpaMnnLinearResampler;
  p->impl = std::make_unique<sherpa_mnn::LinearResample>(
      samp_rate_in_hz, samp_rate_out_hz, filter_cutoff_hz, num_zeros);

  return p;
}

void SherpaMnnDestroyLinearResampler(const SherpaMnnLinearResampler *p) {
  delete p;
}

const SherpaMnnResampleOut *SherpaMnnLinearResamplerResample(
    const SherpaMnnLinearResampler *p, const float *input, int32_t input_dim,
    int32_t flush) {
  std::vector<float> o;
  p->impl->Resample(input, input_dim, flush, &o);

  float *s = new float[o.size()];
  std::copy(o.begin(), o.end(), s);

  SherpaMnnResampleOut *ans = new SherpaMnnResampleOut;
  ans->samples = s;
  ans->n = static_cast<int32_t>(o.size());

  return ans;
}

void SherpaMnnLinearResamplerResampleFree(const SherpaMnnResampleOut *p) {
  delete[] p->samples;
  delete p;
}

int32_t SherpaMnnLinearResamplerResampleGetInputSampleRate(
    const SherpaMnnLinearResampler *p) {
  return p->impl->GetInputSamplingRate();
}

int32_t SherpaMnnLinearResamplerResampleGetOutputSampleRate(
    const SherpaMnnLinearResampler *p) {
  return p->impl->GetOutputSamplingRate();
}

void SherpaMnnLinearResamplerReset(SherpaMnnLinearResampler *p) {
  p->impl->Reset();
}

int32_t SherpaMnnFileExists(const char *filename) {
  return sherpa_mnn::FileExists(filename);
}

struct SherpaMnnOfflineSpeechDenoiser {
  std::unique_ptr<sherpa_mnn::OfflineSpeechDenoiser> impl;
};

static sherpa_mnn::OfflineSpeechDenoiserConfig GetOfflineSpeechDenoiserConfig(
    const SherpaMnnOfflineSpeechDenoiserConfig *config) {
  sherpa_mnn::OfflineSpeechDenoiserConfig c;
  c.model.gtcrn.model = SHERPA_ONNX_OR(config->model.gtcrn.model, "");
  c.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  c.model.debug = config->model.debug;
  c.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");

  if (c.model.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", c.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", c.ToString().c_str());
#endif
  }

  return c;
}

const SherpaMnnOfflineSpeechDenoiser *SherpaMnnCreateOfflineSpeechDenoiser(
    const SherpaMnnOfflineSpeechDenoiserConfig *config) {
  auto sd_config = GetOfflineSpeechDenoiserConfig(config);

  if (!sd_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnOfflineSpeechDenoiser *sd = new SherpaMnnOfflineSpeechDenoiser;

  sd->impl = std::make_unique<sherpa_mnn::OfflineSpeechDenoiser>(sd_config);

  return sd;
}

void SherpaMnnDestroyOfflineSpeechDenoiser(
    const SherpaMnnOfflineSpeechDenoiser *sd) {
  delete sd;
}

int32_t SherpaMnnOfflineSpeechDenoiserGetSampleRate(
    const SherpaMnnOfflineSpeechDenoiser *sd) {
  return sd->impl->GetSampleRate();
}

const SherpaMnnDenoisedAudio *SherpaMnnOfflineSpeechDenoiserRun(
    const SherpaMnnOfflineSpeechDenoiser *sd, const float *samples, int32_t n,
    int32_t sample_rate) {
  auto audio = sd->impl->Run(samples, n, sample_rate);

  auto ans = new SherpaMnnDenoisedAudio;

  float *denoised_samples = new float[audio.samples.size()];
  std::copy(audio.samples.begin(), audio.samples.end(), denoised_samples);

  ans->samples = denoised_samples;
  ans->n = audio.samples.size();
  ans->sample_rate = audio.sample_rate;

  return ans;
}

void SherpaMnnDestroyDenoisedAudio(const SherpaMnnDenoisedAudio *p) {
  delete[] p->samples;
  delete p;
}

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1

struct SherpaMnnOfflineSpeakerDiarization {
  std::unique_ptr<sherpa_mnn::OfflineSpeakerDiarization> impl;
};

struct SherpaMnnOfflineSpeakerDiarizationResult {
  sherpa_mnn::OfflineSpeakerDiarizationResult impl;
};

static sherpa_mnn::OfflineSpeakerDiarizationConfig
GetOfflineSpeakerDiarizationConfig(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config) {
  sherpa_mnn::OfflineSpeakerDiarizationConfig sd_config;

  sd_config.segmentation.pyannote.model =
      SHERPA_ONNX_OR(config->segmentation.pyannote.model, "");
  sd_config.segmentation.num_threads =
      SHERPA_ONNX_OR(config->segmentation.num_threads, 1);
  sd_config.segmentation.debug = config->segmentation.debug;
  sd_config.segmentation.provider =
      SHERPA_ONNX_OR(config->segmentation.provider, "cpu");
  if (sd_config.segmentation.provider.empty()) {
    sd_config.segmentation.provider = "cpu";
  }

  sd_config.embedding.model = SHERPA_ONNX_OR(config->embedding.model, "");
  sd_config.embedding.num_threads =
      SHERPA_ONNX_OR(config->embedding.num_threads, 1);
  sd_config.embedding.debug = config->embedding.debug;
  sd_config.embedding.provider =
      SHERPA_ONNX_OR(config->embedding.provider, "cpu");
  if (sd_config.embedding.provider.empty()) {
    sd_config.embedding.provider = "cpu";
  }

  sd_config.clustering.num_clusters =
      SHERPA_ONNX_OR(config->clustering.num_clusters, -1);

  sd_config.clustering.threshold =
      SHERPA_ONNX_OR(config->clustering.threshold, 0.5);

  sd_config.min_duration_on = SHERPA_ONNX_OR(config->min_duration_on, 0.3);

  sd_config.min_duration_off = SHERPA_ONNX_OR(config->min_duration_off, 0.5);

  if (sd_config.segmentation.debug || sd_config.embedding.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", sd_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", sd_config.ToString().c_str());
#endif
  }

  return sd_config;
}

const SherpaMnnOfflineSpeakerDiarization *
SherpaMnnCreateOfflineSpeakerDiarization(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config) {
  auto sd_config = GetOfflineSpeakerDiarizationConfig(config);

  if (!sd_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaMnnOfflineSpeakerDiarization *sd =
      new SherpaMnnOfflineSpeakerDiarization;

  sd->impl =
      std::make_unique<sherpa_mnn::OfflineSpeakerDiarization>(sd_config);

  return sd;
}

void SherpaMnnDestroyOfflineSpeakerDiarization(
    const SherpaMnnOfflineSpeakerDiarization *sd) {
  delete sd;
}

int32_t SherpaMnnOfflineSpeakerDiarizationGetSampleRate(
    const SherpaMnnOfflineSpeakerDiarization *sd) {
  return sd->impl->SampleRate();
}

void SherpaMnnOfflineSpeakerDiarizationSetConfig(
    const SherpaMnnOfflineSpeakerDiarization *sd,
    const SherpaMnnOfflineSpeakerDiarizationConfig *config) {
  sherpa_mnn::OfflineSpeakerDiarizationConfig sd_config;

  sd_config.clustering.num_clusters =
      SHERPA_ONNX_OR(config->clustering.num_clusters, -1);

  sd_config.clustering.threshold =
      SHERPA_ONNX_OR(config->clustering.threshold, 0.5);

  sd->impl->SetConfig(sd_config);
}

int32_t SherpaMnnOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  return r->impl.NumSpeakers();
}

int32_t SherpaMnnOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  return r->impl.NumSegments();
}

const SherpaMnnOfflineSpeakerDiarizationSegment *
SherpaMnnOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  if (r->impl.NumSegments() == 0) {
    return nullptr;
  }

  auto segments = r->impl.SortByStartTime();

  int32_t n = segments.size();
  SherpaMnnOfflineSpeakerDiarizationSegment *ans =
      new SherpaMnnOfflineSpeakerDiarizationSegment[n];

  for (int32_t i = 0; i != n; ++i) {
    const auto &s = segments[i];

    ans[i].start = s.Start();
    ans[i].end = s.End();
    ans[i].speaker = s.Speaker();
  }

  return ans;
}

void SherpaMnnOfflineSpeakerDiarizationDestroySegment(
    const SherpaMnnOfflineSpeakerDiarizationSegment *s) {
  delete[] s;
}

const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcess(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n) {
  auto ans = new SherpaMnnOfflineSpeakerDiarizationResult;
  ans->impl = sd->impl->Process(samples, n);

  return ans;
}

void SherpaMnnOfflineSpeakerDiarizationDestroyResult(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  delete r;
}

const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaMnnOfflineSpeakerDiarizationProgressCallback callback,
    void *arg) {
  auto ans = new SherpaMnnOfflineSpeakerDiarizationResult;
  ans->impl = sd->impl->Process(samples, n, callback, arg);

  return ans;
}

const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcessWithCallbackNoArg(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n,
    SherpaMnnOfflineSpeakerDiarizationProgressCallbackNoArg callback) {
  auto wrapper = [callback](int32_t num_processed_chunks,
                            int32_t num_total_chunks, void *) {
    return callback(num_processed_chunks, num_total_chunks);
  };

  auto ans = new SherpaMnnOfflineSpeakerDiarizationResult;
  ans->impl = sd->impl->Process(samples, n, wrapper);

  return ans;
}
#else

const SherpaMnnOfflineSpeakerDiarization *
SherpaMnnCreateOfflineSpeakerDiarization(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

void SherpaMnnDestroyOfflineSpeakerDiarization(
    const SherpaMnnOfflineSpeakerDiarization *sd) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
}

int32_t SherpaMnnOfflineSpeakerDiarizationGetSampleRate(
    const SherpaMnnOfflineSpeakerDiarization *sd) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return 0;
}

void SherpaMnnOfflineSpeakerDiarizationSetConfig(
    const SherpaMnnOfflineSpeakerDiarization *sd,
    const SherpaMnnOfflineSpeakerDiarizationConfig *config) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
}

int32_t SherpaMnnOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return 0;
}

int32_t SherpaMnnOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return 0;
}

const SherpaMnnOfflineSpeakerDiarizationSegment *
SherpaMnnOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

void SherpaMnnOfflineSpeakerDiarizationDestroySegment(
    const SherpaMnnOfflineSpeakerDiarizationSegment *s) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
}

const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcess(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaMnnOfflineSpeakerDiarizationProgressCallback callback,
    void *arg) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcessWithCallbackNoArg(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n,
    SherpaMnnOfflineSpeakerDiarizationProgressCallbackNoArg callback) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

void SherpaMnnOfflineSpeakerDiarizationDestroyResult(
    const SherpaMnnOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
}

#endif

#ifdef __OHOS__

const SherpaMnnOfflineSpeechDenoiser *
SherpaMnnCreateOfflineSpeechDenoiserOHOS(
    const SherpaMnnOfflineSpeechDenoiserConfig *config,
    NativeResourceManager *mgr) {
  auto sd_config = GetOfflineSpeechDenoiserConfig(config);

  SherpaMnnOfflineSpeechDenoiser *sd = new SherpaMnnOfflineSpeechDenoiser;

  sd->impl = std::make_unique<sherpa_mnn::OfflineSpeechDenoiser>(sd_config);

  return sd;
}

const SherpaMnnOnlineRecognizer *SherpaMnnCreateOnlineRecognizerOHOS(
    const SherpaMnnOnlineRecognizerConfig *config,
    NativeResourceManager *mgr) {
  sherpa_mnn::OnlineRecognizerConfig recognizer_config =
      GetOnlineRecognizerConfig(config);

  SherpaMnnOnlineRecognizer *recognizer = new SherpaMnnOnlineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_mnn::OnlineRecognizer>(mgr, recognizer_config);

  return recognizer;
}

const SherpaMnnOfflineRecognizer *SherpaMnnCreateOfflineRecognizerOHOS(
    const SherpaMnnOfflineRecognizerConfig *config,
    NativeResourceManager *mgr) {
  if (mgr == nullptr) {
    return SherpaMnnCreateOfflineRecognizer(config);
  }

  sherpa_mnn::OfflineRecognizerConfig recognizer_config =
      GetOfflineRecognizerConfig(config);

  SherpaMnnOfflineRecognizer *recognizer = new SherpaMnnOfflineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_mnn::OfflineRecognizer>(mgr, recognizer_config);

  return recognizer;
}

const SherpaMnnVoiceActivityDetector *
SherpaMnnCreateVoiceActivityDetectorOHOS(
    const SherpaMnnVadModelConfig *config, float buffer_size_in_seconds,
    NativeResourceManager *mgr) {
  if (mgr == nullptr) {
    return SherpaMnnCreateVoiceActivityDetector(config,
                                                 buffer_size_in_seconds);
  }

  auto vad_config = GetVadModelConfig(config);

  SherpaMnnVoiceActivityDetector *p = new SherpaMnnVoiceActivityDetector;
  p->impl = std::make_unique<sherpa_mnn::VoiceActivityDetector>(
      mgr, vad_config, buffer_size_in_seconds);

  return p;
}

const SherpaMnnSpeakerEmbeddingExtractor *
SherpaMnnCreateSpeakerEmbeddingExtractorOHOS(
    const SherpaMnnSpeakerEmbeddingExtractorConfig *config,
    NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaMnnCreateSpeakerEmbeddingExtractor(config);
  }

  auto c = GetSpeakerEmbeddingExtractorConfig(config);

  auto p = new SherpaMnnSpeakerEmbeddingExtractor;

  p->impl = std::make_unique<sherpa_mnn::SpeakerEmbeddingExtractor>(mgr, c);

  return p;
}

const SherpaMnnKeywordSpotter *SherpaMnnCreateKeywordSpotterOHOS(
    const SherpaMnnKeywordSpotterConfig *config, NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaMnnCreateKeywordSpotter(config);
  }

  auto spotter_config = GetKeywordSpotterConfig(config);

  SherpaMnnKeywordSpotter *spotter = new SherpaMnnKeywordSpotter;

  spotter->impl =
      std::make_unique<sherpa_mnn::KeywordSpotter>(mgr, spotter_config);

  return spotter;
}

#if SHERPA_MNN_ENABLE_TTS == 1
const SherpaMnnOfflineTts *SherpaMnnCreateOfflineTtsOHOS(
    const SherpaMnnOfflineTtsConfig *config, NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaMnnCreateOfflineTts(config);
  }

  auto tts_config = GetOfflineTtsConfig(config);

  SherpaMnnOfflineTts *tts = new SherpaMnnOfflineTts;

  tts->impl = std::make_unique<sherpa_mnn::OfflineTts>(mgr, tts_config);

  return tts;
}
#else
const SherpaMnnOfflineTts *SherpaMnnCreateOfflineTtsOHOS(
    const SherpaMnnOfflineTtsConfig *config, NativeResourceManager *mgr) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}
#endif  // #if SHERPA_MNN_ENABLE_TTS == 1
        //
#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
const SherpaMnnOfflineSpeakerDiarization *
SherpaMnnCreateOfflineSpeakerDiarizationOHOS(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config,
    NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaMnnCreateOfflineSpeakerDiarization(config);
  }

  auto sd_config = GetOfflineSpeakerDiarizationConfig(config);

  SherpaMnnOfflineSpeakerDiarization *sd =
      new SherpaMnnOfflineSpeakerDiarization;

  sd->impl =
      std::make_unique<sherpa_mnn::OfflineSpeakerDiarization>(mgr, sd_config);

  return sd;
}
#else

const SherpaMnnOfflineSpeakerDiarization *
SherpaMnnCreateOfflineSpeakerDiarizationOHOS(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config,
    NativeResourceManager *mgr) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-mnn");
  return nullptr;
}

#endif  // #if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1

#endif  // #ifdef __OHOS__
