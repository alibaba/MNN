// c-api-examples/sense-voice-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

//
// This file demonstrates how to use SenseVoice with sherpa-onnx's C API.
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
// tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
// rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  const char *wav_filename =
      "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav";
  const char *model_filename =
      "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx";
  const char *tokens_filename =
      "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt";
  const char *language = "auto";
  const char *provider = "cpu";
  int32_t use_inverse_text_normalization = 1;

  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaMnnOfflineSenseVoiceModelConfig sense_voice_config;
  memset(&sense_voice_config, 0, sizeof(sense_voice_config));
  sense_voice_config.model = model_filename;
  sense_voice_config.language = language;
  sense_voice_config.use_itn = use_inverse_text_normalization;

  // Offline model config
  SherpaMnnOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));
  offline_model_config.debug = 1;
  offline_model_config.num_threads = 1;
  offline_model_config.provider = provider;
  offline_model_config.tokens = tokens_filename;
  offline_model_config.sense_voice = sense_voice_config;

  // Recognizer config
  SherpaMnnOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = offline_model_config;

  const SherpaMnnOfflineRecognizer *recognizer =
      SherpaMnnCreateOfflineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaMnnFreeWave(wave);
    return -1;
  }

  const SherpaMnnOfflineStream *stream =
      SherpaMnnCreateOfflineStream(recognizer);

  SherpaMnnAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);
  SherpaMnnDecodeOfflineStream(recognizer, stream);
  const SherpaMnnOfflineRecognizerResult *result =
      SherpaMnnGetOfflineStreamResult(stream);

  fprintf(stderr, "Decoded text: %s\n", result->text);

  SherpaMnnDestroyOfflineRecognizerResult(result);
  SherpaMnnDestroyOfflineStream(stream);
  SherpaMnnDestroyOfflineRecognizer(recognizer);
  SherpaMnnFreeWave(wave);

  return 0;
}
