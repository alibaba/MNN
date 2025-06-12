// c-api-examples/vad-moonshine-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

//
// This file demonstrates how to use VAD + Moonshine with sherpa-onnx's C API.
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
// tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
// rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  const char *wav_filename = "./Obama.wav";
  const char *vad_filename = "./silero_vad.onnx";

  const char *preprocessor =
      "./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx";
  const char *encoder = "./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx";
  const char *uncached_decoder =
      "./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx";
  const char *cached_decoder =
      "./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx";
  const char *tokens = "./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt";

  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  if (wave->sample_rate != 16000) {
    fprintf(stderr, "Expect the sample rate to be 16000. Given: %d\n",
            wave->sample_rate);
    SherpaMnnFreeWave(wave);
    return -1;
  }

  // Offline model config
  SherpaMnnOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));
  offline_model_config.debug = 0;
  offline_model_config.num_threads = 1;
  offline_model_config.provider = "cpu";
  offline_model_config.tokens = tokens;
  offline_model_config.moonshine.preprocessor = preprocessor;
  offline_model_config.moonshine.encoder = encoder;
  offline_model_config.moonshine.uncached_decoder = uncached_decoder;
  offline_model_config.moonshine.cached_decoder = cached_decoder;

  // Recognizer config
  SherpaMnnOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = offline_model_config;

  const SherpaMnnOfflineRecognizer *recognizer =
      SherpaMnnCreateOfflineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your recognizer config!\n");
    SherpaMnnFreeWave(wave);
    return -1;
  }

  SherpaMnnVadModelConfig vadConfig;
  memset(&vadConfig, 0, sizeof(vadConfig));
  vadConfig.silero_vad.model = vad_filename;
  vadConfig.silero_vad.threshold = 0.5;
  vadConfig.silero_vad.min_silence_duration = 0.5;
  vadConfig.silero_vad.min_speech_duration = 0.5;
  vadConfig.silero_vad.max_speech_duration = 10;
  vadConfig.silero_vad.window_size = 512;
  vadConfig.sample_rate = 16000;
  vadConfig.num_threads = 1;
  vadConfig.debug = 1;

  SherpaMnnVoiceActivityDetector *vad =
      SherpaMnnCreateVoiceActivityDetector(&vadConfig, 30);

  if (vad == NULL) {
    fprintf(stderr, "Please check your recognizer config!\n");
    SherpaMnnFreeWave(wave);
    SherpaMnnDestroyOfflineRecognizer(recognizer);
    return -1;
  }

  int32_t window_size = vadConfig.silero_vad.window_size;
  int32_t i = 0;
  int is_eof = 0;

  while (!is_eof) {
    if (i + window_size < wave->num_samples) {
      SherpaMnnVoiceActivityDetectorAcceptWaveform(vad, wave->samples + i,
                                                    window_size);
    } else {
      SherpaMnnVoiceActivityDetectorFlush(vad);
      is_eof = 1;
    }
    while (!SherpaMnnVoiceActivityDetectorEmpty(vad)) {
      const SherpaMnnSpeechSegment *segment =
          SherpaMnnVoiceActivityDetectorFront(vad);

      const SherpaMnnOfflineStream *stream =
          SherpaMnnCreateOfflineStream(recognizer);

      SherpaMnnAcceptWaveformOffline(stream, wave->sample_rate,
                                      segment->samples, segment->n);

      SherpaMnnDecodeOfflineStream(recognizer, stream);

      const SherpaMnnOfflineRecognizerResult *result =
          SherpaMnnGetOfflineStreamResult(stream);

      float start = segment->start / 16000.0f;
      float duration = segment->n / 16000.0f;
      float stop = start + duration;

      fprintf(stderr, "%.3f -- %.3f: %s\n", start, stop, result->text);

      SherpaMnnDestroyOfflineRecognizerResult(result);
      SherpaMnnDestroyOfflineStream(stream);

      SherpaMnnDestroySpeechSegment(segment);
      SherpaMnnVoiceActivityDetectorPop(vad);
    }
    i += window_size;
  }

  SherpaMnnDestroyOfflineRecognizer(recognizer);
  SherpaMnnDestroyVoiceActivityDetector(vad);
  SherpaMnnFreeWave(wave);

  return 0;
}
