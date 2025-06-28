// c-api-examples/zipformer-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

//
// This file demonstrates how to use non-streaming Zipformer with sherpa-onnx's
// C API.
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
// tar xvf sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
// rm sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  const char *wav_filename =
      "sherpa-onnx-zipformer-small-en-2023-06-26/test_wavs/0.wav";
  const char *encoder_filename =
      "sherpa-onnx-zipformer-small-en-2023-06-26/encoder-epoch-99-avg-1.onnx";
  const char *decoder_filename =
      "sherpa-onnx-zipformer-small-en-2023-06-26/decoder-epoch-99-avg-1.onnx";
  const char *joiner_filename =
      "sherpa-onnx-zipformer-small-en-2023-06-26/joiner-epoch-99-avg-1.onnx";
  const char *tokens_filename =
      "sherpa-onnx-zipformer-small-en-2023-06-26/tokens.txt";
  const char *provider = "cpu";

  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Zipformer config
  SherpaMnnOfflineTransducerModelConfig zipformer_config;
  memset(&zipformer_config, 0, sizeof(zipformer_config));
  zipformer_config.encoder = encoder_filename;
  zipformer_config.decoder = decoder_filename;
  zipformer_config.joiner = joiner_filename;

  // Offline model config
  SherpaMnnOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));
  offline_model_config.debug = 1;
  offline_model_config.num_threads = 1;
  offline_model_config.provider = provider;
  offline_model_config.tokens = tokens_filename;
  offline_model_config.transducer = zipformer_config;

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
