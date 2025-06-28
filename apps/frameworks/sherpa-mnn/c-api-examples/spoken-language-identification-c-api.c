// c-api-examples/spoken-language-identification-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

// We assume you have pre-downloaded the whisper multi-lingual models
// from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
// An example command to download the "tiny" whisper model is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
// tar xvf sherpa-onnx-whisper-tiny.tar.bz2
// rm sherpa-onnx-whisper-tiny.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  SherpaMnnSpokenLanguageIdentificationConfig config;

  memset(&config, 0, sizeof(config));

  config.whisper.encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx";
  config.whisper.decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx";
  config.num_threads = 1;
  config.debug = 1;
  config.provider = "cpu";

  const SherpaMnnSpokenLanguageIdentification *slid =
      SherpaMnnCreateSpokenLanguageIdentification(&config);
  if (!slid) {
    fprintf(stderr, "Failed to create spoken language identifier");
    return -1;
  }

  // You can find more test waves from
  // https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/tree/main/test_wavs
  const char *wav_filename = "./sherpa-onnx-whisper-tiny/test_wavs/0.wav";
  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaMnnOfflineStream *stream =
      SherpaMnnSpokenLanguageIdentificationCreateOfflineStream(slid);

  SherpaMnnAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);

  const SherpaMnnSpokenLanguageIdentificationResult *result =
      SherpaMnnSpokenLanguageIdentificationCompute(slid, stream);

  fprintf(stderr, "wav_filename: %s\n", wav_filename);
  fprintf(stderr, "Detected language: %s\n", result->lang);

  SherpaMnnDestroySpokenLanguageIdentificationResult(result);
  SherpaMnnDestroyOfflineStream(stream);
  SherpaMnnFreeWave(wave);
  SherpaMnnDestroySpokenLanguageIdentification(slid);

  return 0;
}
