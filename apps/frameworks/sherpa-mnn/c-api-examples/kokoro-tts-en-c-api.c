// c-api-examples/kokoro-tts-en-c-api.c
//
// Copyright (c)  2025  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// for English TTS with Kokoro.
//
// clang-format off
/*
Usage


wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

./kokoro-tts-en-c-api

 */
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  SherpaMnnOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.kokoro.model = "./kokoro-en-v0_19/model.onnx";
  config.model.kokoro.voices = "./kokoro-en-v0_19/voices.bin";
  config.model.kokoro.tokens = "./kokoro-en-v0_19/tokens.txt";
  config.model.kokoro.data_dir = "./kokoro-en-v0_19/espeak-ng-data";

  config.model.num_threads = 2;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  const char *filename = "./generated-kokoro-en.wav";
  const char *text =
      "Today as always, men fall into two groups: slaves and free men. Whoever "
      "does not have two-thirds of his day for himself, is a slave, whatever "
      "he may be: a statesman, a businessman, an official, or a scholar. "
      "Friends fell out often because life was changing so fast. The easiest "
      "thing in the world was to lose touch with someone.";

  const SherpaMnnOfflineTts *tts = SherpaMnnCreateOfflineTts(&config);
  // mapping of sid to voice name
  // 0->af, 1->af_bella, 2->af_nicole, 3->af_sarah, 4->af_sky, 5->am_adam
  // 6->am_michael, 7->bf_emma, 8->bf_isabella, 9->bm_george, 10->bm_lewis
  int32_t sid = 0;
  float speed = 1.0;  // larger -> faster in speech speed

#if 0
  // If you don't want to use a callback, then please enable this branch
  const SherpaMnnGeneratedAudio *audio =
      SherpaMnnOfflineTtsGenerate(tts, text, sid, speed);
#else
  const SherpaMnnGeneratedAudio *audio =
      SherpaMnnOfflineTtsGenerateWithProgressCallback(tts, text, sid, speed,
                                                       ProgressCallback);
#endif

  SherpaMnnWriteWave(audio->samples, audio->n, audio->sample_rate, filename);

  SherpaMnnDestroyOfflineTtsGeneratedAudio(audio);
  SherpaMnnDestroyOfflineTts(tts);

  fprintf(stderr, "Input text is: %s\n", text);
  fprintf(stderr, "Speaker ID is is: %d\n", sid);
  fprintf(stderr, "Saved to: %s\n", filename);

  return 0;
}
