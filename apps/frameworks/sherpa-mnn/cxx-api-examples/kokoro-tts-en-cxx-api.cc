// cxx-api-examples/kokoro-tts-en-cxx-api.c
//
// Copyright (c)  2025  Xiaomi Corporation

// This file shows how to use sherpa-onnx CXX API
// for English TTS with Kokoro.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

./kokoro-tts-en-cxx-api

 */
// clang-format on

#include <string>

#include "sherpa-mnn/c-api/cxx-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_mnn::cxx;  // NOLINT
  OfflineTtsConfig config;

  config.model.kokoro.model = "./kokoro-en-v0_19/model.onnx";
  config.model.kokoro.voices = "./kokoro-en-v0_19/voices.bin";
  config.model.kokoro.tokens = "./kokoro-en-v0_19/tokens.txt";
  config.model.kokoro.data_dir = "./kokoro-en-v0_19/espeak-ng-data";

  config.model.num_threads = 2;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  std::string filename = "./generated-kokoro-en-cxx.wav";
  std::string text =
      "Today as always, men fall into two groups: slaves and free men. Whoever "
      "does not have two-thirds of his day for himself, is a slave, whatever "
      "he may be: a statesman, a businessman, an official, or a scholar. "
      "Friends fell out often because life was changing so fast. The easiest "
      "thing in the world was to lose touch with someone.";

  auto tts = OfflineTts::Create(config);
  int32_t sid = 0;
  float speed = 1.0;  // larger -> faster in speech speed

#if 0
  // If you don't want to use a callback, then please enable this branch
  GeneratedAudio audio = tts.Generate(text, sid, speed);
#else
  GeneratedAudio audio = tts.Generate(text, sid, speed, ProgressCallback);
#endif

  WriteWave(filename, {audio.samples, audio.sample_rate});

  fprintf(stderr, "Input text is: %s\n", text.c_str());
  fprintf(stderr, "Speaker ID is is: %d\n", sid);
  fprintf(stderr, "Saved to: %s\n", filename.c_str());

  return 0;
}
