// cxx-api-examples/matcha-tts-en-cxx-api.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// This file shows how to use sherpa-onnx CXX API
// for Chinese TTS with MatchaTTS.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

./matcha-tts-en-cxx-api

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

  config.model.matcha.acoustic_model =
      "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx";

  config.model.matcha.vocoder = "./hifigan_v2.onnx";

  config.model.matcha.tokens = "./matcha-icefall-en_US-ljspeech/tokens.txt";

  config.model.matcha.data_dir =
      "./matcha-icefall-en_US-ljspeech/espeak-ng-data";

  config.model.num_threads = 1;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  std::string filename = "./generated-matcha-en-cxx.wav";
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
