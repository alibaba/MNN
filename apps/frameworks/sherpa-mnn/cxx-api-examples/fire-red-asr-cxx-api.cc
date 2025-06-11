// cxx-api-examples/fire-red-asr-cxx-api.cc
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use FireRedAsr AED with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
// tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
// rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-mnn/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_mnn::cxx;  // NOLINT
  OfflineRecognizerConfig config;

  config.model_config.fire_red_asr.encoder =
      "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx";
  config.model_config.fire_red_asr.decoder =
      "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx";
  config.model_config.tokens =
      "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt";

  config.model_config.num_threads = 1;

  std::cout << "Loading model\n";
  OfflineRecognizer recongizer = OfflineRecognizer::Create(config);
  if (!recongizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav";
  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OfflineStream stream = recongizer.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());

  recongizer.Decode(&stream);

  OfflineRecognizerResult result = recongizer.GetResult(&stream);

  const auto end = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  std::cout << "text: " << result.text << "\n";
  printf("Number of threads: %d\n", config.model_config.num_threads);
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);

  return 0;
}
