// cxx-api-examples/speech-enhancement-gtcrn-cxx-api.cc
//
// Copyright (c)  2025  Xiaomi Corporation
//
// We assume you have pre-downloaded model
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
//
// An example command to download
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
*/
// clang-format on
#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-mnn/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_mnn::cxx;  // NOLINT

  OfflineSpeechDenoiserConfig config;
  std::string wav_filename = "./inp_16k.wav";
  std::string out_wave_filename = "./enhanced_16k.wav";

  config.model.gtcrn.model = "./gtcrn_simple.onnx";

  auto sd = OfflineSpeechDenoiser::Create(config);
  if (!sd.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wav_filename << "'\n";
    return -1;
  }

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();
  auto denoised =
      sd.Run(wave.samples.data(), wave.samples.size(), wave.sample_rate);
  const auto end = std::chrono::steady_clock::now();
  std::cout << "Done\n";

  WriteWave(out_wave_filename, {denoised.samples, denoised.sample_rate});

  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  std::cout << "Saved to " << out_wave_filename << "\n";
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);
}
