// sherpa-mnn/csrc/sherpa-mnn-offline-denoiser.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include <stdio.h>

#include <chrono>  // NOLINT

#include "sherpa-mnn/csrc/offline-speech-denoiser.h"
#include "sherpa-mnn/csrc/wave-reader.h"
#include "sherpa-mnn/csrc/wave-writer.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Non-stremaing speech denoising with sherpa-mnn.

Please visit
https://github.com/k2-fsa/sherpa-mnn/releases/tag/speech-enhancement-models
to download models.

Usage:

(1) Use gtcrn models

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/speech-enhancement-models/gtcrn_simple.onnx
./bin/sherpa-mnn-offline-denoiser \
  --speech-denoiser-gtcrn-model=gtcrn_simple.onnx \
  --input-wav input.wav \
  --output-wav output_16k.wav
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OfflineSpeechDenoiserConfig config;
  std::string input_wave;
  std::string output_wave;

  config.Register(&po);
  po.Register("input-wav", &input_wave, "Path to input wav.");
  po.Register("output-wav", &output_wave, "Path to output wav");

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    fprintf(stderr, "Please don't give positional arguments\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (input_wave.empty()) {
    fprintf(stderr, "Please provide --input-wav\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (output_wave.empty()) {
    fprintf(stderr, "Please provide --output-wav\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  sherpa_mnn::OfflineSpeechDenoiser denoiser(config);
  int32_t sampling_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_mnn::ReadWave(input_wave, &sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", input_wave.c_str());
    return -1;
  }

  fprintf(stderr, "Started\n");
  const auto begin = std::chrono::steady_clock::now();
  auto result = denoiser.Run(samples.data(), samples.size(), sampling_rate);
  const auto end = std::chrono::steady_clock::now();

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "Done\n");
  is_ok = sherpa_mnn::WriteWave(output_wave, result.sample_rate,
                                 result.samples.data(), result.samples.size());
  if (is_ok) {
    fprintf(stderr, "Saved to %s\n", output_wave.c_str());
  } else {
    fprintf(stderr, "Failed to save to %s\n", output_wave.c_str());
  }

  float duration = samples.size() / static_cast<float>(sampling_rate);
  fprintf(stderr, "num threads: %d\n", config.model.num_threads);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);
}
