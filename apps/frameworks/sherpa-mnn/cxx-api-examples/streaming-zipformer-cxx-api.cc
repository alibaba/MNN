// cxx-api-examples/streaming-zipformer-cxx-api.cc
// Copyright (c)  2024  Xiaomi Corporation

//
// This file demonstrates how to use streaming Zipformer
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
// rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-mnn/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_mnn::cxx;  // NOLINT
  OnlineRecognizerConfig config;

  // please see
  // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
  config.model_config.transducer.encoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
      "encoder-epoch-99-avg-1.int8.onnx";

  // Note: We recommend not using int8.onnx for the decoder.
  config.model_config.transducer.decoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
      "decoder-epoch-99-avg-1.onnx";

  config.model_config.transducer.joiner =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
      "joiner-epoch-99-avg-1.int8.onnx";

  config.model_config.tokens =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";

  config.model_config.num_threads = 1;

  std::cout << "Loading model\n";
  OnlineRecognizer recongizer = OnlineRecognizer::Create(config);
  if (!recongizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/"
      "0.wav";
  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recongizer.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());
  stream.InputFinished();

  while (recongizer.IsReady(&stream)) {
    recongizer.Decode(&stream);
  }

  OnlineRecognizerResult result = recongizer.GetResult(&stream);

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
