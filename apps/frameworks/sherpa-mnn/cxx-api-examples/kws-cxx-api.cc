// cxx-api-examples/kws-cxx-api.cc
//
// Copyright (c)  2025  Xiaomi Corporation
//
// This file demonstrates how to use keywords spotter with sherpa-onnx's C
// clang-format off
//
// Usage
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile.tar.bz2
// tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile.tar.bz2
// rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile.tar.bz2
//
// ./kws-cxx-api
//
// clang-format on
#include <array>
#include <iostream>

#include "sherpa-mnn/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_mnn::cxx;  // NOLINT

  KeywordSpotterConfig config;
  config.model_config.transducer.encoder =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";

  config.model_config.transducer.decoder =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "decoder-epoch-12-avg-2-chunk-16-left-64.onnx";

  config.model_config.transducer.joiner =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx";

  config.model_config.tokens =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "tokens.txt";

  config.model_config.provider = "cpu";
  config.model_config.num_threads = 1;
  config.model_config.debug = 1;

  config.keywords_file =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "test_wavs/test_keywords.txt";

  KeywordSpotter kws = KeywordSpotter::Create(config);
  if (!kws.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  std::cout
      << "--Test pre-defined keywords from test_wavs/test_keywords.txt--\n";

  std::string wave_filename =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "test_wavs/3.wav";

  std::array<float, 8000> tail_paddings = {0};  // 0.5 seconds

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  OnlineStream stream = kws.CreateStream();
  if (!stream.Get()) {
    std::cerr << "Failed to create stream\n";
    return -1;
  }

  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());

  stream.AcceptWaveform(wave.sample_rate, tail_paddings.data(),
                        tail_paddings.size());
  stream.InputFinished();

  while (kws.IsReady(&stream)) {
    kws.Decode(&stream);
    auto r = kws.GetResult(&stream);
    if (!r.keyword.empty()) {
      std::cout << "Detected keyword: " << r.json << "\n";

      // Remember to reset the keyword stream right after a keyword is detected
      kws.Reset(&stream);
    }
  }

  // --------------------------------------------------------------------------

  std::cout << "--Use pre-defined keywords + add a new keyword--\n";

  stream = kws.CreateStream("y ǎn y uán @演员");

  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());

  stream.AcceptWaveform(wave.sample_rate, tail_paddings.data(),
                        tail_paddings.size());
  stream.InputFinished();

  while (kws.IsReady(&stream)) {
    kws.Decode(&stream);
    auto r = kws.GetResult(&stream);
    if (!r.keyword.empty()) {
      std::cout << "Detected keyword: " << r.json << "\n";

      // Remember to reset the keyword stream right after a keyword is detected
      kws.Reset(&stream);
    }
  }

  // --------------------------------------------------------------------------

  std::cout << "--Use pre-defined keywords + add two new keywords--\n";

  stream = kws.CreateStream("y ǎn y uán @演员/zh ī m íng @知名");

  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());

  stream.AcceptWaveform(wave.sample_rate, tail_paddings.data(),
                        tail_paddings.size());
  stream.InputFinished();

  while (kws.IsReady(&stream)) {
    kws.Decode(&stream);
    auto r = kws.GetResult(&stream);
    if (!r.keyword.empty()) {
      std::cout << "Detected keyword: " << r.json << "\n";

      // Remember to reset the keyword stream right after a keyword is detected
      kws.Reset(&stream);
    }
  }
  return 0;
}
