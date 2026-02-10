// c-api-examples/kws-c-api.c
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
// ./kws-c-api
//
// clang-format on
#include <stdio.h>
#include <stdlib.h>  // exit
#include <string.h>  // memset

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  SherpaMnnKeywordSpotterConfig config;

  memset(&config, 0, sizeof(config));
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

  const SherpaMnnKeywordSpotter *kws = SherpaMnnCreateKeywordSpotter(&config);
  if (!kws) {
    fprintf(stderr, "Please check your config");
    exit(-1);
  }

  fprintf(stderr,
          "--Test pre-defined keywords from test_wavs/test_keywords.txt--\n");

  const char *wav_filename =
      "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "test_wavs/3.wav";

  float tail_paddings[8000] = {0};  // 0.5 seconds

  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    exit(-1);
  }

  const SherpaMnnOnlineStream *stream = SherpaMnnCreateKeywordStream(kws);
  if (!stream) {
    fprintf(stderr, "Failed to create stream\n");
    exit(-1);
  }

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples,
                                       wave->num_samples);

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       sizeof(tail_paddings) / sizeof(float));
  SherpaMnnOnlineStreamInputFinished(stream);
  while (SherpaMnnIsKeywordStreamReady(kws, stream)) {
    SherpaMnnDecodeKeywordStream(kws, stream);
    const SherpaMnnKeywordResult *r = SherpaMnnGetKeywordResult(kws, stream);
    if (r && r->json && strlen(r->keyword)) {
      fprintf(stderr, "Detected keyword: %s\n", r->json);

      // Remember to reset the keyword stream right after a keyword is detected
      SherpaMnnResetKeywordStream(kws, stream);
    }
    SherpaMnnDestroyKeywordResult(r);
  }
  SherpaMnnDestroyOnlineStream(stream);

  // --------------------------------------------------------------------------

  fprintf(stderr, "--Use pre-defined keywords + add a new keyword--\n");

  stream = SherpaMnnCreateKeywordStreamWithKeywords(kws, "y ǎn y uán @演员");

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples,
                                       wave->num_samples);

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       sizeof(tail_paddings) / sizeof(float));
  SherpaMnnOnlineStreamInputFinished(stream);
  while (SherpaMnnIsKeywordStreamReady(kws, stream)) {
    SherpaMnnDecodeKeywordStream(kws, stream);
    const SherpaMnnKeywordResult *r = SherpaMnnGetKeywordResult(kws, stream);
    if (r && r->json && strlen(r->keyword)) {
      fprintf(stderr, "Detected keyword: %s\n", r->json);

      // Remember to reset the keyword stream
      SherpaMnnResetKeywordStream(kws, stream);
    }
    SherpaMnnDestroyKeywordResult(r);
  }
  SherpaMnnDestroyOnlineStream(stream);

  // --------------------------------------------------------------------------

  fprintf(stderr, "--Use pre-defined keywords + add two new keywords--\n");

  stream = SherpaMnnCreateKeywordStreamWithKeywords(
      kws, "y ǎn y uán @演员/zh ī m íng @知名");

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples,
                                       wave->num_samples);

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       sizeof(tail_paddings) / sizeof(float));
  SherpaMnnOnlineStreamInputFinished(stream);
  while (SherpaMnnIsKeywordStreamReady(kws, stream)) {
    SherpaMnnDecodeKeywordStream(kws, stream);
    const SherpaMnnKeywordResult *r = SherpaMnnGetKeywordResult(kws, stream);
    if (r && r->json && strlen(r->keyword)) {
      fprintf(stderr, "Detected keyword: %s\n", r->json);

      // Remember to reset the keyword stream
      SherpaMnnResetKeywordStream(kws, stream);
    }
    SherpaMnnDestroyKeywordResult(r);
  }
  SherpaMnnDestroyOnlineStream(stream);

  SherpaMnnFreeWave(wave);
  SherpaMnnDestroyKeywordSpotter(kws);

  return 0;
}
