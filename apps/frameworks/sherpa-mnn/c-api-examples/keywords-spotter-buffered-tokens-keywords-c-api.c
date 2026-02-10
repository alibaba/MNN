// c-api-examples/keywords-spotter-buffered-tokens-keywords-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Luo Xiao

//
// This file demonstrates how to use keywords spotter with sherpa-onnx's C
// API and with tokens and keywords loaded from buffered strings instead of from
// external files API.
// clang-format off
// 
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile.tar.bz2
// tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile.tar.bz2
// rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

static size_t ReadFile(const char *filename, const char **buffer_out) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Failed to open %s\n", filename);
    return -1;
  }
  fseek(file, 0L, SEEK_END);
  long size = ftell(file);
  rewind(file);
  *buffer_out = malloc(size);
  if (*buffer_out == NULL) {
    fclose(file);
    fprintf(stderr, "Memory error\n");
    return -1;
  }
  size_t read_bytes = fread((void *)*buffer_out, 1, size, file);
  if (read_bytes != size) {
    printf("Errors occured in reading the file %s\n", filename);
    free((void *)*buffer_out);
    *buffer_out = NULL;
    fclose(file);
    return -1;
  }
  fclose(file);
  return read_bytes;
}

int32_t main() {
  const char *wav_filename =
      "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/test_wavs/"
      "6.wav";
  const char *encoder_filename =
      "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
  const char *decoder_filename =
      "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "decoder-epoch-12-avg-2-chunk-16-left-64.onnx";
  const char *joiner_filename =
      "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/"
      "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
  const char *provider = "cpu";
  const char *tokens_filename =
      "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/tokens.txt";
  const char *keywords_filename =
      "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile/test_wavs/"
      "test_keywords.txt";
  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // reading tokens and keywords to buffers
  const char *tokens_buf;
  size_t token_buf_size = ReadFile(tokens_filename, &tokens_buf);
  if (token_buf_size < 1) {
    fprintf(stderr, "Please check your tokens.txt!\n");
    free((void *)tokens_buf);
    return -1;
  }
  const char *keywords_buf;
  size_t keywords_buf_size = ReadFile(keywords_filename, &keywords_buf);
  if (keywords_buf_size < 1) {
    fprintf(stderr, "Please check your keywords.txt!\n");
    free((void *)keywords_buf);
    return -1;
  }

  // Zipformer config
  SherpaMnnOnlineTransducerModelConfig zipformer_config;
  memset(&zipformer_config, 0, sizeof(zipformer_config));
  zipformer_config.encoder = encoder_filename;
  zipformer_config.decoder = decoder_filename;
  zipformer_config.joiner = joiner_filename;

  // Online model config
  SherpaMnnOnlineModelConfig online_model_config;
  memset(&online_model_config, 0, sizeof(online_model_config));
  online_model_config.debug = 1;
  online_model_config.num_threads = 1;
  online_model_config.provider = provider;
  online_model_config.tokens_buf = tokens_buf;
  online_model_config.tokens_buf_size = token_buf_size;
  online_model_config.transducer = zipformer_config;

  // Keywords-spotter config
  SherpaMnnKeywordSpotterConfig keywords_spotter_config;
  memset(&keywords_spotter_config, 0, sizeof(keywords_spotter_config));
  keywords_spotter_config.max_active_paths = 4;
  keywords_spotter_config.keywords_threshold = 0.1;
  keywords_spotter_config.keywords_score = 3.0;
  keywords_spotter_config.model_config = online_model_config;
  keywords_spotter_config.keywords_buf = keywords_buf;
  keywords_spotter_config.keywords_buf_size = keywords_buf_size;

  const SherpaMnnKeywordSpotter *keywords_spotter =
      SherpaMnnCreateKeywordSpotter(&keywords_spotter_config);

  free((void *)tokens_buf);
  tokens_buf = NULL;
  free((void *)keywords_buf);
  keywords_buf = NULL;

  if (keywords_spotter == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaMnnFreeWave(wave);
    return -1;
  }

  const SherpaMnnOnlineStream *stream =
      SherpaMnnCreateKeywordStream(keywords_spotter);

  const SherpaMnnDisplay *display = SherpaMnnCreateDisplay(50);
  int32_t segment_id = 0;

// simulate streaming. You can choose an arbitrary N
#define N 3200

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  int32_t k = 0;
  while (k < wave->num_samples) {
    int32_t start = k;
    int32_t end =
        (start + N > wave->num_samples) ? wave->num_samples : (start + N);
    k += N;

    SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + start, end - start);
    while (SherpaMnnIsKeywordStreamReady(keywords_spotter, stream)) {
      SherpaMnnDecodeKeywordStream(keywords_spotter, stream);
    }

    const SherpaMnnKeywordResult *r =
        SherpaMnnGetKeywordResult(keywords_spotter, stream);

    if (strlen(r->keyword)) {
      SherpaMnnPrint(display, segment_id, r->keyword);
    }

    SherpaMnnDestroyKeywordResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaMnnFreeWave(wave);

  SherpaMnnOnlineStreamInputFinished(stream);
  while (SherpaMnnIsKeywordStreamReady(keywords_spotter, stream)) {
    SherpaMnnDecodeKeywordStream(keywords_spotter, stream);
  }

  const SherpaMnnKeywordResult *r =
      SherpaMnnGetKeywordResult(keywords_spotter, stream);

  if (strlen(r->keyword)) {
    SherpaMnnPrint(display, segment_id, r->keyword);
  }

  SherpaMnnDestroyKeywordResult(r);

  SherpaMnnDestroyDisplay(display);
  SherpaMnnDestroyOnlineStream(stream);
  SherpaMnnDestroyKeywordSpotter(keywords_spotter);
  fprintf(stderr, "\n");

  return 0;
}
