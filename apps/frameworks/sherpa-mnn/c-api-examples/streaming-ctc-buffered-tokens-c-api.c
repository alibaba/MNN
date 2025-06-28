// c-api-examples/streaming-ctc-buffered-tokens-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Luo Xiao

//
// This file demonstrates how to use streaming Zipformer2 Ctc with sherpa-onnx's
// C API and with tokens loaded from buffered strings instead of
// from external files API.
// clang-format off
// 
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
// rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
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
      "sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/"
      "DEV_T0000000000.wav";
  const char *model_filename =
      "sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/"
      "ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx";
  const char *tokens_filename =
      "sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt";
  const char *provider = "cpu";

  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // reading tokens to buffers
  const char *tokens_buf;
  size_t token_buf_size = ReadFile(tokens_filename, &tokens_buf);
  if (token_buf_size < 1) {
    fprintf(stderr, "Please check your tokens.txt!\n");
    free((void *)tokens_buf);
    return -1;
  }

  // Zipformer2Ctc config
  SherpaMnnOnlineZipformer2CtcModelConfig zipformer2_ctc_config;
  memset(&zipformer2_ctc_config, 0, sizeof(zipformer2_ctc_config));
  zipformer2_ctc_config.model = model_filename;

  // Online model config
  SherpaMnnOnlineModelConfig online_model_config;
  memset(&online_model_config, 0, sizeof(online_model_config));
  online_model_config.debug = 1;
  online_model_config.num_threads = 1;
  online_model_config.provider = provider;
  online_model_config.tokens_buf = tokens_buf;
  online_model_config.tokens_buf_size = token_buf_size;
  online_model_config.zipformer2_ctc = zipformer2_ctc_config;

  // Recognizer config
  SherpaMnnOnlineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = online_model_config;

  const SherpaMnnOnlineRecognizer *recognizer =
      SherpaMnnCreateOnlineRecognizer(&recognizer_config);

  free((void *)tokens_buf);
  tokens_buf = NULL;

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaMnnFreeWave(wave);
    return -1;
  }

  const SherpaMnnOnlineStream *stream =
      SherpaMnnCreateOnlineStream(recognizer);

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
    while (SherpaMnnIsOnlineStreamReady(recognizer, stream)) {
      SherpaMnnDecodeOnlineStream(recognizer, stream);
    }

    const SherpaMnnOnlineRecognizerResult *r =
        SherpaMnnGetOnlineStreamResult(recognizer, stream);

    if (strlen(r->text)) {
      SherpaMnnPrint(display, segment_id, r->text);
    }

    if (SherpaMnnOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++segment_id;
      }
      SherpaMnnOnlineStreamReset(recognizer, stream);
    }

    SherpaMnnDestroyOnlineRecognizerResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaMnnFreeWave(wave);

  SherpaMnnOnlineStreamInputFinished(stream);
  while (SherpaMnnIsOnlineStreamReady(recognizer, stream)) {
    SherpaMnnDecodeOnlineStream(recognizer, stream);
  }

  const SherpaMnnOnlineRecognizerResult *r =
      SherpaMnnGetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaMnnPrint(display, segment_id, r->text);
  }

  SherpaMnnDestroyOnlineRecognizerResult(r);

  SherpaMnnDestroyDisplay(display);
  SherpaMnnDestroyOnlineStream(stream);
  SherpaMnnDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
