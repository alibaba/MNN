// c-api-examples/streaming-zipformer-buffered-tokens-hotwords-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Luo Xiao

//
// This file demonstrates how to use streaming Zipformer with sherpa-onnx's C
// API and with tokens and hotwords loaded from buffered strings instead of from
// external files API.
// clang-format off
// 
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
// rm sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
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
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav";
  const char *encoder_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "encoder-epoch-99-avg-1.onnx";
  const char *decoder_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "decoder-epoch-99-avg-1.onnx";
  const char *joiner_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "joiner-epoch-99-avg-1.onnx";
  const char *provider = "cpu";
  const char *modeling_unit = "bpe";
  const char *tokens_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt";
  const char *hotwords_filename =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/hotwords.txt";
  const char *bpe_vocab =
      "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/"
      "bpe.vocab";
  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // reading tokens and hotwords to buffers
  const char *tokens_buf;
  size_t token_buf_size = ReadFile(tokens_filename, &tokens_buf);
  if (token_buf_size < 1) {
    fprintf(stderr, "Please check your tokens.txt!\n");
    free((void *)tokens_buf);
    return -1;
  }
  const char *hotwords_buf;
  size_t hotwords_buf_size = ReadFile(hotwords_filename, &hotwords_buf);
  if (hotwords_buf_size < 1) {
    fprintf(stderr, "Please check your hotwords.txt!\n");
    free((void *)hotwords_buf);
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

  // Recognizer config
  SherpaMnnOnlineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "modified_beam_search";
  recognizer_config.model_config = online_model_config;
  recognizer_config.hotwords_buf = hotwords_buf;
  recognizer_config.hotwords_buf_size = hotwords_buf_size;

  const SherpaMnnOnlineRecognizer *recognizer =
      SherpaMnnCreateOnlineRecognizer(&recognizer_config);

  free((void *)tokens_buf);
  tokens_buf = NULL;
  free((void *)hotwords_buf);
  hotwords_buf = NULL;

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
